import transformers
import torch
import contextlib
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import colossalai
from tqdm import tqdm
from lm_eval.base import BaseLM
from lm_eval import utils
from colossalai.core import global_context as gpc
from colossalai.utils import is_using_pp
from colossalai.utils.checkpointing import load_checkpoint
from model_zoo.gpt.gpt import GPTLMLoss
from colossalai.logging import get_dist_logger
from colossalai.communication import all_gather
from colossalai.context import ParallelMode
from colossalai.engine.schedule import NonPipelineSchedule
from colossalai.zero.init_ctx import ZeroInitContext
from model.hf_model import build_model, load_config_from_colossalai

# import os
# import sys
# sys.path.append(os.environ['PROJECT'])
from model import *

class GPT2LM(BaseLM):

    def __init__(self, device='cuda', revision='main', subfolder=None, tokenizer=None):
        super().__init__()

        assert isinstance(device, str)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.tokenizer = transformers.GPT2Tokenizer
        token_type = gpc.config.model_type.split('-')[0]
        assert token_type in ['gpt2', 'gpt3'], f'The tokenizer type {token_type} is unknown.'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = gpc.config.BATCH_SIZE * \
            gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)

        self._schedule = None
        logger = get_dist_logger()

        logger.info('Build model', ranks=[0])
        use_zero3 = hasattr(gpc.config, 'zero')

        ctx = contextlib.nullcontext()
        if use_zero3:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True
                                  )
        with ctx:
            model = gpc.config.model.pop('type')(**gpc.config.model)

        self.gpt = model
        self.criterion = getattr(gpc.config, 'loss_fn', None)
        if self.criterion is not None:
            self.criterion = self.criterion.type()
        else:
            self.criterion = GPTLMLoss()

        logger.info('Build optimizer', ranks=[0])
        self.optimizer = gpc.config.optimizer.pop('type')(
            model.parameters(), **gpc.config.optimizer)

        self.engine, _, _, _ = colossalai.initialize(self.gpt, self.optimizer, self.criterion)
        self.engine.eval()

        assert getattr(gpc.config, "checkpoint_path", None) is not None, \
            f'Please give the checkpoint path in configuration.'
        last_epoch = load_checkpoint(gpc.config.checkpoint_path, self.gpt, self.optimizer, strict=False)
        logger.info(f'checkpoint loading finised, resume from {last_epoch} epoch', ranks=[0])

        logger.info(f'Init done, global batch size = {self.batch_size_per_gpu}', ranks=[0])
        tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)

        self._schedule = NonPipelineSchedule()
        self._schedule.pre_processing(self.engine)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return gpc.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    @property
    def schedule(self):
        return self._schedule

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, data_loader):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        data_iter = iter(data_loader)
        with torch.no_grad():
            parallel_logits, _, _ = self.schedule.forward_backward_step(
                self.engine,
                data_iter,
                forward_only=True,
                return_loss=True,
                return_output_label=True,
            )

        return parallel_logits[:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        # return self.gpt2.generate(
        #     context,
        #     max_length=max_length,
        #     eos_token_id=eos_token_id,
        #     do_sample=False
        # )
        raise NotImplementedError()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # TODO: automatic (variable) batch size detection for vectorization
        reord = utils.Reorderer(requests, _collate)
        request_dataset = utils.Request_Context_Dataset(reord.get_reordered(), self.max_length)
        request_dataloader = DataLoader(request_dataset, batch_size=gpc.config.BATCH_SIZE, shuffle=False, sampler=None,
                   batch_sampler=None, collate_fn=utils.collate_fn,
                   pin_memory=False, drop_last=False)
        # import pdb; pdb.set_trace()
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), gpc.config.BATCH_SIZE):
            cont_toks_list = []
            inplens = []
            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length+1):][:-1],
                    dtype=torch.long
                ).to(self.device)

                inplen, = inp.shape
                cont = continuation_enc
                cont_toks_list.append(cont)
                inplens.append(inplen)

            multi_logits = F.log_softmax(self._model_call(request_dataloader), dim=-1).cpu()  # [batch, padding_length, vocab]
            for (cache_key, _, _), logits, inplen, cont_toks \
                    in zip(chunk, multi_logits, inplens, cont_toks_list):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen-contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)
        # import pdb; pdb.set_trace()
        return reord.get_original(res)

    def greedy_until(self, requests):
        raise NotImplementedError
