import transformers
import time
import copy
import contextlib
import colossalai
import json
from typing import Union
from colossalai.utils import print_rank_0
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils.checkpointing import load_checkpoint
from colossalai.engine.schedule import NonPipelineSchedule
from colossalai.zero.init_ctx import ZeroInitContext
from model_zoo.gpt.gpt import GPTLMLoss
from torch.utils.data import DataLoader
from .utils import *
from model.hf_model import build_model, load_config_from_colossalai

class GPT_nonepipe_generate(object):
    def __init__(self, device='cuda', revision='main', subfolder=None, tokenizer=None):
        super().__init__()
        self.seq_length = gpc.config.SEQ_LEN

        assert isinstance(device, str)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.tokenizer = transformers.GPT2Tokenizer
        pretrained = gpc.config.get('model_type', None)
        assert pretrained is not None, 'The model type could not be None'
        token_type = pretrained.split('-')[0]
        assert token_type in ['gpt2', 'gpt3'], f'The tokenizer type {token_type} is unknown.'

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            token_type if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

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

        self.model = model

        self.criterion = getattr(gpc.config, 'loss_fn', None)
        if self.criterion is not None:
            self.criterion = self.criterion.type()
        else:
            self.criterion = GPTLMLoss()

        logger.info('Build optimizer', ranks=[0])
        self.optimizer = gpc.config.optimizer.pop('type')(
            model.parameters(), **gpc.config.optimizer)

        self.engine, _, _, _ = colossalai.initialize(self.model, self.optimizer, self.criterion)
        self.engine.eval()

        assert getattr(gpc.config, "checkpoint_path", None) is not None, \
            f'Please give the checkpoint path in configuration.'
        last_epoch = load_checkpoint(gpc.config.checkpoint_path, self.model, self.optimizer, strict=False)
        logger.info(f'checkpoint loading finised, resume from {last_epoch} epoch', ranks=[0])

        logger.info(f'Init done', ranks=[0])

        self._schedule = NonPipelineSchedule()
        self._schedule.pre_processing(self.engine)

    @property
    def schedule(self):
        return self._schedule

    def _model_call(self, data_loader):
        """
        data_loader: a torch DataLoader containing input tensor, attention_masks
        and labels (due to the need of colossalai). The shape of input tensor is
        [batch, sequence], the size of sequence may vary from call to call.

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        data_iter = iter(data_loader)
        with torch.no_grad():
            logit, _, _ = self.schedule.forward_backward_step(
                self.engine,
                data_iter,
                forward_only=True,
                return_loss=True,
                return_output_label=True,
            )

        return logit.contiguous().cuda()

    def generate_samples_from_prompt(
        self,
        text: Union[List[str], str],
        eos_token_id: int = None,
        maximum_tokens: int = 64,
        recompute: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        stop_tokens=None,
    ):
        """
        Generates samples from raw text and returns them in a dictionary.
        text: either a single prompt (str) or a list of prompts (List[str]).
        eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
        maximum_tokens: maximum number of tokens to be generated
        recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
        temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
        top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
        top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
        note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
        returns: List[dict] -> a list of dicts containing the following fields:
            - 'context' (the input)
            - 'text' (the completion)
            - 'length' (the length of the completion in number of tokens)
            - 'finished':
            - 'message': a messaged associated with the generation procedure, can be a warning or error
            - 'duration_seconds': duration of the generation in seconds
        """
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id

        # type check
        assert any(
            [isinstance(text, str), isinstance(text, list)]
        ), "Text should be in string or list form"

        if isinstance(text, str):
            text = [text]

        input_count = len(text)
        input_pos = 0

        # generate completions
        generated_texts = []
        while True:
            # model.module.clear_cache()  # clear kv cache between batches

            start_time = time.time()
            # Tokenize text, and check whether we should terminate process
            terminate_runs = 0
            if input_pos == input_count:
                terminate_runs = 1
            else:
                raw_text = text[input_pos]
                input_pos += 1

                if raw_text == "":
                    context_tokens = [eos_token_id]
                else:
                    context_tokens = self.tokenizer.encode(raw_text)
                context_length = len(context_tokens)

                if context_length >= (self.seq_length // 2):
                    print_rank_0(
                        f"\nWarning! Context length {context_length} is too large. Please give smaller context"
                    )

            if terminate_runs == 1:
                return generated_texts

            for (
                batch_context_tokens,
                batch_token_generation_start_index,
                batch_token_generation_end_index,
                is_done,
            ) in self.stream_tokens_nonepipe(
                context_tokens=[context_tokens],
                eos_token_id=eos_token_id,
                maximum_tokens=maximum_tokens,
                recompute=recompute,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_tokens,
            ):
                pass  # finish generation and use all results below

            batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
            batch_token_generation_start_index = (
                batch_token_generation_start_index.cpu().numpy().tolist()
            )
            batch_token_generation_end_index = (
                batch_token_generation_end_index.cpu().numpy().tolist()
            )
            batch_is_done = is_done.cpu().numpy().tolist()

            for tokens, start_index, end_index, is_done in zip(
                batch_context_tokens,
                batch_token_generation_start_index,
                batch_token_generation_end_index,
                batch_is_done,
            ):

                if end_index >= start_index:
                    generated_tokens = tokens[start_index : end_index + 1]
                    try:
                        generated_text = self.tokenizer.decode(generated_tokens)
                        message = None
                    except KeyError:
                        generated_text = None
                        message = "WARNING: generated token which doesn't exist."
                else:
                    generated_text = None
                    generated_tokens = []
                    # this will happen if the first generated token is a stop token or eos token
                    message = "WARNING: text generation did not start; try different batching or adjust parameters"

                data = {
                    "context": raw_text,
                    "text": generated_text,
                    "length": len(generated_tokens),
                    "finished": is_done,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),
                }
                generated_texts.append(data)

        return generated_texts

    def stream_tokens_nonepipe(
        self,
        eos_token_id: int,
        context_tokens: List[List[int]],
        maximum_tokens: int = None,
        recompute: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        stop_tokens=None,
    ):
        """
        iterator producing text completions
        model: a Megatron model.
        context_tokens: the prompt to complete; unpadded list of lists of tokens ids
        context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
        eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
        attention_mask: attention mask for megatron model.
        position_ids: position ids for positional encoding.
        maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards.
                        longer batch items get less generated tokens.
        recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
        temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
        top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
        top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
        note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
        yields: (
                    tokens (completions from model),
                    token_generation_start_index (token index per batch item for the first generated token),
                    token_generation_end_index (token index per batch item for the last generated token),
                    logits (logits which are so far computed, zeros otherwise),
                    is_done (flag for each bach item indicating whether an eod token was generated)
                )
                * each iteration adds a generated token to the context_tokens
                * output contains both context_tokens from input and generated tokens
                * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
        """
        assert eos_token_id is not None, "eos_token_id could not be None"

        # pad batch in order to allow conversion to tensor
        context_tokens, context_lengths, attention_mask = pad_batch(
            copy.deepcopy(context_tokens),
            pad_id=eos_token_id,
            pad_len=self.seq_length,
        )

        # convert to tensor
        context_tokens = torch.cuda.LongTensor(context_tokens)
        attention_mask = torch.cuda.LongTensor(attention_mask)
        if stop_tokens:
            stop_tokens = torch.cuda.LongTensor(stop_tokens)
            if stop_tokens.ndim == 1:
                stop_tokens = stop_tokens.unsqueeze(0)

        token_generation_start_index = torch.cuda.LongTensor(context_lengths)

        # get attention mask / position ids
        context_tokens, triangular_mask, position_ids = get_batch(context_tokens)

        # set variables
        maximum_tokens = maximum_tokens or (
            self.seq_length - token_generation_start_index.max().item() - 1
        )
        batch_size = context_tokens.size(0)

        # get the context_index at which generation is to start
        # we start generation at the position where the smallest context ends
        token_index_to_generate = token_generation_start_index.min().item()
        first_token_index_to_generate = token_index_to_generate
        last_token_index_to_generate = min(
            self.seq_length - 1,  # never generate more than the model's sequence length
            token_index_to_generate + maximum_tokens - 1,
        )

        with torch.no_grad():
            # initialize generation variables
            state_is_done = torch.zeros([batch_size]).byte().cuda()
            token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)

            while token_index_to_generate <= last_token_index_to_generate:
                if recompute:  # recompute all tokens
                ########################################################################
                    model_inputs = Text_Dataset(
                        context_tokens,
                        position_ids,
                        attention_mask,
                    )
                    text_dataloader = DataLoader(model_inputs, batch_size=1, shuffle=False,
                                                    sampler=None,
                                                    batch_sampler=None, collate_fn=None,
                                                    pin_memory=False, drop_last=False)
                    logits = self._model_call(text_dataloader)
                ########################################################################
                    if logits is not None:  # if pipe parallel, not all ranks return logits
                        generated_token_logits = logits[
                            :, token_index_to_generate - 1, :
                        ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
                else:  # use kv cache
                    if token_index_to_generate == first_token_index_to_generate:
                        tokens_to_use = context_tokens[:, :token_index_to_generate]
                        positions_to_use = position_ids[:, :token_index_to_generate]
                        attention_mask_to_use = attention_mask[:, :token_index_to_generate]
                    else:
                        tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                            batch_size, -1
                        )
                        positions_to_use = position_ids[:, token_index_to_generate - 1].view(
                            batch_size, -1
                        )
                        attention_mask_to_use = attention_mask[:, token_index_to_generate - 1].view(
                            batch_size, -1
                        )
                ########################################################################
                    model_inputs = Text_Dataset(
                        tokens_to_use,  # input_ids
                        positions_to_use,  # position_ids
                        attention_mask_to_use,  # attention_mask
                    )
                    text_dataloader = DataLoader(model_inputs, batch_size=1, shuffle=False,
                                                    sampler=None,
                                                    batch_sampler=None, collate_fn=None,
                                                    pin_memory=False, drop_last=False)
                    logits = self._model_call(text_dataloader)
                ########################################################################
                    if logits is not None:  # if pipe parallel, not all ranks return logits
                        generated_token_logits = (
                            logits[:, -1].view(batch_size, -1).contiguous()
                        )  # [bs, seq, vocab_size] -> [bs, vocab_size]

                if logits is not None:
                    # sample token id of the to be generated token
                    if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                        generated_tokens = torch.argmax(
                            generated_token_logits, dim=-1
                        ).view(-1)
                    else:
                        generated_token_logits = generated_token_logits.float()
                        if temperature > 0.0:
                            generated_token_logits /= temperature
                        generated_token_logits = filter_logits(
                            generated_token_logits, top_k=top_k, top_p=top_p
                        )
                        next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                        generated_tokens = torch.multinomial(
                            next_token_log_probs, num_samples=1
                        ).view(-1)

                # determine if state has started for each batch item
                state_started = (
                    token_generation_start_index <= token_index_to_generate
                )  # check which batch items have been started

                # switch out padding tokens for generated tokens
                context_tokens[:, token_index_to_generate] = switch(
                    context_tokens[:, token_index_to_generate].view(-1),
                    generated_tokens,
                    state_started,
                )

                # switch out zeros for ones in attention_mask
                attention_mask[:, token_index_to_generate] = switch(
                    context_tokens[:, token_index_to_generate].view(-1),
                    torch.ones(batch_size).cuda(),
                    state_started,
                )


                # determine if state has finished for each batch item
                state_done = (
                    generated_tokens == eos_token_id
                ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
                state_just_finished = (state_done & ~state_is_done).bool()
                state_is_done = state_is_done | state_done
                stop_tokens_produced = torch.zeros_like(state_is_done)
                for batch_idx, ctx in enumerate(context_tokens):
                    stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                        stop_tokens, context_tokens, batch_idx, token_index_to_generate
                    )
                state_is_done = state_is_done | stop_tokens_produced

                token_generation_end_index[
                    (state_started.byte() & ~state_is_done).bool()
                ] = token_index_to_generate

                token_index_to_generate += 1

                yield context_tokens, token_generation_start_index, token_generation_end_index, state_is_done.bool()
                if torch.all(state_is_done):
                    break


    def generate_samples_input_from_file(
        self,
        input_file,
        output_file=None,
        eos_token_id: int = None,
        maximum_tokens: int = 64,
        recompute: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ):
        """
        Generates samples from an input file and writes them to an output file.
        Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file
        neox_args: NeoXArgs.
        model: a Megatron model
        input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
        output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined
        eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
        maximum_tokens: maximum number of tokens to be generated
        recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
        temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
        top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
        top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
        note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
        returns: List[dict] -> a list of dicts containing the following fields:
            - 'context' (the input)
            - 'text' (the completion)
            - 'length' (the length of the completion in number of tokens)
            - 'finished':
            - 'message': a messaged associated with the generation procedure, can be a warning or error
            - 'duration_seconds': duration of the generation in seconds
        """
        # Read the sample file
        print_rank_0(
            "generate_samples_input_from_file() loading input from {}".format(input_file)
        )
        with open(input_file, "r") as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
        prompts = [p for p in prompts if len(p) > 0]
        print_rank_0(
            "generate_samples_input_from_file() prompts loaded: {}".format(len(prompts))
        )

        if gpc.get_global_rank() == 0:
            if output_file is None:
                output_file = str(input_file) + ".output.jsonl"
                print_rank_0(
                    "generate_samples_input_from_file() setting default output file to {}".format(
                        output_file
                    )
                )

        print_rank_0("generate_samples_input_from_file() generating...")
        generated_texts = self.generate_samples_from_prompt(
            text=prompts,
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if gpc.get_global_rank() == 0:
            with open(output_file, "w") as f_out:
                for item in generated_texts:
                    f_out.write(json.dumps(item) + "\n")
        print_rank_0("generate_samples_input_from_file() done")
        return generated_texts


    def generate_samples_unconditional(
        self,
        number_of_samples: int = 10,
        output_file=None,
        eos_token_id: int = None,
        maximum_tokens: int = 64,
        recompute: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ):
        """
        Generates samples unconditionially (no prompt) and yields them in a dictionary.
        neox_args: NeoXArgs.
        model: a Megatron model
        number_of_samples (default 10): number of unconditional samples to be generated
        output_file: file where generation results are to be stored in jsonl format. no file will be stored if ommitted
        eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
        maximum_tokens: maximum number of tokens to be generated
        recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
        temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
        top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
        top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
        note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
        yields: dict containing the following fields:
            - 'context' (the input)
            - 'text' (the completion)
            - 'length' (the length of the completion in number of tokens)
            - 'finished':
            - 'message': a messaged associated with the generation procedure, can be a warning or error
            - 'duration_seconds': duration of the generation in seconds
        """

        print_rank_0("generate_samples_unconditional() generating...")
        assert number_of_samples > 0, "number_of_samples must be > 0"
        generated_texts = self.generate_samples_from_prompt(
            text=["" for _ in range(number_of_samples)],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if gpc.get_global_rank() == 0:
            if output_file is not None:
                with open(output_file, "w") as f_out:
                    for item in generated_texts:
                        f_out.write(json.dumps(item) + "\n")
        print_rank_0("generate_samples_unconditional() done")
        return generated_texts
