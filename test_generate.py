from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger, disable_existing_loggers
import colossalai
import os
from colossalai.core import global_context as gpc
from colossalai.utils.timer import MultiTimer
from colossalai.zero import zero3_model_context
import colossalai.utils as utils
from colossalai.trainer import hooks, Trainer
from colossalai.nn import LinearWarmupLR
import torch.nn as nn
from dataset.webtext import WebtextDataset
import contextlib
from colossalai.engine.schedule import PipelineSchedule, InterleavedPipelineSchedule
from model_zoo.gpt.gpt import GPTLMLoss
from colossalai.utils import is_using_pp


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=42)

    logger = get_dist_logger()

    logger.info('Build model', ranks=[0])
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    use_zero3 = hasattr(gpc.config, 'zero') and gpc.config.zero.level == 3
    ctx = zero3_model_context() if use_zero3 else contextlib.nullcontext()
    with ctx:
        model = gpc.config.model.pop('type')(**gpc.config.model)
        # model = GPT2_exlarge_pipeline_hybrid(num_chunks=gpc.config.model.num_chunks, checkpoint=True, dtype=torch.half)

    if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)

    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion)

    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])
    tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
    schedule = None
    if use_pipeline:
        if use_interleaved:
            logger.info('Build InterleavedPipelineSchedule', ranks=[0])
            schedule = InterleavedPipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                                   gpc.config.model.num_chunks, tensor_shape=tensor_shape, scatter_gather_tensors=True)
        else:
            logger.info('Build PipelineSchedule', ranks=[0])
            schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                        tensor_shape=tensor_shape, scatter_gather_tensors=True)

    timier = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        schedule=schedule,
        timer=timier
    )

    trainer.predict()

def _get_input():
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
    for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
        inps = []
        cont_toks_list = []
        inplens = []

        padding_length = None

        # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
        # tensors, then we pack them together into a batch, call the model, and then pick it all apart
        # again because vectorizing is annoying

        for _, context_enc, continuation_enc in chunk:
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.max_length+1):][:-1],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape

            cont = continuation_enc

            # since in _collate we make sure length is descending, the longest is always the first one.
            padding_length = padding_length if padding_length is not None else inplen

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
            ], dim=0)

            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length

if __name__ == '__main__':
    main()
