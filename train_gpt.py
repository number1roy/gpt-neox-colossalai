import os
import contextlib

import colossalai
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.utils.timer import MultiTimer
import colossalai.utils as utils
from colossalai.trainer import hooks, Trainer
from colossalai.nn import LinearWarmupLR
from colossalai.engine.schedule import PipelineSchedule, InterleavedPipelineSchedule
from model_zoo.gpt.gpt import GPTLMLoss
from colossalai.utils import is_using_pp
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.utils.checkpointing import load_checkpoint

from dataset.webtext import WebtextDataset

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

    logger.info('Build data loader', ranks=[0])
    train_ds = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LEN)
    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)

    logger.info('Build model', ranks=[0])
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    use_zero3 = hasattr(gpc.config, 'zero')

    ctx = contextlib.nullcontext()
    if use_zero3:
        ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                              shard_strategy=gpc.config.zero.model_config.shard_strategy,
                              shard_param=True
                              )

    with ctx:
        model = gpc.config.model.pop('type')(**gpc.config.model)
    # model = GPT2_exlarge_pipeline_hybrid(num_chunks=gpc.config.model.num_chunks, checkpoint=True, dtype=torch.half)

    if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    criterion = getattr(gpc.config, 'loss', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)

    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    if getattr(gpc.config, "checkpoint_path", None) is not None:
        last_epoch = load_checkpoint(gpc.config.checkpoint_path, model, optimizer, strict=False)
        logger.info(f'checkpoint loading finised, resume from {last_epoch} epoch', ranks=[0])
    else:
        logger.info(f'No checkpoint used, start from first epoch', ranks=[0])

    max_steps = getattr(gpc.config, "max_steps", None)

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

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger),
        hooks.SaveCheckpointHook(interval=2,
            checkpoint_dir=gpc.config.save_checkpoint_path)
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        max_steps= max_steps,
        test_interval=10,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False
    )


if __name__ == '__main__':
    main()
