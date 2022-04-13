import torch
from torch import dtype, nn
from colossalai import nn as col_nn
from model_zoo.gpt.gpt import GPT, PipelineGPT, GPTLMLoss, _create_gpt_model
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.builder.pipeline import partition_uniform
from colossalai.registry import LAYERS, LOSSES, MODELS
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.utils import get_current_device
from .pipeline_gpt1d import _build_gpt_pipeline_1d, _build_gpt_pipeline_hybrid

def create_gpt_model(**model_kwargs):
    model = GPT(**model_kwargs)
    return model


def create_gpt_pipeline_model(depth=48, num_chunks=1, layer_partitions=None, **model_kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(depth, pipeline_size,
                              num_chunks)[pipeline_rank] if layer_partitions is None else layer_partitions
    models = []
    for start, end in parts:
        model_kwargs['first'] = start == 0
        model_kwargs['last'] = end == depth
        model_kwargs['depth'] = end - start
        chunk = PipelineGPT(**model_kwargs).to(get_current_device())
        if start == 0:
            wrapper.register_parameter(chunk.embed.word_embedding_weight)
        elif end == depth:
            wrapper.register_parameter(chunk.head.weight)
        models.append(chunk)
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return

@MODELS.register_module
def neox_20B(**kwargs):
    model_kwargs = dict(vocab_size=53228,dim=6144, max_position_embeddings=2048, depth=44, num_heads=64, **kwargs)
    return create_gpt_model(**model_kwargs)

@MODELS.register_module
def neox_20B_PP(**kwargs):
    model_kwargs = dict(vocab_size=53228,dim=6144, max_position_embeddings=2048, depth=44, num_heads=64, **kwargs)
    return create_gpt_pipeline_model(**model_kwargs)

@MODELS.register_module
def neox_20B_PP_1d(num_chunks=4, checkpoint=False, dtype=torch.float, embed_split_hidden=False):
    cfg = dict(hidden_size=6144, num_attention_heads=64,
               checkpoint=checkpoint, max_position_embeddings=2048, dtype=dtype, embed_split_hidden=embed_split_hidden)
    return _build_gpt_pipeline_hybrid(44, num_chunks, **cfg)
