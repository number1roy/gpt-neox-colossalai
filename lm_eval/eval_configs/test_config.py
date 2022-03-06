from model import GPT_custom_pipeline_1D
from torch.optim import Adam
from colossalai.amp import AMP_TYPE
import torch
from model import vocab_parallel_cross_entropy

BATCH_SIZE = 8
NUM_EPOCHS = 1
SEQ_LEN = 1024

NUM_MICRO_BATCHES = 1
HIDDEN_SIZE = 768
PIPELINE = 2
TENSOR_PARALLEL = 2
MODE  = '1d'
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

num_attention_heads = 12
num_layers = 12

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=PIPELINE,
    tensor=dict(mode=MODE, size=TENSOR_PARALLEL)
)

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=GPT_custom_pipeline_1D,
    checkpoint=True,
    dtype=torch.half,
)

loss_fn = dict(type=vocab_parallel_cross_entropy)

checkpoint_path = '/home/lclbw/gpt_pretrained/small_ckpt'

task_list = ['cola', 'mnli', 'mnli_mismatched', 'mrpc', 'rte', 'qnli', 'qqp', 'sst', 'wnli']
num_fewshot = 0
max_position_embeddings = 512

description_dict_path = None
output_path = '/home/lclbw/eval_output'
