from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model_zoo.gpt.gpt import GPTLMLoss
from model.custom_model_zoo import neox_20B
from model import vocab_parallel_cross_entropy

BATCH_SIZE = 16
NUM_EPOCHS = 60
SEQ_LEN = 2048

HIDDEN_SIZE = 6144
TENSOR_PARALLEL = 4
MODE  = '2d'

zero = dict(
    model_config=dict(
        offload_config=dict(device="cpu"),
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        cpu_offload=True,
    )
)

loss = dict(
    type=GPTLMLoss,
)

parallel = dict(
    pipeline=1,
    tensor=dict(mode=MODE, size=TENSOR_PARALLEL)
)

optimizer = dict(
    type=CPUAdam,
    lr=0.97e-4,
    betas=(0.9, 0.95),
    eps=1.0e-8,
    weight_decay=1e-2,
)

model = dict(
    type=neox_20B,
    checkpoint=True,
)

max_steps = 10
# checkpoint_path = '../gpt_pretrained/neox/neox_20B.pt'
save_checkpoint_path = '../gpt_pretrained/neox/neox_20B.pt'
