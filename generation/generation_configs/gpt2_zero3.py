from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model_zoo.gpt.gpt import gpt2_small

SEQ_LEN = 1024
model_type = 'gpt2-small'

zero = dict(
    model_config=dict(
        offload_config=dict(device="cpu"),
        shard_strategy=TensorShardStrategy()
    ),
    optimizer_config=dict(
        cpu_offload=True,
    )
)


optimizer = dict(
    type=CPUAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt2_small,
    checkpoint=True,
)

text_gen_type = 'input-file'
checkpoint_path = '/home/lclbw/gpt_pretrained/small_ckpt.pt'
sample_output_file = '/home/lclbw/generate_output'
sample_input_file = '/home/lclbw/generate_input'
maximum_tokens = 10
recompute = False
temperature = 1.0
top_k = 5
top_p = 0.7
