from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.shard_utils import TensorShardStrategy
from model_zoo.gpt.gpt import gpt2_small

model_type = 'gpt2-small'
BATCH_SIZE = 2
SEQ_LEN = 1024
no_cache=True


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

checkpoint_path = '/home/lclbw/gpt_pretrained/small_ckpt.pt'
# task_list = ['blimp_anaphor_gender_agreement']
task_list = ['wnli']
num_fewshot = 0
max_position_embeddings = 512

description_dict_path = None
output_path = '/home/lclbw/eval_output'
