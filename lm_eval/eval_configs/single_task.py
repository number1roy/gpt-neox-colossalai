from colossalai.amp import AMP_TYPE
from model_zoo.gpt.gpt import gpt2_small
from torch.optim import Adam

model_type = 'gpt2'
BATCH_SIZE = 2
NUM_EPOCHS = 1
SEQ_LEN = 1024

optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)


model = dict(
    type=gpt2_small,
    checkpoint=True,
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=1, mode=None),
)

checkpoint_path = '/home/lclbw/gpt_pretrained/small_ckpt'
task_list = ['blimp_anaphor_gender_agreement']
num_fewshot = 0
max_position_embeddings = 512

description_dict_path = None
output_path = '/home/lclbw/eval_output'
