from colossalai.amp import AMP_TYPE
from model_zoo.gpt.gpt import gpt2_small
from torch.optim import Adam


SEQ_LEN = 1024
model_type = 'gpt2-small'

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

text_gen_type = 'input-file'
checkpoint_path = '/home/lclbw/gpt_pretrained/small_ckpt.pt'
sample_output_file = '/home/lclbw/generate_output'
sample_input_file = '/home/lclbw/generate_input'
maximum_tokens = 10
recompute = False
temperature = 1.0
top_k = 5
top_p = 0.7
