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

task_list = ['cola', 'mnli', 'mnli_mismatched', 'mrpc', 'rte', 'qnli', 'qqp', 'sst', 'wnli',
            'boolq','cb','copa','multirc','record','wic','wsc','drop','lambada',
            'piqa','prost','mc_taco','sciq','qa4mre_2013','triviaqa',
            'arc_challenge','logiqa','hellaswag','openbookqa','race','headqa','mathqa',
            'webqs','wsc273','winogrande','anli_r3','ethics_cm','mutual','math_algebra',
            'arithmetic_2da','hendrycksTest-abstract_algebra','wmt14-en-fr','blimp_adjunct_island']
# task_list = ['blimp_anaphor_gender_agreement']
num_fewshot = 0
max_position_embeddings = 512

description_dict_path = None
output_path = '/home/lclbw/eval_output'
