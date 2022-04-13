# Run GPT With Colossal-AI

## Overview

In Colossal-AI, there are many ways to run GPT in a distributed manner. The `train_gpt.py` script runs training with the specific configuration scripts in `gpt2_configs/` for different parallelisms of GPT-2 . We have provided some example configuration files of GPT-2 and you can modify them to adapt to your own use.

## How to Prepare Webtext Dataset

We do not host any datasets for GPT or BERT training, however, we provide a detailed guide on how to prepare the dataset so that our results may be reproduced.

### Oveview
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library by [jcpeterson](https://github.com/jcpeterson/openwebtext) and  [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls to different web pages. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in Megatron's [openwebtext](./tools/openwebtext) directory.

The details of preparing the Dataset could be found in https://github.com/hpcaitech/ColossalAI-Examples/blob/d50ef2db51e7d02ed3f7e9de13f9af86b04eaae9/language/gpt/readme.md

## How to train the GPT model
After Dataset preparation, you could easily start training your model by using

```Bash
#!/usr/bin/env sh
export DATA=/path/to/train_data.json

torchrun --standalone --nproc_per_node=<num_gpus> train_gpt.py --config=training_configs/gpt2_configs/<config_file> --from_torch
```

You can copy it and save it as `run.sh`. Then use `bash ./run.sh` to run the script in your terminal.

Please modify `DATA`, `num_gpus` and `config_file` with the path to your dataset, the number of GPUs and the config file path, respectively.
If you are going to train gpt3, just replace `gpt2_configs` with `gpt3_configs`. If you are going to use custom config, please follow the guide in the training_configs part.

## How to evaluate the GPT model
The evaluation function is mainly adapted from https://github.com/EleutherAI/lm-evaluation-harness. We use ColossalAI (see https://github.com/hpcaitech/ColossalAI) to support distributed training and evaluation. The evaluation for GPT model could be simply done by using

```Bash
#!/usr/bin/env sh
torchrun --standalone --nproc_per_node=<num_gpus> eval.py --config=lm_eval/eval_configs/<config_file> --from_torch
```

The configs for evaluation are similar with training. But you should specify the path to pretrained model, model type and the task list. Two simple configuration files could be found in `./lm_eval/eval_configs`. When you set `no_cache=False`, you could find your cached evaluation results in `./lm_cache`.

### Tasks for evaluation
Now we support single GPUs for multiple classification task. The multiple GPUs may exist some issues. And for generation tasks, we are still working on in and would be updated in the near future. The task list we support is recorded here

| Task name | Metric |
| ------------ | ----- |
| cola | mcc |
| mnli | acc |
| mnli_mismatched | acc |
| mrpc | acc, f1 |
| rte | acc |
| qnli | acc |
| qqp | acc, f1 |
| sst | acc |
| wnli | acc |
| boolq | acc |
| cb | acc, f1 |
| copa | acc |
| record | f1, em |
| wic | acc |
| wsc | acc |
| drop | em, f1 |
| lambada | ppl, acc |
| piqa | acc, acc_norm |
| prost | acc, acc_norm |
| mc_taco | em, f1 |
| sciq | acc, acc_norm |
| qa4mre_2013 | acc, acc_norm |
| arc_challenge | acc, acc_norm |
| logiqa | acc, acc_norm |
| hellaswag | acc, acc_norm |
| openbookqa | acc, acc_norm |
| race | acc |
| headqa | acc, acc_norm |
| mathqa | acc, acc_norm |
| webqs | acc |
| wsc273 | acc |
| winogrande | acc |
| anli_r3 | acc |
| ethics_cm | acc |
| math_algebra | acc |
| arithmetic_2da | acc |
| hendrycksTest-abstract_algebra | acc, acc_norm |
| wmt14-en-fr | bleu, chrf, ter |
| blimp_adjunct_island | acc |

Please make sure your network connection available, due to tasks document needed to be downloaded during evaluation.

### Tasks for generation

The generation part is mainly adapted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/text_generation_utils.py. Now we support unconditional generation and generation from files. You could
set your generation mode in your configuration files. In your configuration files you have to give the path
to your checkpoint files used for generation. Then you can start your generation by using command below:

```Bash
#!/usr/bin/env sh
torchrun --standalone --nproc_per_node=<num_gpus> generate.py --config=./generation/generate_configs/<config_file> --from_torch
```
