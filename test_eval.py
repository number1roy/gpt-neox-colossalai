import torch
import colossalai
from colossalai.logging import get_dist_logger, disable_existing_loggers
from lm_eval.models import get_model
from colossalai.core import global_context as gpc
from lm_eval import tasks, evaluator
import json

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

    if len(gpc.config.task_list) == 1 and gpc.config.task_list[0] == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = gpc.config.task_list

    description_dict = {}
    if gpc.config.description_dict_path:
        with open(gpc.config.description_dict_path, 'r') as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model='gpt2',
        model_args=None,
        tasks=gpc.config.task_list,
        num_fewshot=0,
        device='cuda:0',
        no_cache=False,
        limit=None,
        description_dict=None
    )

    dumped = json.dumps(results, indent=2)

    print(dumped)

    if gpc.config.output_path:
        with open(gpc.config.output_path, "w") as f:
            f.write(dumped)

    # print(
    #     f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
    #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    # )
    print(evaluator.make_table(results))


if __name__ == '__main__':
    main()
