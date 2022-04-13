import colossalai
from colossalai.logging import disable_existing_loggers
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

    if len(gpc.config.task_list) == 1 and gpc.config.task_list[0] == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = gpc.config.task_list

    description_dict = None
    if gpc.config.description_dict_path:
        with open(gpc.config.description_dict_path, 'r') as f:
            description_dict = json.load(f)

    model_type = gpc.config.get('model_type', None)
    assert model_type is not None, 'The model type could not be None'
    type = model_type.split('-')[0]

    results = evaluator.simple_evaluate(
        model=type,
        model_args=None,
        tasks=task_names,
        num_fewshot=gpc.config.num_fewshot,
        device='cuda',
        no_cache=gpc.config.no_cache,
        limit=None,
        description_dict=description_dict
    )

    dumped = json.dumps(results, indent=2)

    if gpc.get_global_rank() == 0:
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
