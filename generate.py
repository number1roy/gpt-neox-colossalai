from colossalai.logging import disable_existing_loggers
import colossalai
from colossalai.core import global_context as gpc
from colossalai.utils import print_rank_0
from generation.model import GPT_nonepipe_generate


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

    model = GPT_nonepipe_generate()

    if gpc.config.text_gen_type == "unconditional":
        print_rank_0(
            f"Generating samples unconditionally and saving results to {gpc.config.sample_output_file}"
        )
        model.generate_samples_unconditional(
            number_of_samples=gpc.config.num_samples,
            output_file=gpc.config.sample_output_file,
            maximum_tokens=gpc.config.maximum_tokens,
            recompute=gpc.config.recompute,
            temperature=gpc.config.temperature,
            top_k=gpc.config.top_k,
            top_p=gpc.config.top_p,
        )

    elif gpc.config.text_gen_type == "input-file":
        print_rank_0(
            f"Generating samples from input file {gpc.config.sample_input_file}"
        )
        assert gpc.config.sample_input_file is not None
        model.generate_samples_input_from_file(
            input_file=gpc.config.sample_input_file,
            output_file=gpc.config.sample_output_file,
            maximum_tokens=gpc.config.maximum_tokens,
            recompute=gpc.config.recompute,
            temperature=gpc.config.temperature,
            top_k=gpc.config.top_k,
            top_p=gpc.config.top_p,
        )
        print_rank_0(
            f"Generated samples saving to {gpc.config.sample_output_file}"
        )
    else:
        raise ValueError(
            f"`text-gen-type` either not specified or not recognised: {gpc.config.text_gen_type}"
        )





if __name__ == '__main__':
    main()
