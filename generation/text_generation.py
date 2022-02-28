from text_generation_utils import *
import colossalai
import torch


def generate_samples_from_prompt(
    neox_args,
    model,
    text: Union[List[str], str],
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    Generates samples from raw text and returns them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds

    """
    ################################## To Do #####################################
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    ##############################################################################

    # type check
    assert any(
        [isinstance(text, str), isinstance(text, list)]
    ), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]

    input_count = len(text)
    input_pos = 0

    # generate completions
    generated_texts = []
    while True:
        ################################## To Do #####################################
        model.module.clear_cache()  # clear kv cache between batches
        ##############################################################################

        start_time = time.time()
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if input_pos == input_count:
            terminate_runs = 1
        else:
            raw_text = text[input_pos]
            input_pos += 1

            if raw_text == "":
                context_tokens = [eos_token_id]
            else:
                context_tokens = neox_args.tokenizer.tokenize(raw_text)
            context_length = len(context_tokens)

            if context_length >= (neox_args.seq_length // 2):
                print_rank_0(
                    "\nWarning! Context length",
                    context_length,
                    "\nPlease give smaller context (e.g. half of the "
                    "max sequence length)!",
                )
        if not is_mp_rank_0():
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts

        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pass  # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = (
            batch_token_generation_start_index.cpu().numpy().tolist()
        )
        batch_token_generation_end_index = (
            batch_token_generation_end_index.cpu().numpy().tolist()
        )
        batch_is_done = is_done.cpu().numpy().tolist()

        for tokens, start_index, end_index, is_done in zip(
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_is_done,
        ):

            if end_index >= start_index:
                generated_tokens = tokens[start_index : end_index + 1]
                try:
                    generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                    message = None
                except KeyError:
                    generated_text = None
                    message = "WARNING: generated token which doesn't exist."
            else:
                generated_text = None
                generated_tokens = []
                # this will happen if the first generated token is a stop token or eos token
                message = "WARNING: text generation did not start; try different batching or adjust parameters"
            if is_mp_rank_0():
                data = {
                    "context": raw_text,
                    "text": generated_text,
                    "length": len(generated_tokens),
                    "finished": is_done,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),
                }
                generated_texts.append(data)

    return generated_texts


def generate_samples_input_from_file(
    neox_args,
    model,
    input_file,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file

    neox_args: NeoXArgs.
    model: a Megatron model

    input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
    output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0


    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """
    # Read the sample file
    print_rank_0(
        "generate_samples_input_from_file() loading input from {}".format(input_file)
    )
    with open(input_file, "r") as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0(
        "generate_samples_input_from_file() prompts loaded: {}".format(len(prompts))
    )

    if is_mp_rank_0():
        if output_file is None:
            output_file = str(input_file) + ".output.jsonl"
            print_rank_0(
                "generate_samples_input_from_file() setting default output file to {}".format(
                    output_file
                )
            )

    print_rank_0("generate_samples_input_from_file() generating...")
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=prompts,
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if is_mp_rank_0():
        with open(output_file, "w") as f_out:
            for item in generated_texts:
                f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_input_from_file() done")
    return generated_texts


def generate_samples_unconditional(
    neox_args,
    model,
    number_of_samples: int = 10,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    number_of_samples (default 10): number of unconditional samples to be generated

    output_file: file where generation results are to be stored in jsonl format. no file will be stored if ommitted

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    print_rank_0("generate_samples_unconditional() generating...")
    assert number_of_samples > 0, "number_of_samples must be > 0"
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=["" for _ in range(number_of_samples)],
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if is_mp_rank_0():
        if output_file is not None:
            with open(output_file, "w") as f_out:
                for item in generated_texts:
                    f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_unconditional() done")
    return generated_texts


def generate_samples_interactive(
    neox_args,
    model,
    maximum_tokens: int = 64,
    eos_token_id: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs.
    model: a Megatron model

    maximum_tokens: maximum number of tokens to be generated
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """

    while True:
        model.module.clear_cache()  # clear kv cache between batches
        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        terminate_runs = 0

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            os.system("clear")
            raw_text = input("Context prompt >>> ")
            context_tokens = neox_args.tokenizer.tokenize(raw_text)
            if len(context_tokens) == 0:
                context_tokens = [neox_args.tokenizer.eod]
            context_length = len(context_tokens)
            if context_length >= (neox_args.seq_length - 1):
                print_rank_0(
                    "\nContext length"
                    + str(context_length)
                    + "\nReached max sequence length!"
                )
                terminate_runs = 1
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return
        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            if mpu.get_model_parallel_rank() == 0:
                generated_tokens = (
                    batch_context_tokens[0]
                    .cpu()
                    .numpy()
                    .tolist()[
                        batch_token_generation_start_index[0]
                        .item() : batch_token_generation_end_index[0]
                        .item()
                    ]
                )
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                print_rank_0("Generated Text: " + generated_text)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            _ = input("\n<press enter to continue>")
