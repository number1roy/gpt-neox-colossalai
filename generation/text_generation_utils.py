import copy
import json
import os
import time
from typing import List, Union

import torch
import torch.nn.functional as F
from colossalai.core import global_context as gpc

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0

def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5

def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    eod_mask_loss=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids

def get_batch(context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.
    context_tokens: torch tensor with dimensions [batch, context_size]

    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod, ##################### To Do #############################
        eod_mask_loss=neox_args.eod_mask_loss,
    )
    return tokens, attention_mask, position_ids

def pad_batch(context_tokens: List[List[int]], pad_id: int, pad_len: int):
    """
    pads context lengths in context_tokens with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.

    context_tokens: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, context length to be padded; all batch items will be padded to the same length

    returns: tuple of padded context tokens and a list of unpadded token count
    """

    context_lengths = []
    for tokens in context_tokens:
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
        elif context_length > pad_len:
            raise ValueError("context_length is bigger than to be padded length")
        context_lengths.append(context_length)
    return context_tokens, context_lengths

def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filters the logits using top_k / top_p, filling any filtered vocab items with filter_value (defaults to -inf).

    This function has been mostly taken from huggingface conversational ai code at
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    logits: torch.Tensor -> logits of megatron model.
    top_k: integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p: float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

    returns: (filtered) logits"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits

def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


############################### To Do ######################################################
def forward_model(model, model_inputs, is_pipe_parallel=False) -> torch.Tensor:
    """
    Runs model.forward(model_inputs)

    We need to create a wrapper for this function because deepspeed pipe parallel modules operate differently to normal models.

    model: a Megatron model.
    model_inputs: tuple containing model args

    returns: torch.Tensor containing the logits of the model
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model differently to a normal model
    if not is_pipe_parallel:
        return model.module(model_inputs)
    else:
        # we need to format inputs this way because:
        # a) deepspeed pipeline only accepts iterables
        # b) deepspeed pipeline *requires* that you pass in labels for the loss, it's not easy to get around this
        # so we wrap the inputs in an iterable, and pad them (because internally, we get labels as inputs[:, 1:] and inputs as inputs[:, :-1])
        model_inputs = iter([{"text": F.pad(model_inputs[0], pad=(0, 1))}])

        # set num microbatches to 1 at inference time
        micro_batches_before = model.micro_batches
        model.micro_batches = 1

        # deepspeed sends metadata across pipeline stages only once in the first step, then assumes it will stay
        # constant. In inference, the metadata of the tensors being sent across pipe stages may change, so we need to set
        # these two flags in order for deepspeed to send the metadata every step, otherwise torch.distributed hangs
        # silently. Fun stuff.
        model.first_output_send = True
        model.pipe_recv_buf = None

        loss, logits = model.eval_batch(model_inputs, return_logits=True)
        model.micro_batches = micro_batches_before
        return logits

############################## To Do #####################################################
def broadcast_terminate_signal(terminate_runs: int):
    """Send signal to all workers to terminate if we've finished the process"""
    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(
        terminate_runs_tensor,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    return terminate_runs_tensor[0].item()

def stop_tokens_in_completion(stop_tokens, context_tokens, batch_index, current_index):
    if stop_tokens is None:
        return False
    res = []
    for token_group in stop_tokens:
        context = context_tokens[batch_index, : current_index + 1]
        context = context[-len(token_group) :]
        if context.shape[0] == token_group.shape[0]:
            res.append(all(token_group == context))
        else:
            res.append(False)
    return any(res)


####################################### To Do ########################################
def stream_tokens(
    neox_args,
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """
    iterator producing text completions

    neox_args: NeoXArgs.
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids
    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.
    maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards.
                    longer batch items get less generated tokens.
    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
    yields: (
                tokens (completions from model),
                token_generation_start_index (token index per batch item for the first generated token),
                token_generation_end_index (token index per batch item for the last generated token),
                logits (logits which are so far computed, zeros otherwise),
                is_done (flag for each bach item indicating whether an eod token was generated)
            )

            * each iteration adds a generated token to the context_tokens
            * output contains both context_tokens from input and generated tokens
            * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
    """

    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        stop_tokens = torch.cuda.LongTensor(stop_tokens)
        if stop_tokens.ndim == 1:
            stop_tokens = stop_tokens.unsqueeze(0)

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # set variables
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    maximum_tokens = maximum_tokens or (
        neox_args.seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:, :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )

                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

            if neox_args.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            yield context_tokens, token_generation_start_index, token_generation_end_index, state_is_done.bool()
            if torch.all(state_is_done):
                break
