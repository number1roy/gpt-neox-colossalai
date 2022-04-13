import torch
import torch.nn.functional as F
from typing import List
from torch.utils.data import Dataset

def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2

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

def get_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
    )

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, position_ids

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
    attention_masks = []
    for i in range(len(context_tokens)):
        attention_masks.append([1]*len(context_tokens[i]))
    for tokens, masks in zip(context_tokens, attention_masks):
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
            masks.extend([0] * (pad_len - context_length))
        elif context_length > pad_len:
            raise ValueError("context_length is bigger than to be padded length")
        context_lengths.append(context_length)
    return context_tokens, context_lengths, attention_masks

def get_batch(context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.
    context_tokens: torch tensor with dimensions [batch, context_size]
    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, position_ids = get_masks_and_position_ids(data=tokens)
    return tokens, attention_mask, position_ids

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

class Text_Dataset(Dataset):
    def __init__(self, input_ids, position_ids, attention_mask):
        super().__init__()
        assert input_ids is not None, 'input could not be None'
        assert attention_mask is not None, 'attention mask could not be None'
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.label = torch.ones(self.input_ids.size(), dtype=int)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'input_ids':self.input_ids[index][:],
                'attention_mask':self.attention_mask[index]}, self.label[index]
