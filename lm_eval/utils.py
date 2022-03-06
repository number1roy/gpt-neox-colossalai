import os
import re
import collections
import functools
import inspect
import torch
from torch.utils.data import Dataset


class ExitCodeError(Exception):
    pass

class Request_Context_Dataset(Dataset):
    def __init__(self, chunks, max_length):
        super().__init__()

        self.doc = []
        self.context_enc = []
        self.continuation_enc = []
        self.max_length = max_length
        self.length = 0

        for chunk in chunks:
            context_enc = chunk[1]
            continuation_enc = chunk[2]
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length
            self.doc.append(chunk[0])
            self.context_enc.append(context_enc)
            self.continuation_enc.append(continuation_enc)
            self.length += 1

        self.inplens = []
        self.cont_toks_list = []

    def _get_inplens(self):
        return self.inplens

    def _get_cont_toks_list(self):
        return self.cont_toks_list

    def _get_context_enc(self):
        return self.context_enc

    def _get_continuation_enc(self):
        return self.continuation_enc

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.context_enc[index], self.continuation_enc[index]

def collate_fn(batch):
    inps = []
    masks = []
    padding_length = None
    max_length = 1024
    for data in batch:
        # how this all works:
        #          CTX      CONT
        # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
        # gpt2    \               \
        # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
        # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

        # when too long to fit in context, truncate from the left (according to [-(max_length+1):])
        # data[0] + data[1] equals concatenate operation
        # finally delete continuation_enc, length of which equals 1 (according to [:-1])
        inp = torch.tensor(
            (data[0] + data[1])[-(max_length+1):][:-1],
            dtype=torch.long
        ).to('cuda')
        inplen, = inp.shape

        cont = data[1]

        # since in _collate we make sure length is descending, the longest is always the first one.
        padding_length = padding_length if padding_length is not None else inplen

        # pad length from seq to padding_length
        inp = torch.cat([
            inp,  # [seq]
            torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
        ], dim=0)

        attention_mask = torch.where(inp!=0, 1, 0)

        inps.append(inp.unsqueeze(0))  # [1, padding_length]
        masks.append(attention_mask.unsqueeze(0))

    batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
    batched_masks = torch.cat(masks, dim=0)
    return [batched_inps, batched_masks]

def sh(x):
    if os.system(x):
        raise ExitCodeError()


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = args_string.split(",")
    args_dict = {}
    for arg in arg_list:
        k, v = arg.split("=")
        args_dict[k] = v
    return args_dict

def join_iters(iters):
    for iter in iters:
        yield from iter


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr: yield arr

def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace("\" ", "\"")
    string = string.replace(" \"", "\"")
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield (
        [prefix_token] + token_list[:first_seq_len - 1],
        token_list[:first_seq_len]
    )
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1:window_end - 1],
            token_list[window_end - window_pred_len:window_end],
        )
        predicted += window_pred_len

def make_disjoint_window(pair):
    """ Takes output from get_rolling_token_windows and makes the context not overlap with the continuation """

    a, b = pair

    return a[:-(len(b) - 1)], b

class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [
            ([y[0] for y in x], x[0][1]) for x in arr
        ]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr


    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res

def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!")
        return fn(*args, **kwargs)
    return _wrapper
