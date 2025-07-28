import torch
import triton

from triton import heuristics, jit
from triton import language as tl
from triton import next_power_of_2


def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16


def get_configs():
    return [
        triton.Config(kwargs={'BI_BLOCK': 2**i}, num_stages=num_stage, num_warps=1)
        for i in range(5)
        for num_stage in [0, 1]
    ]


@triton.autotune(configs=get_configs(), key=['N'])
@jit
def _forward_with_small_n(LOGITS, PROBS, IDX, LOSS, N: tl.constexpr, B_BLOCK: tl.constexpr, M: tl.constexpr,
                          BI_BLOCK: tl.constexpr):
    ub = tl.minimum(M, (tl.program_id(0) + 1) * B_BLOCK)
    for start_row in range(tl.program_id(0) * B_BLOCK, ub, BI_BLOCK):
        row = start_row + tl.arange(0, BI_BLOCK)
        cols = tl.arange(0, N)
        idx = tl.load(IDX + row, mask=row < ub)
        # pointers to logit and probs
        LOGITS_TO_LOAD = LOGITS + row[:, None] * N + cols[None, :]
        WRIT_PROBS = PROBS + row[:, None] * N + cols[None, :]
        READ_PROBS = PROBS + row * N + idx
        # write-back negative log-probs
        logits = tl.load(LOGITS_TO_LOAD, mask=(row < ub)[:, None])
        logits = logits.to(tl.float32)

        logits = logits - tl.max(logits, 1)[:, None]
        probs = tl.log(tl.sum(tl.exp(logits), 1))[:, None] - logits

        tl.store(WRIT_PROBS, probs, mask=(row < ub)[:, None])
        # write-back loss
        probs = tl.load(READ_PROBS, mask=row < ub)
        tl.store(LOSS + row, probs, mask=row < ub)


@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _forward(LOGITS, PROBS, IDX, LOSS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to logit and probs
    LOGITS = LOGITS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    READ_PROBS = PROBS + row * N + idx
    # write-back negative log-probs
    logits = tl.load(LOGITS, mask=cols < N, other=-float('inf'))
    logits = logits.to(tl.float32)
    logits = logits - tl.max(logits, 0)
    probs = tl.log(tl.sum(tl.exp(logits), 0)) - logits
    tl.store(WRIT_PROBS, probs, mask=cols < N)
    # write-back loss
    probs = tl.load(READ_PROBS)
    tl.store(LOSS + row, probs)


@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _backward(PROBS, IDX, DPROBS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to probs
    PROBS = PROBS + row * N + cols
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    probs = -tl.load(PROBS, mask=cols < N, other=float('inf'))
    probs = tl.exp(probs.to(tl.float32))
    delta = cols == idx
    # write result in-place in PROBS
    dout = tl.load(DPROBS + row)
    din = (probs - delta) * dout
    tl.store(PROBS, din.to(PROBS.dtype.element_ty), mask=cols < N)


class _cross_entropy(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, logits, indices):
        # make sure we can use triton
        assert (indices.dtype == torch.int64), "Indices are expected to be of type long."
        # make kernel
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        # run the kernel
        result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)

        if n_cols < 2**15:
            import math
            bs = logits.numel() // n_cols
            n_batch = math.ceil(bs / 48)
            _forward_with_small_n[(48, )](logits, neg_logprobs, indices.to(torch.int32), result, n_cols, n_batch, bs)
        else:
            grid = lambda opt: (logits.numel() // n_cols, )
            # FIXME(liangyuefeng): int64 do not support in torch_mlu.
            _forward[grid](logits, neg_logprobs, indices.to(torch.int32), result, n_cols)

        # save for backward
        ctx.save_for_backward(neg_logprobs, indices)
        return result

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        # load saved tensors
        neg_logprobs, indices = ctx.saved_tensors
        # run the kernel
        # neg_logprobs will be modified in place to become our gradient:
        n_cols = neg_logprobs.shape[-1]
        grid = lambda opt: (neg_logprobs.numel() // n_cols, )
        # FIXME(liangyuefeng): int64 do not support in torch_mlu.
        _backward[grid](neg_logprobs, indices.to(torch.int32), dneg_logprobs, n_cols)
        return neg_logprobs, None


cross_entropy = _cross_entropy.apply
