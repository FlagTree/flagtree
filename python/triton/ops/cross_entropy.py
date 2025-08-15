HAS_TORCH = False
HAS_PADDLE = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    import paddle
    HAS_PADDLE = True
    

from .. import heuristics, jit
from .. import language as tl
from .. import next_power_of_2


def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16


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
    # There is a bug in the compiler, which fails to insert a barrier here.
    # We add it explicitly for now. Will be fixed soon.
    tl.debug_barrier()
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


# ---- 框架封装：优先 Paddle，无 Paddle 用 Torch ----

if HAS_TORCH:
        # Torch 版（保持原逻辑）
    class _cross_entropy(torch.autograd.Function):

        @classmethod
        def forward(cls, ctx, logits, indices):
            assert (indices.dtype == torch.int64), "Indices are expected to be of type long."
            device, dtype = logits.device, logits.dtype
            n_cols = logits.shape[-1]
            result = torch.empty_like(indices, dtype=dtype, device=device)
            neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
            grid = lambda opt: (logits.numel() // n_cols, )
            _forward[grid](logits, neg_logprobs, indices, result, n_cols)
            ctx.save_for_backward(neg_logprobs, indices)
            return result

        @classmethod
        def backward(cls, ctx, grad_out):
            neg_logprobs, indices = ctx.saved_tensors
            n_cols = neg_logprobs.shape[-1]
            grid = lambda opt: (neg_logprobs.numel() // n_cols, )
            _backward[grid](neg_logprobs, indices, grad_out, n_cols)
            return neg_logprobs, None
    # Paddle 版自定义反向
else:
    class _cross_entropy(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, logits, indices):
            # indices 必须 int64
            assert indices.dtype == paddle.int64, "Indices are expected to be int64."
            n_cols = logits.shape[-1]
            # 用 like 保持 dtype/device
            result = paddle.empty_like(indices, dtype=logits.dtype)
            neg_logprobs = paddle.empty_like(logits, dtype=logits.dtype)
            grid = lambda opt: (logits.numel() // n_cols, )
            _forward[grid](logits, neg_logprobs, indices, result, n_cols)
            ctx.save_for_backward(neg_logprobs, indices)
            return result

        @staticmethod
        def backward(ctx, grad_out):
            neg_logprobs, indices = ctx.saved_tensor()
            n_cols = neg_logprobs.shape[-1]
            grid = lambda opt: (neg_logprobs.numel() // n_cols, )
            _backward[grid](neg_logprobs, indices, grad_out, n_cols)
            # 对应 forward(logits, indices)：返回 dlogits, None
            return neg_logprobs, None

    def cross_entropy(logits, indices):
        return _cross_entropy_paddle.apply(logits, indices)



cross_entropy = _cross_entropy.apply
