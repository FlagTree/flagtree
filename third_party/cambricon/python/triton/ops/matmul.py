import torch

import triton
import triton.language as tl

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [0]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 1 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                      num_stages=num_stages, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 180, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 384, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 256, 'SPLIT_K': 1}, num_stages=10, num_warps=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 180, 'BLOCK_N': 512, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 384, 'BLOCK_N': 512, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1024, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1024, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 256, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'SPLIT_K': 1}, num_stages=10, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_M': 2048, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1024, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
            EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr):
    # matrix multiplication
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid = pid_0 * grid_n + pid_1
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    pid_z = tl.program_id(2)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A, mask=(rm < M)[:, None], other=0.)
            b = tl.load(B, mask=(rn < N)[None, :], other=0.)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=(rk[None, :] < k_remaining) & (rm < M)[:, None], other=0.)
            b = tl.load(B, mask=(rk[:, None] < k_remaining) & (rn < N)[None, :], other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    #tl.store(C, acc, mask=mask)
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        # common type between a and b
        ab_dtype = get_higher_dtype(a.dtype, b.dtype)

        # allocates output
        output_dtype = ab_dtype
        if a.dtype in [torch.int8] or b.dtype in [torch.int8]:
            output_dtype = torch.float32
        c = torch.empty((M, N), device=device, dtype=output_dtype)

        # Allowed types for acc_type given the types of a and b.
        supported_acc_dtypes = {
            torch.float16: (torch.float32, torch.float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32, ), torch.int8: (torch.float32, )
        }

        acc_dtype = supported_acc_dtypes[ab_dtype][0]
        assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
        assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
        assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"
        if acc_dtype in [torch.bfloat16, torch.int8]:
            acc_dtype = tl.float32

        def to_tl_type(ty):
            return getattr(tl, str(ty).split(".")[-1])

        acc_dtype = to_tl_type(acc_dtype)
        ab_dtype = to_tl_type(ab_dtype)
        output_dtype = to_tl_type(output_dtype)

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                      GROUP_M=8, ACC_TYPE=acc_dtype)
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply
