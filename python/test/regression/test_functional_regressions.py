import numpy as np
import pytest

HAS_TORCH = False
HAS_PADDLE = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    import paddle
    HAS_PADDLE = True

from numpy.random import RandomState

import triton
import triton.language as tl


def test_vecmat(device = 'cuda'):

    @triton.jit
    def batched_vecmat(
            # inputs
            A,  # shape: [dim_m, dim_k]
            B,  # shape: [dim_m, dim_n, dim_k]
            # dimensions
        dim_m, dim_n, dim_k,
            # outputs
            output,
            # block information
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
        m_index = tl.program_id(0)
        n_index = tl.program_id(1)
        # Output tile
        output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
            + (n_index * block_n + tl.arange(0, block_n))[None, :]

        vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
        k_blocks = dim_k // block_k
        for k_index in range(k_blocks):
            # Load A tile
            a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
                + (k_index * block_k + tl.arange(0, block_k))[None, :]
            a = tl.load(A + a_tile)

            # Load B tile, transposed to [n, m, k] in order to broadcast A on a
            # leading dimension.
            b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
                + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
                + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
            b = tl.load(B + b_tile)

            expanded_a, _ = tl.broadcast(a, b)
            vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

        tl.store(output + output_tile, vecmat)

    M, N, K = 128, 128, 128
    block_m, block_n, block_k = 16, 32, 64

    rs = RandomState(17)
    A_vec = rs.randint(0, 4, (M, K)).astype('float32')
    B_vec = rs.randint(0, 4, (M, N, K)).astype('float32')
    A = A_vec
    B = B_vec

    if HAS_PADDLE:
        A_tri = paddle.to_tensor(A)
        B_tri = paddle.to_tensor(B)
        C_tri = paddle.zeros((M, N), dtype='float32')
    else:
        A_tri = torch.tensor(A, device=device)
        B_tri = torch.tensor(B, device=device)
        C_tri = torch.zeros((M, N), dtype=torch.float32, device=device)

    grid = (M // block_m, N // block_n)

    batched_vecmat[grid](
        A_tri, B_tri, M, N, K, C_tri,  #
        block_m=block_m, block_n=block_n, block_k=block_k,  #
        num_warps=4, num_stages=1)

    A_expanded = A[:, np.newaxis, :]
    A_broadcasted = np.broadcast_to(A_expanded, (M, N, K))
    AB = A_broadcasted * B
    C_ref = np.sum(AB, axis=2)

    np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("type",
                         ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"])
def test_iv_dependent_matmul(type, device='cuda'):

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr,  #
               M, N, K,  #
               stride_am, stride_ak,  #
               stride_bk, stride_bn,  #
               stride_cm, stride_cn,  #
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
               type: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a_ptrs = a_ptr
        b_ptrs = b_ptr
        if type == "post_load_two_iters":
            a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
            a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if type == "pre_load":
                a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptr + k * BLOCK_SIZE_K * stride_bk
            elif type == "post_pre_mixed":
                a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b)
            if type == "post_load":
                a_ptrs = a_ptr + (k + 1) * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
            elif type == "post_pre_mixed":
                b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
            elif type == "post_load_two_iters":
                a_ptrs = a_ptrs_next
                b_ptrs = b_ptrs_next
                a_ptrs_next = a_ptr + (k + 2) * BLOCK_SIZE_K * stride_ak
                b_ptrs_next = b_ptr + (k + 2) * BLOCK_SIZE_K * stride_bk
            elif type == "post_load_three_iters":
                a_ptrs = a_ptrs_next
                b_ptrs = b_ptrs_next
                a_ptrs_next = a_ptrs_next_next
                b_ptrs_next = b_ptrs_next_next
                a_ptrs_next_next = a_ptr + (k + 3) * BLOCK_SIZE_K * stride_ak
                b_ptrs_next_next = b_ptr + (k + 3) * BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    if HAS_PADDLE:
        a = paddle.rand((M, K))
        b = paddle.rand((K, N))
        torch_output = paddle.matmul(a, b)
        triton_output = paddle.empty_like(torch_output)
    else:
        a = torch.rand((M, K), device=device)
        b = torch.rand((K, N), device=device)
        torch_output = torch.mm(a, b)
        triton_output = torch.empty_like(torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    num_stages = 4 if type == "post_load_three_iters" else 3
    
    def get_stride(shape: list[int]) -> list[int]:
    # row-major
        stride = [1] * len(shape)
        for i in reversed(range(len(shape) - 1)):
            stride[i] = stride[i + 1] * shape[i + 1]
        return stride

    a_stride = get_stride(a.shape)
    b_stride = get_stride(b.shape)
    triton_output_stride = get_stride(triton_output.shape)
    
    kernel[grid](
        a, b, triton_output, M, N, K,  #
        a_stride[0], a_stride[1], b_stride[0], b_stride[1],  #
        triton_output_stride[0], triton_output_stride[1],  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages)
    
    if HAS_PADDLE:
        assert paddle.allclose(torch_output, triton_output, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)
