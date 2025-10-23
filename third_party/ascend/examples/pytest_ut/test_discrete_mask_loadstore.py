import torch
import triton
import triton.language as tl
import torch_npu
import pytest


@triton.jit
def simple_discrete_mask_load_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offs = tl.arange(0, N)
    even_col_offs = tl.arange(0, N // 2) * 2
    even_col_mask = even_col_offs < N
    row_offs = tl.arange(0, M)
    row_mask = row_offs < M
    in_even_ptr = in_ptr + row_offs[:, None] * N + even_col_offs[None, :]
    in_odd_ptr = in_ptr + row_offs[:, None] * N + even_col_offs[None, :] + 1
    even_data = tl.load(in_even_ptr, mask=row_mask[:, None] & even_col_mask[None, :], other=0.0) 
    odd_data = tl.load(in_odd_ptr)
    rotated_data = tl.interleave(-odd_data, even_data)
    out_ptr = out_ptr + row_offs[:, None] * N + col_offs[None, :]
    tl.store(out_ptr, rotated_data)


@pytest.mark.parametrize("M", [(4)])
@pytest.mark.parametrize("N", [(8)])
def test_discrete_mask_load_store(M, N):
    input_tensor = torch.arange(M * N, dtype=torch.float16, device='npu').reshape(M, N)
    output_tensor = torch.empty_like(input_tensor)
    grid = (1,)
    simple_discrete_mask_load_kernel[grid](
        input_tensor,
        output_tensor,
        M=M,
        N=N,
    )
    even_cols = input_tensor[:, 0::2]
    odd_cols = input_tensor[:, 1::2]
    ref_output = torch.empty_like(input_tensor)
    ref_output[:, 0::2] = -odd_cols
    ref_output[:, 1::2] = even_cols
    assert torch.allclose(output_tensor.float(), ref_output.float())
