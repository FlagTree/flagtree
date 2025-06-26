import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel_2d(x_ptr, y_ptr, output_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn, stride_outm, stride_outn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = row_offsets < M
    mask_n = col_offsets < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + row_offsets[:, None] * stride_xm + col_offsets[None, :] * stride_xn
    y_ptrs = y_ptr + row_offsets[:, None] * stride_ym + col_offsets[None, :] * stride_yn
    out_ptrs = output_ptr + row_offsets[:, None] * stride_outm + col_offsets[None, :] * stride_outn

    x_vals = tl.load(x_ptrs, mask=mask)
    y_vals = tl.load(y_ptrs, mask=mask)
    out_vals = x_vals + y_vals
    tl.store(out_ptrs, out_vals, mask=mask)


def add_2d(x: torch.Tensor, y: torch.Tensor):

    M, N = x.shape
    output = torch.empty_like(x)

    BLOCK_M = 32
    BLOCK_N = 512

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    add_kernel_2d[grid](
        x,
        y,
        output,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return output.cpu()


def test_vector_add_2d():
    torch.manual_seed(0)

    M, N = 129, 4432
    x = torch.rand((M, N), device=DEVICE)
    y = torch.rand((M, N), device=DEVICE)

    output_torch = x.cpu() + y.cpu()
    output_triton = add_2d(x, y)

    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
    assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)


if __name__ == "__main__":
    test_vector_add_2d()
