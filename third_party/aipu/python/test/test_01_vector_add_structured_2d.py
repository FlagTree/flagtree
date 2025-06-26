import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel_structured_2d(x_ptr, y_ptr, output_ptr, M, N, stride_xm, stride_xn, stride_ym, stride_yn, stride_om,
                             stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    x_vals = tl.load(x_block_ptr)
    y_vals = tl.load(y_block_ptr)
    out_vals = x_vals + y_vals
    tl.store(output_block_ptr, out_vals)


def add_structured_2d(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)

    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 512

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    add_kernel_structured_2d[grid](
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


def test_add_structured_2d():
    torch.manual_seed(0)

    M, N = 128, 4096
    x = torch.rand((M, N), device=DEVICE)
    y = torch.rand((M, N), device=DEVICE)

    output_torch = x.cpu() + y.cpu()
    output_triton = add_structured_2d(x, y)

    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
    assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)


if __name__ == "__main__":
    test_add_structured_2d()
