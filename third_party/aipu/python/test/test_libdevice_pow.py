import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit()
def pow_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.extra.aipu.libdevice.pow(x, y)
    tl.store(y_ptr + offsets, z, mask=mask)


def test_libdevice_pow():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, dtype=torch.float32, device=DEVICE)
    output_triton = torch.rand(size, device=DEVICE)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    pow_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
