"""
Libdevice (`tl.extra.mlu.libdevice`) function
==============================
Triton can invoke a custom function from an external library.
In this example, we will use the `libdevice` library (a.k.a `math` in triton) to apply `asin` on a tensor.
In `triton/language/math.py`, we try to aggregate functions with the same computation but different data types together.
Using triton, you can simply call `tl.math.asin`.
Triton automatically selects the correct underlying device function to invoke based on input and output types.
"""

# %%
#  asin Kernel
# ------------

import os
import torch
import torch_mlu

import triton
import triton.language as tl
from triton.language.extra.mlu import libdevice
from triton.backends.mlu.driver import BangDriver


@triton.jit
def asin_kernel(
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
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
#  Using the default libdevice library path
# -----------------------------------------
# We can use the default libdevice library path encoded in `triton/language/math.py`

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='mlu')
    output_triton = torch.zeros(size, device='mlu')
    output_torch = torch.asin(x)
    assert x.is_mlu and output_triton.is_mlu
    n_elements = output_torch.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
#  Customize the libdevice library path
# -------------------------------------
# We can also customize the libdevice library path by passing the path to the `libdevice` library to the `asin` kernel.

if __name__ == "__main__":

    def get_bc_file():
        home = os.getenv('NEUWARE_HOME', '/usr/local/neuware')
        bang_driver = BangDriver()
        capability = bang_driver.get_current_target().arch // 100
        assert capability in [3, 5]
        return f'{home}/mlvm/libdevice/libdevice.compute_{capability}0.bc'

    output_triton = torch.empty_like(x)
    asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs={'libdevice': get_bc_file()})
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
