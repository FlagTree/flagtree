"""
Vector Addition
===============

This test case is to test the aipu using shared memory:

* Through the HINT (# @hint: shared_memory) to use the shared_memory

* The size of Shared_memory is 256KB and when the data type is f32, it can store 64KB of data.
  For this example, there are 2 load operations, and 4 tecs access blockSize * 2 * 4 data in total.
  When the blockSize is 8192 (8KB), a total of 64KB data is accessed, which is the boundary value.
  If the blockSize exceeds this value, the dma operation will be out of bounds.

* When the block size exceeds the boundary value,
  the aipu backend needs to identify it and report an error.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)  # @hint: shared_memory
    y = tl.load(y_ptr + offsets, mask=mask)  # @hint: shared_memory
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    # Here are five sets of tests that currently get stuck when blockSize is greater than 8192;
    # Todo: When the blockSize is greater than 8192, codeGen needs to recognize it and report an error.

    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=8192)
    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16384)
    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=32768)
    # add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=65536)

    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output.cpu()


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:


def test_vector_add():
    torch.manual_seed(0)
    # size = 256
    size = 100000
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x.cpu() + y.cpu()

    output_triton = add(x, y)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
    assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)


if __name__ == "__main__":
    test_vector_add()
