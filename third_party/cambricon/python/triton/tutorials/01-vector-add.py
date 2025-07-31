"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import torch
import torch_mlu
import numpy as np

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 18432}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 32768}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 43520}, num_stages=0, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_stages=3, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 18432}, num_stages=3, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 21760}, num_stages=3, num_warps=1),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
        x_ptr: *Pointer* to first input vector.
        y_ptr: *Pointer* to second input vector.
        output_ptr: *Pointer* to output vector.
        n_elements: Size of the vector.
        BLOCK_SIZE: Number of elements each program should process. `constexpr` so it can be used as a shape value.
    """
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_jobs = tl.num_programs(axis=0)
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    # Add to tl.int64 for large tensor
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements

        # Load input from DRAM, masking out any extra elements in case the inputs is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        # Write output to DRAM.
        tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_mlu and y.is_mlu and output.is_mlu
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to MLU launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (min(triton.cdiv(n_elements, meta['BLOCK_SIZE']), 48), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable MLU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements)
    # We return a handle to z but, since `torch.mlu.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='mlu')
    y = torch.rand(size, device='mlu')
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='mlu', dtype=torch.float32)
    y = torch.rand(size, device='mlu', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
