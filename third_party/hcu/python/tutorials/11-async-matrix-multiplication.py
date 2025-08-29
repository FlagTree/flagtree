"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = tl.program_id(axis=0)
#    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    pid_m = pid // grid_n
#    pid_n = pid % grid_n
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl

import os
# os.environ["USE_TTGIR_LOC"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
# os.environ["LLVM_IR_ENABLE_DUMP"] = "1"
# os.environ["HCUGCN_ENABLE_DUMP"] = "1"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' # and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        # triton.Config(
        #     {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
        #     num_warps=4, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
        #     num_warps=8, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
        #     num_warps=8, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
        #     num_warps=4, num_stages=2),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
        #     num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0},
            num_warps=1, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

# ttgir = "/home/lisj/chaindot/matmul_kernel_stage2.ttgir"
matmul_kernel_ttgir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc = loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0)
#mma = #triton_gpu.hcu_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false, interleave = false}>
#shared = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "hip:gfx928", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.ptr_int32_range = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.ptr_int32_range = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.ptr_int32_range = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg3: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg4: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg5: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg6: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg7: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0), %arg8: i32 {tt.divisibility = 16 : i32, tt.non_negative = 1 : i32} loc("/home/lisj/chaindot/03-matrix-multiplication.py":245:0)) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked> loc(#loc1)
    %c63_i32 = arith.constant 63 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c63_i32 : i32 loc(#loc48)
    %2 = arith.divsi %1, %c64_i32 : i32 loc(#loc49)
    %3 = arith.addi %arg4, %c127_i32 : i32 loc(#loc50)
    %4 = arith.divsi %3, %c128_i32 : i32 loc(#loc51)
    %5 = arith.divsi %0, %4 : i32 loc(#loc7)
    %6 = arith.subi %2, %5 : i32 loc(#loc8)
    %7 = arith.minsi %6, %c1_i32 : i32 loc(#loc9)
    %8 = arith.remsi %0, %4 : i32 loc(#loc10)
    %9 = arith.remsi %8, %7 : i32 loc(#loc11)
    %10 = arith.addi %5, %9 : i32 loc(#loc12)
    %11 = arith.divsi %8, %7 : i32 loc(#loc13)
    %12 = arith.muli %10, %c64_i32 : i32 loc(#loc14)
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc15)
    %14 = tt.splat %12 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc16)
    %15 = arith.addi %14, %13 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc16)
    %16 = tt.splat %arg3 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc17)
    %17 = arith.remsi %15, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc17)
    %18 = arith.muli %11, %c128_i32 : i32 loc(#loc18)
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc19)
    %20 = tt.splat %18 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc20)
    %21 = arith.addi %20, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc20)
    %22 = tt.splat %arg4 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc21)
    %23 = arith.remsi %21, %22 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc21)
    %24 = tt.expand_dims %17 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc22)
    %25 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked> loc(#loc22)
    %26 = arith.muli %24, %25 : tensor<64x1xi32, #blocked> loc(#loc22)
    %27 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked> loc(#loc23)
    %28 = tt.broadcast %26 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked> loc(#loc22)
    %29 = tt.broadcast %27 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked> loc(#loc22)
    %30 = arith.addi %28, %29 : tensor<64x128xi32, #blocked> loc(#loc22)
    %31 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f16>, i32 loc(#loc22)
    %32 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc24)
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked> loc(#loc25)
    %34 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked> loc(#loc25)
    %35 = arith.muli %33, %34 : tensor<128x1xi32, #blocked> loc(#loc25)
    %36 = tt.expand_dims %23 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked> loc(#loc25)
    %37 = tt.broadcast %35 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked> loc(#loc25)
    %38 = tt.broadcast %36 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked> loc(#loc25)
    %39 = arith.addi %37, %38 : tensor<128x128xi32, #blocked> loc(#loc25)
    %40 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f16>, i32 loc(#loc25)
    %41 = arith.addi %arg5, %c127_i32 : i32 loc(#loc52)
    %42 = arith.divsi %41, %c128_i32 : i32 loc(#loc53)
    %43 = arith.muli %arg7, %c128_i32 : i32 loc(#loc27)
    %44 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
    %45 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
    %46 = arith.cmpi sgt, %42, %c0_i32 : i32 loc(#loc30)
    %47 = tt.splat %arg5 : i32 -> tensor<1x128xi32, #blocked> loc(#loc31)
    %48 = arith.cmpi slt, %27, %47 : tensor<1x128xi32, #blocked> loc(#loc31)
    %49 = tt.broadcast %48 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc28)
    %50 = triton_gpu.memdesc_subview %44[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
    %51 = tt.splat %46 : i1 -> tensor<64x128xi1, #blocked> loc(#loc30)
    %52 = arith.andi %51, %49 : tensor<64x128xi1, #blocked> loc(#loc30)
    %53 = tt.splat %31 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked> loc(#loc28)
    %54 = tt.addptr %53, %30 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked> loc(#loc28)
    %55 = triton_gpu.async_copy_global_to_local %54, %50 mask %52 other %cst : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
    %56 = triton_gpu.async_commit_group %55 loc(#loc28)
    %57 = tt.splat %arg5 : i32 -> tensor<128x1xi32, #blocked> loc(#loc32)
    %58 = arith.cmpi slt, %33, %57 : tensor<128x1xi32, #blocked> loc(#loc32)
    %59 = tt.broadcast %58 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked> loc(#loc29)
    %60 = triton_gpu.memdesc_subview %45[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
    %61 = tt.splat %46 : i1 -> tensor<128x128xi1, #blocked> loc(#loc30)
    %62 = arith.andi %61, %59 : tensor<128x128xi1, #blocked> loc(#loc30)
    %63 = tt.splat %40 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked> loc(#loc29)
    %64 = tt.addptr %63, %39 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked> loc(#loc29)
    %65 = triton_gpu.async_copy_global_to_local %64, %60 mask %62 other %cst_0 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
    %66 = triton_gpu.async_commit_group %65 loc(#loc29)
    %67 = triton_gpu.async_wait %66 {num = 0 : i32} loc(#loc28)
    %68:7 = scf.for %arg9 = %c0_i32 to %42 step %c1_i32 iter_args(%arg10 = %cst_1, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %50, %arg14 = %60, %arg15 = %31, %arg16 = %40) -> (tensor<64x128xf32, #mma>, i32, i32, !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, !tt.ptr<f16>, !tt.ptr<f16>)  : i32 {
      %91 = arith.subi %42, %c1_i32 : i32 loc(#loc30)
      %92 = arith.cmpi slt, %arg9, %91 : i32 loc(#loc30)
      %93 = triton_gpu.local_load %arg13 : !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> loc(#loc28)
      %94 = triton_gpu.local_load %arg14 : !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc29)
      %95 = tt.dot %93, %94, %arg10 : tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x128xf32, #mma> loc(#loc33)
      hcugpu.instruction_sched_hint loc(#loc33)
      %96 = tt.addptr %arg15, %c128_i32 : !tt.ptr<f16>, i32 loc(#loc34)
      %97 = tt.addptr %arg16, %43 : !tt.ptr<f16>, i32 loc(#loc35)
      %98 = arith.addi %arg11, %c1_i32 : i32 loc(#loc30)
      %99 = arith.cmpi slt, %98, %c1_i32 : i32 loc(#loc30)
      %100 = arith.select %99, %98, %c0_i32 : i32 loc(#loc30)
      %101 = arith.addi %arg9, %c1_i32 : i32 loc(#loc30)
      %102 = arith.muli %101, %c128_i32 : i32 loc(#loc36)
      %103 = arith.subi %arg5, %102 : i32 loc(#loc37)
      %104 = tt.splat %103 : i32 -> tensor<1x128xi32, #blocked> loc(#loc31)
      %105 = arith.cmpi slt, %27, %104 : tensor<1x128xi32, #blocked> loc(#loc31)
      %106 = tt.broadcast %105 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc28)
      %107 = triton_gpu.memdesc_subview %44[%100, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
      %108 = tt.splat %92 : i1 -> tensor<64x128xi1, #blocked> loc(#loc30)
      %109 = arith.andi %108, %106 : tensor<64x128xi1, #blocked> loc(#loc30)
      %110 = tt.splat %96 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked> loc(#loc28)
      %111 = tt.addptr %110, %30 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked> loc(#loc28)
      %112 = triton_gpu.async_copy_global_to_local %111, %107 mask %109 other %cst : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
      %113 = triton_gpu.async_commit_group %112 loc(#loc28)
      %114 = tt.splat %103 : i32 -> tensor<128x1xi32, #blocked> loc(#loc32)
      %115 = arith.cmpi slt, %33, %114 : tensor<128x1xi32, #blocked> loc(#loc32)
      %116 = tt.broadcast %115 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked> loc(#loc29)
      %117 = triton_gpu.memdesc_subview %45[%100, %c0_i32, %c0_i32] : !tt.memdesc<1x128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
      %118 = tt.splat %92 : i1 -> tensor<128x128xi1, #blocked> loc(#loc30)
      %119 = arith.andi %118, %116 : tensor<128x128xi1, #blocked> loc(#loc30)
      %120 = tt.splat %97 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked> loc(#loc29)
      %121 = tt.addptr %120, %39 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked> loc(#loc29)
      %122 = triton_gpu.async_copy_global_to_local %121, %117 mask %119 other %cst_0 : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
      %123 = triton_gpu.async_commit_group %122 loc(#loc29)
      %124 = arith.addi %arg12, %c1_i32 : i32 loc(#loc30)
      %125 = arith.cmpi slt, %124, %c1_i32 : i32 loc(#loc30)
      %126 = arith.select %125, %124, %c0_i32 : i32 loc(#loc30)
      %127 = triton_gpu.memdesc_subview %44[%126, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc28)
      %128 = triton_gpu.async_wait %123 {num = 0 : i32} loc(#loc28)
      %129 = triton_gpu.memdesc_subview %45[%126, %c0_i32, %c0_i32] : !tt.memdesc<1x128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc29)
      scf.yield %95, %100, %126, %127, %129, %96, %97 : tensor<64x128xf32, #mma>, i32, i32, !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, !tt.ptr<f16>, !tt.ptr<f16> loc(#loc30)
    } loc(#loc30)
    %69 = triton_gpu.async_wait  {num = 0 : i32} loc(#loc30)
    triton_gpu.local_dealloc %44 : !tt.memdesc<1x64x128xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc30)
    triton_gpu.local_dealloc %45 : !tt.memdesc<1x128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc30)
    %70 = arith.truncf %68#0 : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma> loc(#loc38)
    %71 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc39)
    %72 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc40)
    %73 = arith.muli %arg8, %12 : i32 loc(#loc40)
    %74 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked> loc(#loc40)
    %75 = arith.muli %74, %72 : tensor<64x1xi32, #blocked> loc(#loc40)
    %76 = tt.addptr %arg2, %73 : !tt.ptr<f16>, i32 loc(#loc40)
    %77 = tt.expand_dims %21 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked> loc(#loc41)
    %78 = tt.broadcast %75 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked> loc(#loc42)
    %79 = tt.addptr %76, %18 : !tt.ptr<f16>, i32 loc(#loc42)
    %80 = arith.addi %29, %78 : tensor<64x128xi32, #blocked> loc(#loc42)
    %81 = tt.splat %arg3 : i32 -> tensor<64x1xi32, #blocked> loc(#loc43)
    %82 = arith.cmpi slt, %71, %81 : tensor<64x1xi32, #blocked> loc(#loc43)
    %83 = tt.splat %arg4 : i32 -> tensor<1x128xi32, #blocked> loc(#loc44)
    %84 = arith.cmpi slt, %77, %83 : tensor<1x128xi32, #blocked> loc(#loc44)
    %85 = tt.broadcast %82 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc45)
    %86 = tt.broadcast %84 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc45)
    %87 = arith.andi %85, %86 : tensor<64x128xi1, #blocked> loc(#loc45)
    %88 = triton_gpu.convert_layout %70 : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked> loc(#loc46)
    %89 = tt.splat %79 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked> loc(#loc46)
    %90 = tt.addptr %89, %80 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked> loc(#loc46)
    tt.store %90, %88, %87 : tensor<64x128x!tt.ptr<f16>, #blocked> loc(#loc46)
    tt.return loc(#loc47)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":268:24)
#loc3 = loc("/usr/local/lib/python3.10/site-packages/triton/language/standard.py":40:22)
#loc4 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":269:27)
#loc5 = loc("/usr/local/lib/python3.10/site-packages/triton/language/standard.py":40:28)
#loc6 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":270:27)
#loc7 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":272:22)
#loc8 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":274:35)
#loc9 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":274:48)
#loc10 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":275:34)
#loc11 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":275:54)
#loc12 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":275:27)
#loc13 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":276:40)
#loc14 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":285:23)
#loc15 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":285:51)
#loc16 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":285:38)
#loc17 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":285:68)
#loc18 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":286:23)
#loc19 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":286:51)
#loc20 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":286:38)
#loc21 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":286:68)
#loc22 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":288:22)
#loc23 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":288:60)
#loc24 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":289:29)
#loc25 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":289:22)
#loc26 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":297:33)
#loc27 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":306:33)
#loc28 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":300:20)
#loc29 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":301:20)
#loc30 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":297:22)
#loc31 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":300:51)
#loc32 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":301:51)
#loc33 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":303:35)
#loc34 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":305:18)
#loc35 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":306:18)
#loc36 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":300:59)
#loc37 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":300:55)
#loc38 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":311:23)
#loc39 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":317:41)
#loc40 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":317:21)
#loc41 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":317:72)
#loc42 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":317:52)
#loc43 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":318:33)
#loc44 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":318:58)
#loc45 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":318:39)
#loc46 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":319:21)
#loc47 = loc("/home/lisj/chaindot/03-matrix-multiplication.py":319:4)
#loc48 = loc(callsite(#loc3 at #loc4))
#loc49 = loc(callsite(#loc5 at #loc4))
#loc50 = loc(callsite(#loc3 at #loc6))
#loc51 = loc(callsite(#loc5 at #loc6))
#loc52 = loc(callsite(#loc3 at #loc26))
#loc53 = loc(callsite(#loc5 at #loc26))
"""

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.

    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # matmul_kernel[grid](
    #     a, b, c,  #
    #     M, N, K,  #
    #     a.stride(0), a.stride(1),  #
    #     b.stride(0), b.stride(1),  #
    #     c.stride(0), c.stride(1),  #
    #     ACTIVATION=activation  #
    # )

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as ttgir:
        ttgir.write(matmul_kernel_ttgir)
        ttgir.flush()        
        matmul_ttgir_kernel = triton.compile(ttgir.name)

    META = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0}
    matmul_ttgir_kernel = triton.compile(ttgir)
    grid = (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 1, 1)
    matmul_ttgir_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0),
        b.stride(0),
        c.stride(0),
    )
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
torch.cuda.set_device(5)
torch.manual_seed(0)
a = torch.randn((128, 128), device='cuda', dtype=torch.float16)
b = torch.randn((128, 128), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")


rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)
