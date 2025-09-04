import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import os
#os.environ["MFMA_TYPE"] = "16"

tuning_full_space = False


def get_full_tuning_space():
    configs = []
    if not tuning_full_space:
        return configs

    # block_m_range = [32, 64, 128, 256]
    # block_n_range = [32, 64, 128]
    # block_k_range = [16, 32, 64, 128]
    # num_warps_range = [2, 4, 8]
    # group_m_range = [4, 8]

    # block_m_range = [32]
    # block_n_range = [32]
    # block_k_range = [64]
    # num_warps_range = [4]
    # group_m_range = [4]

    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    # num_stage_range = [2]

    #block_m_range = [32,64,128,256]
    block_m_range = [32, 64, 128]
    block_n_range = [32, 64, 128]
    block_k_range = [16, 32, 64]
    num_warps_range = [4, 8]
    group_m_range = [1, 4]
    num_stage_range = [0, 1]
    num_ldmatrixes_range = [1]
    for block_m in block_m_range:
        for block_n in block_n_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for num_stages in num_stage_range:
                            for num_ld in num_ldmatrixes_range:
                                configs.append(
                                    triton.Config(
                                        {
                                            'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k,
                                            'GROUP_SIZE_M': group_m
                                        }, num_stages=num_stages, num_warps=num_warps, num_ldmatrixes=num_ld))
    return configs


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_full_tuning_space() if tuning_full_space else [
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=8),

        # [(2048, 11008, 4096),]
        #triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=0, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_warps=8,
                      num_ctas=1, num_stages=1, num_ldmatrixes=1)
        #triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4}, num_warps=8, num_ctas=1, num_stages=4, num_ldmatrixes=1)
    ],
    key=['M', 'N', 'K'],
)
# @triton.heuristics({
#     'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
# })
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
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
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        # a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)  # (offs_am[:, None]< M ) &
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)  # (offs_am[:, None]< M ) &
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        # a = tl.load(a_ptrs)
        # b = tl.load(b_ptrs)

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):

    # if a.shape[1] > 2048:
    #     pad = (0,32,0,0)
    #     a_pad = F.pad(a, pad, 'constant', 0)
    #     a = a_pad[:, 0:a.shape[1]]

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        ACTIVATION=activation)
    if os.getenv("TRITON_GEMM_LOG", "0") == "1":
        size_str = f'size, (M: {M}, N: {N}, K: {K})'
        print(f"best config: {matmul_kernel.best_config}, {size_str}")
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).


def unitTest(M, N, K):
    torch.manual_seed(0)

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_output = matmul(a, b).to("cpu")

    torch_output = torch.matmul(a, b).to("cpu")
    diff_mask = triton_output != torch_output
    num_diff_elements = torch.sum(diff_mask).item()
    total_elements = diff_mask.numel()
    ratio_diff = num_diff_elements / total_elements
    diff_positions = torch.nonzero(diff_mask)
    print(f"\n不同元素数量: {num_diff_elements}")
    print(f"总元素数量: {total_elements}")
    print(f"不同元素的比例: {ratio_diff:.2%}")
    with open('diff_positions.txt', 'w') as f:
        for pos in diff_positions:
            f.write(
                f"位置 {pos.tolist()} - triton_output: {triton_output[tuple(pos)]}, torch_output: {torch_output[tuple(pos)]}\n"
            )
    torch.set_printoptions(threshold=float('1000'))
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


x_vals = [
    # (64,37888,3584),
    # (128,37888,3584),
    # (256,37888,3584),
    # (512,37888,3584),
    # (1024,37888,3584),
    # (2048,37888,3584),
    # (4096,37888,3584),
    # (1,3584, 3584),
    # (2,3584, 3584),
    # (4,3584, 3584),
    # (8,3584, 3584),
    # (16,3584, 3584),
    # (32,3584, 3584),
    # (64,3584, 3584),
    # (128,3584, 3584),
    # (256,3584, 3584),
    # (512,3584, 3584),
    # (1024,3584, 3584),
    # (2048,3584, 3584),
    #(4096,3584, 3584),

    # (64,4608, 3584),
    # (128,4608, 3584),
    # (256,4608, 3584),
    # (512,4608, 3584),
    # (1024,4608, 3584),
    # (2048,4608, 3584),
    #(4096,4608, 3584),
    #1,4608, 3584),
    # (2,4608, 3584),
    # (4,4608, 3584),
    # (8,4608, 3584),
    # (16,4608, 3584),
    # (32,4608, 3584),
    #(2048, 4096, 4096),
    (22016, 2000, 4096),
    # (64,3584, 18944),
    # (128,3584, 18944),
    # (256,3584, 18944),
    # (512,3584, 18944),
    # (1024,3584, 18944),
    # (2048,3584, 18944),
    #(4096,3584, 18944),

    # gemv
    # (1, 3584, 3584),
    # (2, 3584, 3584),
    # (4, 3584, 3584),
    # (8, 3584, 3584),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        # x_vals=[
        #    128 * i for i in range(2,33)
        # ],  # Different possible values for `x_name`
        x_vals=x_vals,
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['rocblas', 'triton'],
        # Label name for the lines
        line_names=["rocBLAS(T)", "Triton(T)"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rocblas':
        ms, min_ms, max_ms = 1, 1, 1  #triton.testing.do_bench(lambda: torch.matmul(a, b),quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":

    # default tuning_full_space = False, if you should get the best performence of gemm size, please set tuning_full_space = True

    # print the gemm size and best config, please set environment variable TRITON_GEMM_LOG=1,
    # M=1,N=4608,K=3584
    for M, N, K in x_vals:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        matmul(a, b)
    # benchmark
    #benchmark.run(show_plots=False, print_data=True)

    #unit test
    for M, N, K in x_vals:
        unitTest(M, N, K)
