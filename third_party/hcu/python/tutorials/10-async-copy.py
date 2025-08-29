import torch

import triton
import triton.language as tl

import os
# os.environ["USE_TTGIR_LOC"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["HCUGCN_USE_BUFFER_OPS"] = "1"
# os.environ["LLVM_IR_ENABLE_DUMP"] = "1"
# os.environ["HCUGCN_ENABLE_DUMP"] = "1"

# torch.cuda.set_device(5)
# torch.set_printoptions(threshold=10_000, profile="full")

@triton.jit
def copy_kernel(S, D,
                    M: tl.constexpr,
                    N: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    ):
    start_m = tl.program_id(0)
    # block pointers
    S_block_ptr = tl.make_block_ptr(
        base=S,
        shape=(M, N),
        strides=(N, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=D,
        shape=(M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(0, N, BLOCK_N):
        s = tl.load(S_block_ptr)
        acc += s

        S_block_ptr = tl.advance(S_block_ptr, (0, BLOCK_N))
    
    tl.store(O_block_ptr, acc)

copy_kernel_ttgir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#loc = loc("/home/lisj/chaindot/01-copy.py":16:0)
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "hip:gfx928", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @copy_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.ptr_int32_range = 1 : i32} loc("/home/lisj/chaindot/01-copy.py":16:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.ptr_int32_range = 1 : i32} loc("/home/lisj/chaindot/01-copy.py":16:0)) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x1xi64, #blocked> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked> loc(#loc1)
    %c64_i64 = arith.constant 64 : i64 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c64_i32 : i32 loc(#loc3)
    %2 = arith.extsi %1 : i32 to i64 loc(#loc4)
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc5)
    %4 = arith.extsi %3 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc5)
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked> loc(#loc6)
    %6 = arith.muli %2, %c64_i64 : i64 loc(#loc6)
    %7 = arith.muli %5, %cst : tensor<64x1xi64, #blocked> loc(#loc6)
    %8 = tt.broadcast %7 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked> loc(#loc6)
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc5)
    %10 = arith.extsi %9 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc5)
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked> loc(#loc6)
    %12 = tt.broadcast %11 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked> loc(#loc6)
    %13 = arith.addi %8, %12 : tensor<64x64xi64, #blocked> loc(#loc6)
    %14 = tt.addptr %arg0, %6 : !tt.ptr<f32>, i64 loc(#loc5)
    %15 = arith.trunci %13 : tensor<64x64xi64, #blocked> to tensor<64x64xi32, #blocked> loc(#loc5)
    %16 = tt.splat %14 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc5)
    %17 = tt.addptr %16, %15 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc5)

    // %18 = tt.load %17 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc5)

    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst_4 = arith.constant dense<true> : tensor<64x64xi1, #blocked> loc(#loc1)
    %shared = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x64xf32, #shared, #triton_gpu.shared_memory, mutable> loc(#loc8)
    %view = triton_gpu.memdesc_subview %shared[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x64xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x64xf32, #shared, #triton_gpu.shared_memory, mutable> loc(#loc8)
    %copy = triton_gpu.async_copy_global_to_local %17, %view mask %cst_4 : tensor<64x64x!tt.ptr<f32>, #blocked> -> <64x64xf32, #shared, #triton_gpu.shared_memory, mutable> loc(#loc8)
    %commit = triton_gpu.async_commit_group %copy loc(#loc7)
    %wait = triton_gpu.async_wait %commit {num = 0 : i32} loc(#loc8)
    %18 = triton_gpu.local_load %view : !tt.memdesc<64x64xf32, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x64xf32, #blocked> loc(#loc8)
    triton_gpu.local_dealloc %shared : !tt.memdesc<1x64x64xf32, #shared, #triton_gpu.shared_memory, mutable> loc(#loc8)


    %19 = arith.addf %18, %cst_0 : tensor<64x64xf32, #blocked> loc(#loc7)
    %20 = tt.addptr %arg1, %6 : !tt.ptr<f32>, i64 loc(#loc6)
    %21 = tt.splat %20 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc6)
    %22 = tt.addptr %21, %15 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked> loc(#loc6)
    tt.store %22, %19 : tensor<64x64x!tt.ptr<f32>, #blocked> loc(#loc6)
    tt.return loc(#loc8)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/lisj/chaindot/01-copy.py":22:28)
#loc3 = loc("/home/lisj/chaindot/01-copy.py":28:27)
#loc4 = loc("/home/lisj/chaindot/01-copy.py":30:8)
#loc5 = loc("/home/lisj/chaindot/01-copy.py":44:20)
#loc6 = loc("/home/lisj/chaindot/01-copy.py":49:26)
#loc7 = loc("/home/lisj/chaindot/01-copy.py":45:15)
#loc8 = loc("/home/lisj/chaindot/01-copy.py":49:4)
"""

def copy(src):
    M, N = src.shape
    dst = torch.empty_like(src)

    BLOCK_M = M
    BLOCK_N = N

    # grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), )
    # kernel = copy_kernel[grid](src, dst, M, N, BLOCK_M=M, BLOCK_N=N, num_stages=1, num_warps=1)
    # print(kernel.asm)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as ttgir:
        ttgir.write(copy_kernel_ttgir)
        ttgir.flush()        
        copy_ttgir_kernel = triton.compile(ttgir.name)

    grid = (triton.cdiv(M, BLOCK_M), 1, 1)
    copy_ttgir_kernel[grid](src, dst)
    return dst

M, N = 64, 64
torch.cuda.manual_seed(0)
S = torch.randn(M, N, dtype=torch.float32, device="cuda")
D = copy(S)
# torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)
print(f"{S=}")
print(f"{D=}")
rtol = 0 if torch.version.hip is None else 2e-2
if torch.allclose(S, D, atol=2e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")