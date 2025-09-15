import triton
import numpy as np
from mlir import ir
from tvm.compass.dsl.testing import assert_allclose, rand
from triton.backends.aipu.codegen import codegenAIPU
from triton.backends.aipu import transform


def get_vadd_mlir():
    mod_str = """
module {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %2 = arith.addi %1, %c1024 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.minsi %2, %3 : index
    %5 = arith.maxsi %4, %1 : index
    %6 = arith.subi %5, %1 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<1xi32, 11 : i32>
    memref.dma_start %subview[%c0], %subview_0[%c0], %6, %alloc_1[%c0] : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1]>>, memref<1xi32, 11 : i32>
    memref.dma_wait %alloc_1[%c0], %6 : memref<1xi32, 11 : i32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<1024xf32>
    %subview_4 = memref.subview %reinterpret_cast_2[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    %alloc_6 = memref.alloc() : memref<1xi32, 11 : i32>
    memref.dma_start %subview_4[%c0], %subview_5[%c0], %6, %alloc_6[%c0] : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1]>>, memref<1xi32, 11 : i32>
    memref.dma_wait %alloc_6[%c0], %6 : memref<1xi32, 11 : i32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    %c0_8 = arith.constant 0 : index
    %c1024_9 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    scf.for %arg10 = %c0_8 to %c1024_9 step %c8 {
      %cst = arith.constant 0.000000e+00 : f32
      %7 = vector.transfer_read %alloc[%arg10], %cst : memref<1024xf32>, vector<8xf32>
      %cst_13 = arith.constant 0.000000e+00 : f32
      %8 = vector.transfer_read %alloc_3[%arg10], %cst_13 : memref<1024xf32>, vector<8xf32>
      %9 = arith.addf %7, %8 : vector<8xf32>
      vector.transfer_write %9, %alloc_7[%arg10] : vector<8xf32>, memref<1024xf32>
    }
    %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %subview_11 = memref.subview %alloc_7[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    %subview_12 = memref.subview %reinterpret_cast_10[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    memref.copy %subview_11, %subview_12 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    return
  }
}
"""
    return mod_str


def test_dma():
    mod_str = get_vadd_mlir()
    ctx = ir.Context()
    mod = ir.Module.parse(mod_str, ctx)
    transform.binding_tid(mod, ctx)

    ex = codegenAIPU(mod)

    size = 4097
    BLOCKSIZE = 1024
    dtype = "float32"
    a = rand(size, dtype)
    b = rand(size, dtype)
    aipu_out = np.empty(size, dtype=dtype)

    gridX = triton.cdiv(size, BLOCKSIZE)
    np_args = [a, b, aipu_out, size]
    tail_args = [gridX, 1, 1, 0, 0, 0]
    tec_num = 4
    for i in range((gridX + tec_num - 1) // tec_num):
        tail_args[3] = i
        ex(*(np_args + tail_args))

    assert_allclose(aipu_out, a + b)


if __name__ == "__main__":
    test_dma()
