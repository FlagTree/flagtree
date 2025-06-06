import numpy as np
from mlir import ir
from tvm import aipu
from tvm.aipu import testing
from triton.backends.aipu.codegen import AIPUModule, CodeGenerator


def test_dma():
    mod_str = """
  func.func @dma_ops(%inp: memref<16xf32>, %out: memref<16xf32>) {
    %c0 = arith.constant 0 : index

    %lsram = memref.alloc() : memref<16 x f32, affine_map<(d0) -> (d0)>, 4>
    %tag = memref.alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 11>

    %num_elements = arith.constant 16 : index

    memref.dma_start %inp[%c0], %lsram[%c0], %num_elements, %tag[%c0] : memref<16 x f32>, memref<16 x f32, 4>, memref<1 x i32, 11>
    memref.dma_start %lsram[%c0], %out[%c0], %num_elements, %tag[%c0] : memref<16 x f32, 4>, memref<16 x f32>, memref<1 x i32, 11>

    memref.dma_wait %tag[%c0], %num_elements : memref<1 x i32, 11>
    return
  }"""
    mod = AIPUModule(ir.Module.parse(mod_str, ir.Context()))
    cg = CodeGenerator(mod)
    cg.mod.walk_mod(cg.dispatch)

    bm = aipu.tir.BuildManager()
    ex = bm.build(cg.prim_func)
    print(ex.c_code)
    """
    __kernel void dma_ops(__global float* var_3, __global float* var_5) {
      *addr_of_event_state() = 0;
      __lsram float buf[16];
      int cse_var_1 = (16 * 4);
      AsyncDmaDirect_kGlobal_to_kLsram((int)buf, (int)var_3, cse_var_1, cse_var_1, cse_var_1, cse_var_1, alloc_event());
      AsyncDmaDirect_kLsram_to_kGlobal((int)var_5, (int)buf, cse_var_1, cse_var_1, cse_var_1, cse_var_1, alloc_event());
      wait_events((1 << alloc_event()));
      barrier(CLK_LOCAL_MEM_FENCE);return;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    """
    a = np.array(list(range(16)), dtype=np.float32)
    aipu_out = np.empty((16, ), dtype=np.float32)
    ex(a, aipu_out)
    testing.assert_allclose(a, aipu_out)


if __name__ == "__main__":
    test_dma()
