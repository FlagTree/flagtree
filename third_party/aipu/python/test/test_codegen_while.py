import numpy as np
from tvm.compass.dsl import BuildManager
from mlir import ir
from triton.backends.aipu.codegen import AIPUModule, CodeGenerator


def test_while():
    mod_str = """
    func.func @while_loop(%inp: memref<1xi32>, %out: memref<1xi32>) -> i32 {
      %c2_i32 = arith.constant 2 : i32
      %c10_i32 = arith.constant 10 : i32
      %i0 =  arith.constant 0 : index
      %inp_cast = memref.reinterpret_cast %inp to
        offset: [0], sizes: [1], strides: [1]
        : memref<1xi32> to memref<1xi32>
      %init = memref.load %inp_cast[%i0] : memref<1xi32>
      %res = scf.while (%i = %init) : (i32) -> (i32) {
        %cond = arith.cmpi slt, %i, %c10_i32 : i32
        scf.condition(%cond) %i : i32
      } do {
      ^bb0(%arg5: i32):
        %val = arith.addi %arg5, %c2_i32 : i32
        scf.yield %val : i32
      }
      %c0 = arith.constant 0 : i32
      %out_cast = memref.reinterpret_cast %out to
        offset: [0], sizes: [1], strides: [1]
        : memref<1xi32> to memref<1xi32>
      memref.store %res, %out_cast[%i0] : memref<1xi32>
      return %c0 : i32
    }"""
    mod = AIPUModule(ir.Module.parse(mod_str, ir.Context()))
    cg = CodeGenerator(mod)
    cg.mod.walk_mod(cg.dispatch)

    bm = BuildManager(disabled_pass=["tir.CommonSubexprElimTIR"])
    ex = bm.build(cg.ir_mod)
    print(ex.c_code)
    """
    __kernel void while_loop(__global int* var_4, __global int* var_11) {
      int var_5 = var_4[0];
      bool var_6 = (var_5 < 10);
      while (var_6){
        int var_7 = (var_5 + 2);
        var_5 = var_7;
        bool var_8 = (var_6 < 10);
        var_6 = var_8
      }
      var_11[0] = var_5;
    }
    """
    a = np.array([1], dtype=np.int32)
    aipu_out = np.empty((1, ), dtype=np.int32)
    ex(a, aipu_out)
    assert aipu_out[0] == 11


if __name__ == "__main__":
    test_while()
