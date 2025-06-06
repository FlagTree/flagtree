from mlir import ir as mlir_ir
from triton.backends.aipu.analysis import determine_vectorization_factor


def get_module(mod_str):
    return mlir_ir.Module.parse(mod_str, mlir_ir.Context())


def test_no_affine_for(target_vec_register_bit=256):
    mod_str = """module {
  func.func @no_affinefor() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %c0, %c1 : i32
    return
  }
}"""
    mod = get_module(mod_str)
    vfactor = determine_vectorization_factor(mod, target_vec_register_bit, True)
    assert vfactor == 1


def test_affine_for_int8(target_vec_register_bit=256):
    mod_str = """module {
  func.func @add_kernel(%arg0: memref<128xi8>, %arg1: memref<128xi8>) {
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[%arg2] : memref<128xi8>
      %1 = arith.addi %0, %0 : i8
      affine.store %1, %arg1[%arg2] : memref<128xi8>
    }
    return
  }
}
"""
    mod = get_module(mod_str)
    vfactor = determine_vectorization_factor(mod, target_vec_register_bit, True)
    assert vfactor == (target_vec_register_bit // 8)


def test_affine_for_fp16_fp32(target_vec_register_bit=256):
    mod_str = """module {
  func.func @add_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf16>) {
    affine.for %i = 0 to 128 {
      %val_f32 = affine.load %arg0[%i] : memref<128xf32>
      %sum_f32 = arith.addf %val_f32, %val_f32 : f32
      affine.store %sum_f32, %arg0[%i] : memref<128xf32>

      %val_f16 = affine.load %arg1[%i] : memref<128xf16>
      %sum_f16 = arith.addf %val_f16, %val_f16 : f16
      affine.store %sum_f16, %arg1[%i] : memref<128xf16>
    }
    return
  }
}
"""
    mod = get_module(mod_str)
    vfactor = determine_vectorization_factor(mod, target_vec_register_bit, True)
    assert vfactor == (target_vec_register_bit // 16)


def test_affine_for_vector_fp32(target_vec_register_bit=256):
    mod_str = """module {
  func.func @add_kernel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 128 step 4 {
      %0 = vector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, vector<4xf32>
      %1 = arith.addf %0, %0 : vector<4xf32>
      vector.transfer_write %1, %arg1[%arg2] : vector<4xf32>, memref<128xf32>
    }
    return
  }
}
"""
    mod = get_module(mod_str)
    vfactor = determine_vectorization_factor(mod, target_vec_register_bit, True)
    assert vfactor == (target_vec_register_bit // 32)


if __name__ == "__main__":
    test_no_affine_for(target_vec_register_bit=256)
    test_affine_for_int8(target_vec_register_bit=256)
    test_affine_for_fp16_fp32(target_vec_register_bit=256)
    test_affine_for_vector_fp32(target_vec_register_bit=256)

    test_affine_for_int8(target_vec_register_bit=512)
