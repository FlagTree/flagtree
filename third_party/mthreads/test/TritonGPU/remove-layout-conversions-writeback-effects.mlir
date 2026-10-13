// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=8" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=15" | FileCheck %s

#b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
  // An impure address producer must execute only once, keeping the value convert.
  // CHECK-LABEL: tt.func @impure_address
  // CHECK: tt.elementwise_inline_asm
  // CHECK-SAME: pure = false
  // CHECK-NOT: tt.elementwise_inline_asm
  // CHECK: ttg.convert_layout
  // CHECK-NOT: tt.elementwise_inline_asm
  // CHECK: tt.atomic_rmw
  // CHECK-NOT: tt.elementwise_inline_asm
  // CHECK: tt.return
  tt.func @impure_address(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %offset: i32) {
    %x = tt.splat %offset : i32 -> tensor<16x64xi32, #b>
    %i = tt.elementwise_inline_asm "mov $0, $1" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %x : tensor<16x64xi32, #b> -> tensor<16x64xi32, #b>
    %p = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %p, %i : tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xi32, #b>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }

  // The same chain marked pure still permits writeback rematerialization.
  // CHECK-LABEL: tt.func @pure_address
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.elementwise_inline_asm
  // CHECK-SAME: pure = true
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.atomic_rmw
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func @pure_address(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %offset: i32) {
    %x = tt.splat %offset : i32 -> tensor<16x64xi32, #b>
    %i = tt.elementwise_inline_asm "mov $0, $1" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %x : tensor<16x64xi32, #b> -> tensor<16x64xi32, #b>
    %p = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %p, %i : tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xi32, #b>
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }
}
