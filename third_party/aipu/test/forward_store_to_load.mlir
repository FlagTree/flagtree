
// RUN: aipu-opt %s  --forward-store-to-load --canonicalize | FileCheck %s

func.func @forward_single_store_to_load(%arg0: memref<10xi8>) -> i8 {
  %c7 = arith.constant 7 : i8
  %c0 = arith.constant 0 : index
  memref.store %c7, %arg0[%c0] : memref<10xi8>
  %1 = memref.load %arg0[%c0] : memref<10xi8>
  return %1: i8
}
// CHECK-LABEL: func.func @forward_single_store_to_load
// CHECK: %c7_i8 = arith.constant 7 : i8
// CHECK: %c0 = arith.constant 0 : index
// CHECK: memref.store %c7_i8, %arg0[%c0]
// CHECK-NEXT: return %c7_i8 : i8

func.func @remove_unused_store(%memref0: memref<10xi8>) -> i8 {
  %c0 = arith.constant 0 : index
  %val0 = arith.constant 7 : i8
  %val1 = arith.constant 8 : i8
  memref.store %val0, %memref0[%c0] : memref<10xi8>
  memref.store %val1, %memref0[%c0] : memref<10xi8>
  return %val0:i8
}
// CHECK-LABEL: func.func @remove_unused_store
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c7_i8 = arith.constant 7 : i8
// CHECK: %c8_i8 = arith.constant 8 : i8
// CHECK: memref.store %c8_i8, %arg0[%c0]
// CHECK-NEXT: return %c7_i8 : i8

func.func @for_loop_region(%memref0: memref<10xi8>, %memref1: memref<10xi8>) {
  %out = memref.alloc() : memref<10xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = memref.load %memref0[%i] : memref<10xi8>
    memref.store %0, %out[%i] : memref<10xi8>
    %1 = memref.load %out[%i] : memref<10xi8>
    memref.store %1, %memref1[%i] : memref<10xi8>
  }
  return
}
// CHECK-LABEL: func.func @for_loop_region
// CHECK: %c10 = arith.constant 10 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: scf.for %{{.*}} = %c0 to %c10 step %c1 {
// CHECK-NEXT:   %{{.*}} = memref.load %arg0[%{{.*}}] : memref<10xi8>
// CHECK-NEXT:   memref.store %{{.*}}, %arg1[%{{.*}}] : memref<10xi8>
// CHECK: }
// CHECK-NEXT: return

func.func @skip_different_blocks(%memref0: memref<10xi8>) -> i8 {
  %c0 = arith.constant 0 : index
  %val0 = arith.constant 7 : i8
  %true = arith.constant true
  scf.if %true {
    memref.store %val0, %memref0[%c0] : memref<10xi8>
  }
  %1 = memref.load %memref0[%c0] : memref<10xi8>
  return %1: i8
}
// CHECK-LABEL: func.func @skip_different_blocks
// CHECK: memref.load

func.func @skip_diff_indices(%memref0: memref<10xi8>) -> i8 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %val0 = arith.constant 7 : i8
  memref.store %val0, %memref0[%c0] : memref<10xi8>
  %2 = memref.load %memref0[%c1] : memref<10xi8>
  return %2: i8
}
// CHECK-LABEL: func.func @skip_diff_indices
// CHECK: memref.load

func.func @skip_intermediate_region(%memref0: memref<10xi8>) -> i8 {
  %c0 = arith.constant 0 : index
  %val0 = arith.constant 7 : i8
  %val1 = arith.constant 8 : i8
  %true = arith.constant true
  %0 = memref.alloc() : memref<1xi8>
  memref.store %val0, %memref0[%c0] : memref<10xi8>
  scf.if %true {
    memref.store %val1, %0[%c0] : memref<1xi8>
  }
  %2 = memref.load %memref0[%c0] : memref<10xi8>
  return %2: i8
}
// CHECK-LABEL: func.func @skip_intermediate_region
// CHECK: memref.load
