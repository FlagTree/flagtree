// RUN: aipu-opt %s --convert-i64-to-i32 --canonicalize | FileCheck %s


func.func @replace_low_part_i64_with_self(%arg0: i32) -> i32 {
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %1 = arith.extsi %arg0 : i32 to i64
    %2 = arith.andi %1, %c4294967295_i64 : i64
    %3 = arith.trunci %2 : i64 to i32
  return %3 : i32
}
// CHECK-LABEL: func.func @replace_low_part_i64_with_self
// CHECK-NEXT: return %arg0 : i32

func.func @replace_high_part_i64_with_zero(%arg0: i32) -> i32 {
    %c32_i64 = arith.constant 32 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %1 = arith.extsi %arg0 : i32 to i64
    %2 = arith.shrsi %1, %c32_i64 : i64
    %3 = arith.andi %2, %c4294967295_i64 : i64
    %4 = arith.trunci %3 : i64 to i32
  return %4 : i32
}
// CHECK-LABEL: func.func @replace_high_part_i64_with_zero
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: return %c0_i32 : i32

func.func @replace_pack_i64_with_low(%hi: i32, %lo: i32) -> i32 {
    %c32_i64 = arith.constant 32 : i64
    %28 = arith.extui %hi : i32 to i64
    %29 = arith.shli %28, %c32_i64 : i64
    %30 = arith.extui %lo : i32 to i64
    %31 = arith.ori %29, %30 : i64
    %6 = arith.trunci %31 : i64 to i32
  return %6 : i32
}
// CHECK-LABEL: func.func @replace_pack_i64_with_low
// CHECK-NEXT: return %arg1 : i32

func.func @convert_simple_addi(%arg0: i32) -> i32 {
    %14 = arith.extsi %arg0 : i32 to i64
    %15 = arith.addi %14, %14 : i64
    %16 = arith.trunci %15 : i64 to i32
  return %16 : i32
}
// CHECK-LABEL: func.func @convert_simple_addi
// CHECK-NEXT: %0 = arith.addi %arg0, %arg0 : i32
// CHECK-NEXT: return %0 : i32


func.func @test_some_arith_ops(%ret:f32, %26:i32, %27:i32) -> f32 {
    %c32_i64 = arith.constant 32 : i64
    %cm1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %28 = arith.extui %26 : i32 to i64
    %29 = arith.shli %28, %c32_i64 : i64
    %30 = arith.extui %27 : i32 to i64
    %31 = arith.ori %29, %30 : i64
    %32 = arith.subi %cm1_i64, %31 : i64
    %33 = arith.cmpi slt, %31, %c0_i64 : i64
    %34 = arith.select %33, %32, %31 : i64
    %35 = arith.sitofp %34 : i64 to f32
    return %35 : f32
}
// CHECK-LABEL: func.func @test_some_arith_ops
// CHECK-NEXT: %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %0 = arith.subi %c-1_i32, %arg2 : i32
// CHECK-NEXT: %1 = arith.cmpi slt, %arg2, %c0_i32 : i32
// CHECK-NEXT: %2 = arith.select %1, %0, %arg2 : i32
// CHECK-NEXT: %3 = arith.sitofp %2 : i32 to f32
// CHECK-NEXT: return %3 : f32

func.func @test_scf_if_memref(%x:i32, %y:i32, %buf:memref<i64>) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  %2 = arith.cmpi slt, %x, %y : i32
  %3 = scf.if %2 -> (i64) {
    %z = memref.load %buf[] : memref<i64>
    scf.yield %z : i64
  } else {
    memref.store %c0_i64, %buf[] : memref<i64>
    scf.yield %c0_i64 : i64
  }
  return %3 : i64
}

// CHECK-LABEL: func.func @test_scf_if_memref(%arg0: i32, %arg1: i32, %arg2: memref<i32>) -> i32 {
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %0 = arith.cmpi slt, %arg0, %arg1 : i32
// CHECK-NEXT:  %1 = scf.if %0 -> (i32) {
// CHECK-NEXT:    %2 = memref.load %arg2[] : memref<i32>
// CHECK-NEXT:    scf.yield %2 : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    memref.store %c0_i32, %arg2[] : memref<i32>
// CHECK-NEXT:    scf.yield %c0_i32 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  return %1 : i32
// CHECK-NEXT:}


func.func @test_unrank_memref_type(%arg0: memref<*xi64>){
    return
}
// CHECK-LABEL: func.func @test_unrank_memref_type(%arg0: memref<*xi32>) {
// CHECK-NEXT:  return
// CHECK-NEXT:}

func.func @test_vector_const() -> vector<32xi32> {
  %3 = "arith.constant"() <{value = dense<0> : vector<32xi64>}> : () -> vector<32xi64>
  %4 = arith.trunci %3 : vector<32xi64> to vector<32xi32>
  return %4 : vector<32xi32>
}

// CHECK-LABEL:  func.func @test_vector_const() -> vector<32xi32> {
// CHECK-NEXT:  %cst = arith.constant dense<0> : vector<32xi32>
// CHECK-NEXT:  return %cst : vector<32xi32>
// CHECK-NEXT:}
