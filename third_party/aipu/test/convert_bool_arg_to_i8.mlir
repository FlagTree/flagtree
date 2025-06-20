// RUN: aipu-opt %s --convert-bool-arg-to-i8 | FileCheck %s

// CHECK: func.func @test_func(%[[ARG0:.*]]: i8, %[[ARG1:.*]]: i32) -> i1
func.func @test_func(%arg0: i1, %arg1: i32) -> i1 {
  // CHECK: %0 = arith.trunci %arg0 : i8 to i1
  %true = arith.constant true
  %1 = arith.ori %arg0, %true : i1
  return %1 : i1
}
