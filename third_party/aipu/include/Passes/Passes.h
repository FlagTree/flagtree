#ifndef MLIR_DIALECT_AIPU_PASSES_H
#define MLIR_DIALECT_AIPU_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/ErrorHandling.h" // llvm_unreachable

namespace mlir {

namespace aipu {

#define GEN_PASS_DECL
#include "Passes/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes/Passes.h.inc"

} // namespace aipu

} // namespace mlir

#endif // MLIR_DIALECT_AIPU_PASSES_H
