#ifndef MLIR_DIALECT_AIPU_PASSES
#define MLIR_DIALECT_AIPU_PASSES

include "mlir/Pass/PassBase.td"

def AIPUConvertBoolArg2I8 : Pass<"convert-bool-arg-to-i8", "func::FuncOp"> {
    let summary = "Convert bool arg in entry func to i8 type.";

    let dependentDialects = ["func::FuncDialect", "arith::ArithDialect"];
}

#endif // MLIR_DIALECT_AIPU_PASSES
