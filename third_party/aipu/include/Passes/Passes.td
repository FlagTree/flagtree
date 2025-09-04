#ifndef MLIR_DIALECT_AIPU_PASSES
#define MLIR_DIALECT_AIPU_PASSES

include "mlir/Pass/PassBase.td"

def AIPUConvertBoolArg2I8 : Pass<"convert-bool-arg-to-i8", "func::FuncOp"> {
    let summary = "Convert bool arg in entry func to i8 type.";

    let dependentDialects = ["func::FuncDialect", "arith::ArithDialect"];
}

// The following code snippet is from:
// https://chromium.googlesource.com/external/github.com/llvm/llvm-project/+/refs/heads/upstream/users/matthias-springer/scf_bufferization_preprocessing
// License: Apache 2.0
// Modifications: None, used as-is
def SCFLoopBufferizationPreprocessing
    : Pass<"scf-loop-bufferization-preprocessing"> {
  let summary = "Preprocess loops before One-Shot Bufferize";
  let description = [{
    Preprocess `scf.for` loops before running One-Shot Bufferize to support
    loops where a yielded tensor is not equivalent to the respective iter_arg.
    Such IR is currently not supported by One-Shot Bufferize.
    This pass inserts a `bufferization.materialize_in_destination` op for every
    yielded tensor, such that the yielded value is guaranteed to materialize in
    the future buffer of the iter_arg; this is done by copying the tensor
    contents into the iter_arg buffer. Such memcpys are a no-op in case the
    tensor contents already materialize in the iter_arg buffer.
  }];
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "scf::SCFDialect"];
}

def FlattenMemrefsPass : Pass<"flatten-memref"> {
  let summary = "Flatten a multiple dimensional memref to 1-dimensional";
  let description = [{

  }];
  let dependentDialects = [
      "affine::AffineDialect", "memref::MemRefDialect", "vector::VectorDialect"
  ];
}

#endif // MLIR_DIALECT_AIPU_PASSES
