// The following code snippet is from:
// https://chromium.googlesource.com/external/github.com/llvm/llvm-project/+/refs/heads/upstream/users/matthias-springer/scf_bufferization_preprocessing
// License: Apache 2.0
// Modifications:aipu namespace, add condition `if(operand.get() != bbArg)`

#include "Passes/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace aipu {

#define GEN_PASS_DEF_SCFLOOPBUFFERIZATIONPREPROCESSING
#include "Passes/Passes.h.inc"

using namespace mlir::scf;

struct SCFLoopBufferizationPreprocessingPass
    : public impl::SCFLoopBufferizationPreprocessingBase<
          SCFLoopBufferizationPreprocessingPass> {
  void runOnOperation() override {
    OpBuilder builder(getOperation()->getContext());
    getOperation()->walk([&](scf::YieldOp yieldOp) {
      builder.setInsertionPoint(yieldOp);
      // TODO: Support scf.while.
      auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
      if (!forOp)
        return WalkResult::skip();
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        auto tensorType = dyn_cast<TensorType>(operand.get().getType());
        if (!tensorType)
          continue;
        auto bbArg = forOp.getRegionIterArgs()[operand.getOperandNumber()];
        if (operand.get() != bbArg) {
          Value materialized =
              builder
                  .create<bufferization::MaterializeInDestinationOp>(
                      yieldOp.getLoc(), tensorType, operand.get(), bbArg)
                  .getResult();
          operand.set(materialized);
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace aipu

} // namespace mlir
