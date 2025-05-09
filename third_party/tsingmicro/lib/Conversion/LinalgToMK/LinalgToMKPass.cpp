//===------------------- LinalgToMKPass.cpp -------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "linalg-to-mk"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LINALGTOMK
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LinalgToMKPass : public triton::impl::LinalgToMKBase<LinalgToMKPass> {
  using LinalgToMKBase<LinalgToMKPass>::LinalgToMKBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    arith::ArithDialect, mk::MagicKernelDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // TODO: Enable this when all conversion pattern has been implemented.
    //target.addIllegalDialect<linalg::LinalgDialect>();

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        mk::MagicKernelDialect>();

    target.addLegalOp<ModuleOp>();

    triton::populateLinalgToMKConversionPatterns(patterns);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createLinalgToMKPass() {
  return std::make_unique<LinalgToMKPass>();
}
