//===--------------------- MKToTx81Pass.cpp -------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tsingmicro-tx81/Conversion/MKToTx81/MKToTx81.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "mk-to-tx81"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MKTOTX81
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class MKToTx81Pass : public triton::impl::MKToTx81Base<MKToTx81Pass> {
  using MKToTx81Base<MKToTx81Pass>::MKToTx81Base;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, arith::ArithDialect,
                    mk::MagicKernelDialect, tx::Tx81Dialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Register illegal ops for Dialect Conversion
    target.addIllegalDialect<linalg::LinalgDialect,
                             bufferization::BufferizationDialect,
                             mk::MagicKernelDialect>();

    target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                           math::MathDialect, affine::AffineDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           cf::ControlFlowDialect, tx::Tx81Dialect>();

    target.addIllegalOp<memref::CopyOp>();
    target.addLegalOp<ModuleOp>();

    triton::populateMKToTx81ConversionPatterns(patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createMKToTx81Pass() {
  return std::make_unique<MKToTx81Pass>();
}
