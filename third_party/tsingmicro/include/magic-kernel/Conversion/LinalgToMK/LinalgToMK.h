//===------------------- LinalgToMK.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering all linalg ops into mk ops.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_LINALG_TO_MK_H
#define ZTC_CONVERSION_LINALG_TO_MK_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

void populateLinalgToMKCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateLinalgToMKConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToMKPass();

} // namespace triton
} // namespace mlir

#endif // ZTC_CONVERSION_MEMREF_TO_MAGICKERNEL_H