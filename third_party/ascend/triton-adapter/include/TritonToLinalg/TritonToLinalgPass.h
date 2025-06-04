//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "../../include/TritonToLinalg/Passes.h.inc"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass();

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter();
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  // grid构造 num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel);

  void convertTTFunc(triton::FuncOp func, const bool existDot);

  void addDynamicLegal(ConversionTarget &target,
                       TritonTypeConverter &tritonTypeConverter);

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

public:
  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};
} // namespace

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
