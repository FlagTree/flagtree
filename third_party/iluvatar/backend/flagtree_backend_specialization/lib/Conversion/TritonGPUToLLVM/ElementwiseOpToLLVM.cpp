#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"

namespace mlir::triton::gpu {

SmallVector<Value> reorderValues(const SmallVector<Value> &values, Type inType,
                                 Type ouType) {
  return values;
}

SmallVector<Value> unpackI32(const SmallVector<Value> &inValues, Type srcTy,
                             ConversionPatternRewriter &rewriter, Location loc,
                             const LLVMTypeConverter *typeConverter) {
  return inValues;
}

SmallVector<Value> packI32(const SmallVector<Value> &inValues, Type srcTy,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter) {
  return inValues;
}

} // namespace mlir::triton::gpu
