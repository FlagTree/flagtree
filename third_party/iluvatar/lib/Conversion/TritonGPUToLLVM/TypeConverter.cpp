#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](MemDescType type) -> std::optional<Type> {
    return convertMemDescType(type);
  });
  addConversion([&](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
    return convertAsyncToken(type);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
#ifndef __ILUVATAR__
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
#endif
}

Type TritonGPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    auto rankedTensorType = cast<RankedTensorType>(pointeeType);
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, type.getAddressSpace()));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
}

Type TritonGPUToLLVMTypeConverter::getElementTypeForStruct(
    TensorOrMemDesc type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  auto dotOpLayout = mlir::dyn_cast<DotOperandEncodingAttr>(layout);
  if (!dotOpLayout)
    return elemTy;
  if (auto iluvatarmmaParent =
          mlir::dyn_cast<IluvatarMmaEncodingAttr>(dotOpLayout.getParent())) {
    if (iluvatarmmaParent.isVolta()) {
      int bitwidth = elemTy.getIntOrFloatBitWidth();
      if (bitwidth == 8)
        return vec_ty(elemTy, 8);
      return vec_ty(elemTy, 4);
    }
  }
  auto mmaParent =
      mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpLayout.getParent());
  if (!mmaParent || mmaParent.isHopper())
    return elemTy;
  int bitwidth = elemTy.getIntOrFloatBitWidth();
  assert(bitwidth <= 32);
  return IntegerType::get(ctx, 32);
}

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(cast<TensorOrMemDesc>(type));

  if (auto shared_layout = mlir::dyn_cast<SharedEncodingAttr>(layout)) {
    SmallVector<Type, 4> types;
    // base ptr
    auto ptrType = LLVM::LLVMPointerType::get(ctx, 3);
    types.push_back(ptrType);
    // shape dims
    auto rank = type.getRank();
    // offsets + strides
    for (auto i = 0; i < rank * 2; i++) {
      types.push_back(IntegerType::get(ctx, 32));
    }
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertMemDescType(MemDescType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  SmallVector<Type, 4> types;
  // base ptr
  auto ptrType = LLVM::LLVMPointerType::get(ctx, 3);
  types.push_back(ptrType);
  // shape dims
  auto rank = type.getShape().size();
  // offsets + strides
  for (auto i = 0; i < rank * 2; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertAsyncToken(
    triton::gpu::AsyncTokenType type) {
  return IntegerType::get(type.getContext(), 32);
}
