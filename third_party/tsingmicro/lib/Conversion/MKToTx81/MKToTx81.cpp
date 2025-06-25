//===--------------------- MKToTx81.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to convert operations from mk dialect to
// tx81 dialect. It converts memory operations to RdmaOp/WdmaOp and converts
// mk.dot to tx.gemm etc.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/MKToTx81/MKToTx81.h"
#include "Tx81/tx81.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "utils/FusionHelper.h"
#include "llvm/ADT/TypeSwitch.h"

// #define DEBUG_TYPE "mk-to-tx81"

using namespace mlir;
using namespace tx;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

class MKToTx81TypeConverter : public TypeConverter {
public:
  MKToTx81TypeConverter() {
    // Add conversions for MemRef types to UI64 (representing SPM addresses)
    addConversion([](MemRefType type) -> Type {
      return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
    });

    // Add conversions for Tensor types to UI64 (representing SPM addresses)
    addConversion([](TensorType type) -> Type {
      return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
    });

    // Keep other types as is
    addConversion([](Type type) -> Type { return type; });
  }

private:
  MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Get format code for tensor element type
// This maps MLIR types to Tx81 format codes
Data_Format getFormatCode(MemRefType type) {
  auto elemType = type.getElementType();
  if (elemType.isF32()) {
    return Fmt_FP32;
  } else if (elemType.isF16()) {
    return Fmt_FP16;
  } else if (elemType.isBF16()) {
    return Fmt_BF16;
  } else if (elemType.isInteger(8)) {
    return Fmt_INT8;
  } else {
    llvm_unreachable("Tx8 unsupported the element type\n");
  }
  // Default to F32 format
  return Fmt_FP32;
}

bool isSupportedType(MemRefType type) {
  auto elemType = type.getElementType();
  return elemType.isF32() || elemType.isF16() || elemType.isBF16() ||
         elemType.isInteger(8);
}

// Helper function to extract shape from tensor type
SmallVector<int32_t, 4> getShapeFromTensorType(TensorType type) {
  SmallVector<int32_t, 4> shape;
  for (auto dim : type.getShape())
    shape.push_back(static_cast<int32_t>(dim));
  return shape;
}

// Helper function to extract dimensions from memref or tensor type
SmallVector<int32_t, 4> getDimsFromType(Type type) {
  SmallVector<int32_t, 4> dims;
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    for (auto dim : memrefType.getShape())
      dims.push_back(static_cast<int32_t>(dim));
  } else if (auto tensorType = dyn_cast<TensorType>(type)) {
    for (auto dim : tensorType.getShape())
      dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

static uint64_t getElemByte(Type type) {
  static DataLayout dataLayout;
  auto typeSize = dataLayout.getTypeSize(type);
  if (!typeSize.isFixed()) {
    llvm::llvm_unreachable_internal("All element type should have fixed size.");
  }
  return typeSize.getFixedValue();
}

static Value createAddressFromMemref(ConversionPatternRewriter &rewriter,
                                     Location loc, Value memref) {
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value indexBasePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), stridedMetadata.getBaseBuffer());
  auto elemType = dyn_cast<MemRefType>(memref.getType()).getElementType();
  Value elemByte =
      rewriter.create<arith::ConstantIndexOp>(loc, getElemByte(elemType));
  Value offset = stridedMetadata.getOffset();
  Value byteOffset =
      rewriter.create<arith::MulIOp>(loc, offset.getType(), offset, elemByte);
  Value offsetPtr = rewriter.create<arith::AddIOp>(loc, indexBasePtr.getType(),
                                                   indexBasePtr, byteOffset);
  Value i64SPMPtr = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), offsetPtr);
  return i64SPMPtr;
}

static std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
createMetadata(ConversionPatternRewriter &rewriter, Location loc,
               Value operand) {
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
  Value indexBasePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), stridedMetadata.getBaseBuffer());
  auto elemType = dyn_cast<MemRefType>(operand.getType()).getElementType();
  Value elemByte =
      rewriter.create<arith::ConstantIndexOp>(loc, getElemByte(elemType));
  Value offset = stridedMetadata.getOffset();
  Value byteOffset =
      rewriter.create<arith::MulIOp>(loc, offset.getType(), offset, elemByte);
  Value offsetPtr = rewriter.create<arith::AddIOp>(loc, indexBasePtr.getType(),
                                                   indexBasePtr, byteOffset);
  Value i64SPMPtr = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), offsetPtr);

  // FIXME: For multi-dimensional(rank > 2), strides need to be multiplied.
  return {i64SPMPtr, stridedMetadata.getSizes(), stridedMetadata.getStrides()};
}

static SmallVector<Value, 4> padSizesToNHWC(ConversionPatternRewriter &rewriter,
                                            Location loc, ValueRange sizes) {
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  int numPad = 4 - sizes.size();
  SmallVector<Value, 4> nhwcShape;
  while (numPad--) {
    nhwcShape.push_back(one);
  }
  for (auto dim : sizes) {
    nhwcShape.push_back(dim);
  }
  return nhwcShape;
}

// The last stride is always 1, skip it, nhwcStrides.size() will be 3.
static SmallVector<Value, 4>
padStridesToNHWC(ConversionPatternRewriter &rewriter, Location loc,
                 ValueRange strides) {
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  int numPad = 4 - strides.size();
  SmallVector<Value, 4> nhwcStrides;
  while (numPad--) {
    nhwcStrides.push_back(one);
  }
  for (auto dim : strides) {
    nhwcStrides.push_back(dim);
  }
  return nhwcStrides;
}

static Value calculateElemCount(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange sizes) {
  // If we get scalar data, sizes is empty, return 1
  if (sizes.empty()) {
    return rewriter.create<arith::ConstantIndexOp>(loc, 1);
  }

  Value elemCount = sizes[0];
  for (int i = 1; i < sizes.size(); i++) {
    elemCount = rewriter.create<arith::MulIOp>(loc, elemCount.getType(),
                                               elemCount, sizes[i]);
  }
  return elemCount;
}

// Extract the operations from a linalg op region
template <typename T> llvm::SmallVector<Operation *> getRegionOps(T linalgOp) {
  auto regionBlock = linalgOp.getBody();
  return llvm::map_to_vector(regionBlock->without_terminator(),
                             [](Operation &op) { return &op; });
}

static Data_Format getFormatFromValueType(MemRefType valueType) {
  auto elemType = valueType.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 8:
    return Fmt_INT8;
  case 16:
    return Fmt_FP16;
  case 32:
    return Fmt_FP32;
  default:
    llvm_unreachable("Unsupported bit width\n");
  }
  return Fmt_FP32;
}

// Convert integer type to float type for CGRA instruction
// Return the convert float type format code
// TODO: Directly convert memref type?
Data_Format insertConvertTypeOp(Value valuePtr, MemRefType valueType,
                                Value elemCount,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {

  // TODO: Other integer type. May need realloc the memory
  auto elemType = valueType.getElementType();

  if (!isa<IntegerType>(elemType))
    return getFormatCode(valueType);

  Data_Format fmt = Fmt_FP32;
  // Get the bit width from the element type
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 16: { // 16 bit integer
    rewriter.create<tx::INT16ToFP16Op>(loc, rewriter.getI64Type(), valuePtr,
                                       valuePtr, elemCount);
    fmt = Fmt_FP16;
    break;
  }
  case 32: { // 32 bit integer
    rewriter.create<tx::INT32ToFP32Op>(loc, rewriter.getI64Type(), valuePtr,
                                       valuePtr, elemCount,
                                       rewriter.getI16IntegerAttr(0));
    break;
  }
  default: {
    llvm_unreachable("Unsupported integer type\n");
  }
  }
  return fmt;
}

// Restore float type to integer type to for CGRA instruction
Value insertRestoreTypeOp(Value valuePtr, MemRefType valueType, Value elemCount,
                          ConversionPatternRewriter &rewriter, Location loc) {
  // TODO: Other integer type. May need realloc the memory
  auto elemType = valueType.getElementType();
  auto newValue = valuePtr;
  if (!isa<IntegerType>(elemType))
    return newValue;

  // Get the bit width from the element type
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 16: { // 16 bit integer
    newValue = rewriter.create<tx::FP16ToINT16Op>(
        loc, rewriter.getI64Type(), valuePtr, valuePtr, elemCount,
        rewriter.getI16IntegerAttr(0));
    break;
  }
  case 32: { // 32 bit integer
    newValue = rewriter.create<tx::FP32ToINT32Op>(
        loc, rewriter.getI64Type(), valuePtr, valuePtr, elemCount,
        rewriter.getI16IntegerAttr(0));
    break;
  }
  default: {
    llvm_unreachable("Unsupported integer type\n");
  }
  }
  return newValue;
}

SmallVector<int64_t, 4> reshapeReduceShapeTo4d(ArrayRef<int64_t> inputShape,
                                               int64_t dim) {

  auto rank = inputShape.size();
  SmallVector<int64_t, 4> newShape;
  int64_t leftDimsElement = 1;
  int64_t rightDimsElement = 1;

  for (int i = 0; i < dim; i++)
    leftDimsElement *= inputShape[i];

  if (dim == inputShape.size() - 1)
    return {1, 1, leftDimsElement, inputShape[dim]};

  for (int i = dim + 1; i < rank; i++)
    rightDimsElement *= inputShape[i];

  newShape = {1, leftDimsElement, inputShape[dim], rightDimsElement}; // NHWC
  return newShape;
}

uint64_t next_power_of_two_64(uint64_t x) {
  if (x == 0) {
    return 1;
  }
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

class MemoryCopyConvertPattern : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  // Workaround: Avoid analyzing control flow as much as possible
  bool isOperandMemorySpaceSPM(Value operand) const {

    while (auto op = operand.getDefiningOp()) {
      if (isa<memref::AllocOp>(op))
        return true;
      operand = op->getOperand(0);
    }
    return false;
  }

  // View int64 as int8
  void reshapeInt64ToInt8(SmallVector<Value> &srcSizes,
                          SmallVector<Value> &srcStrides,
                          ConversionPatternRewriter &rewriter,
                          Operation *op) const {
    // Shape as int8, multiply bitsize to last dimension
    auto lastDim = srcSizes.back();
    srcSizes.back() = rewriter.create<arith::MulIOp>(
        op->getLoc(), lastDim.getType(), lastDim,
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 8));

    // Stride as int8, multiply bitsize to each stride
    std::transform(srcStrides.begin(), srcStrides.end(), srcStrides.begin(),
                   [&](Value size) {
                     return rewriter.create<arith::MulIOp>(
                         op->getLoc(), size.getType(), size,
                         rewriter.create<arith::ConstantIndexOp>(
                             op->getLoc(), 8)); // 8 bytes for int64_t
                   });
  }

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->hasAttr("srcSpm") && op->hasAttr("dstSpm") &&
           "Can't get memory space attribute\n");
    bool isSrcSPM = op->getAttrOfType<IntegerAttr>("srcSpm").getInt();
    bool isDstSPM = op->getAttrOfType<IntegerAttr>("dstSpm").getInt();

    // DDR to DDR
    if (!isSrcSPM && !isDstSPM)
      return rewriter.notifyMatchFailure(
          op, "Can not copy memory from DDR to DDR.\n");

    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getSource());
    auto [dstPtr, dstSizes, dstStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getTarget());

    auto inputType = dyn_cast<MemRefType>(op.getSource().getType());
    // For memory operations, 64 bit op can work as int8_t mode
    if (inputType.getElementTypeBitWidth() == 64) {
      // Need re-calculate sizes and strides, and convert type to int8_t
      reshapeInt64ToInt8(srcSizes, srcStrides, rewriter, op);
      reshapeInt64ToInt8(dstSizes, dstStrides, rewriter, op);

      // Convert type to int8_t
      SmallVector<int64_t, 4> shape(inputType.getShape().begin(),
                                    inputType.getShape().end());
      shape.back() *= sizeof(int64_t);

      inputType =
          MemRefType::get(shape, rewriter.getI8Type(), inputType.getLayout(),
                          inputType.getMemorySpace());
    }

    auto elemCount = calculateElemCount(rewriter, op->getLoc(), srcSizes);
    auto srcFmt = getFormatFromValueType(inputType);

    // SPM to SPM
    if (isSrcSPM && isDstSPM) {
      // WORKAROUND: Assume no mask.
      srcFmt = insertConvertTypeOp(srcPtr, inputType, elemCount, rewriter,
                                   op->getLoc());
      insertConvertTypeOp(dstPtr, inputType, elemCount, rewriter, op->getLoc());
      auto constValue = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), 0, rewriter.getI32Type());

      rewriter.create<tx::AddVSOp>(op->getLoc(), rewriter.getI64Type(), srcPtr,
                                   constValue, dstPtr,
                                   elemCount, // Element count
                                   rewriter.getI16IntegerAttr(0), // Round mode
                                   rewriter.getI16IntegerAttr(srcFmt) // Format
      );
      insertRestoreTypeOp(srcPtr, inputType, elemCount, rewriter, op->getLoc());
      insertRestoreTypeOp(dstPtr, inputType, elemCount, rewriter, op->getLoc());
      rewriter.eraseOp(op);
      return success();
    }

    auto srcShape4d = padSizesToNHWC(rewriter, op->getLoc(), srcSizes);
    auto srcStrides4d = padStridesToNHWC(rewriter, op->getLoc(), srcStrides);
    auto dstShape4d = padSizesToNHWC(rewriter, op->getLoc(), dstSizes);
    auto dstStrides4d = padStridesToNHWC(rewriter, op->getLoc(), dstStrides);

    int bitWidth = inputType.getElementType().getIntOrFloatBitWidth();
    int elemBytes = bitWidth / 8;

    if (isDstSPM) {
      auto rdmaOp = rewriter.create<tx::RdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          srcShape4d,                            // src shape
          srcStrides4d,                          // src stride
          dstShape4d,                            // dst shape
          dstStrides4d,                          // dst stride
          rewriter.getI32IntegerAttr(elemBytes), // elem bytes
          rewriter.getI32IntegerAttr(srcFmt)     // Format
      );
    } else {
      auto wdmaOp = rewriter.create<tx::WdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          srcShape4d,                            // src shape
          srcStrides4d,                          // src stride
          dstShape4d,                            // dst shape
          dstStrides4d,                          // dst stride
          rewriter.getI32IntegerAttr(elemBytes), // elem bytes
          rewriter.getI32IntegerAttr(srcFmt)     // Format
      );
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert linalg.fill to MemsetOp
class LinalgFillOpConversion : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the value to fill with
    Value fillValue = op.getInputs()[0]; // adaptor.getValue();

    if (op.getOutputs().size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support single output\n");

    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto inputType = op.getInputs()[0].getType();
    auto bitWidth = op.getInputs()[0].getType().getIntOrFloatBitWidth();

    if (bitWidth != 16 && bitWidth != 32) {
      if (failed(linalg::linalgOpToLoops(rewriter, op)))
        return rewriter.notifyMatchFailure(op, " op not yet supported");
      rewriter.eraseOp(op);
      return success();
    }

    // AddVS value need has fmt with input fmt and only support float type
    Data_Format fmt = bitWidth == 16 ? Fmt_FP16 : Fmt_FP32;

    auto bitcastType =
        bitWidth == 16 ? rewriter.getI16Type() : rewriter.getI32Type();
    fillValue =
        rewriter.create<arith::BitcastOp>(op.getLoc(), bitcastType, fillValue);

    if (bitWidth == 16) {
      fillValue = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI32Type(), fillValue);
    }

    // TODO: For scalar data, instead of function call, we should convert
    // linalg.fill to memref.store directly to get better performance.

    // Use xor + addvs to simulate memset operation. Only support type fp32 and
    // fp16
    // 1. xor srcPtr with itself to get zero
    // 2. addvs srcPtr with value to get the fill value
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), srcSizes);

    auto init =
        rewriter.create<tx::XorVV>(op.getLoc(), rewriter.getI64Type(), srcPtr,
                                   srcPtr, srcPtr, elemCount, fmt);
    auto resultOp = rewriter.create<tx::AddVSOp>(
        op.getLoc(), rewriter.getI64Type(), srcPtr, fillValue, srcPtr,
        elemCount,
        rewriter.getI16IntegerAttr(0), // round_mode
        rewriter.getI16IntegerAttr(fmt));

    rewriter.eraseOp(op);

    return success();
  }
};

class TransposeOpConversion : public OpConversionPattern<linalg::TransposeOp> {
public:
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult
  convertToGatherScatter(linalg::TransposeOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
    auto perm = op.getPermutation();
    auto rank = perm.size();

    auto src = op.getInput();
    auto dst = op.getInit();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    // Get NHWC shape
    SmallVector<int64_t, 4> srcShape(srcType.getShape());
    SmallVector<int64_t, 4> dstShape(dstType.getShape());
    SmallVector<int64_t, 4> perm4d(perm.begin(), perm.end());
    while (srcShape.size() < 4) {
      srcShape.push_back(1);
      dstShape.push_back(1);
      perm4d.push_back(perm4d.size());
    }

    // Get inner bytes
    int32_t elemCount = srcShape[3];
    auto elemType = srcType.getElementType();
    auto bitWidth = elemType.getIntOrFloatBitWidth();
    auto byte = bitWidth / 8;
    auto bytes = elemCount * byte;

    // Get strides
    SmallVector<int64_t, 4> srcStride(srcShape.size());
    srcStride[srcShape.size() - 1] = byte;
    for (int i = srcShape.size() - 2; i >= 0; --i) {
      srcStride[i] = srcStride[i + 1] * srcShape[i + 1];
    }
    SmallVector<int64_t, 4> dstStride(dstShape.size());
    for (int i = 0; i < dstShape.size(); i++) {
      dstStride[i] = srcStride[perm4d[i]];
    }

    auto srcPtr = createAddressFromMemref(rewriter, op->getLoc(), src);
    auto dstPtr = createAddressFromMemref(rewriter, op->getLoc(), dst);

    auto newOp = rewriter.create<tx::GatherScatter>(
        op->getLoc(), rewriter.getI64Type(), srcPtr, dstPtr, bytes,
        srcStride[0], srcStride[1], srcStride[2], srcShape[0], srcShape[1],
        srcShape[2], dstStride[0], dstStride[1], dstStride[2], dstShape[0],
        dstShape[1], dstShape[2]);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult convertToTranspose(linalg::TransposeOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    auto src = op.getInput();
    auto dst = op.getInit();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    int32_t dim0 = srcType.getShape()[0];
    int32_t dim1 = srcType.getShape()[1];
    SmallVector<int32_t, 4> srcShape({1, dim0, dim1, 1});
    SmallVector<int32_t, 4> dstShape({1, dim1, dim0, 1});

    auto srcPtr = createAddressFromMemref(rewriter, op->getLoc(), src);
    auto dstPtr = createAddressFromMemref(rewriter, op->getLoc(), dst);
    Data_Format fmt = getFormatCode(srcType);

    auto newOp =
        rewriter.create<tx::Transpose>(op->getLoc(), rewriter.getI64Type(),
                                       srcPtr, dstPtr, srcShape, dstShape, fmt);

    rewriter.eraseOp(op);
    return success();
  }

  template <typename transTy>
  LogicalResult transposeChannel(linalg::TransposeOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    auto src = op.getInput();
    auto dst = op.getInit();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    SmallVector<int32_t, 4> srcShape(srcType.getShape().begin(),
                                     srcType.getShape().end());
    SmallVector<int32_t, 4> dstShape(dstType.getShape().begin(),
                                     dstType.getShape().end());

    auto srcPtr = createAddressFromMemref(rewriter, op->getLoc(), src);
    auto dstPtr = createAddressFromMemref(rewriter, op->getLoc(), dst);
    Data_Format fmt = getFormatCode(srcType);

    auto newOp =
        rewriter.create<transTy>(op->getLoc(), rewriter.getI64Type(), srcPtr,
                                 dstPtr, srcShape, dstShape, fmt);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto perm = op.getPermutation();
    auto rank = perm.size();

    // FIXME: tx.transpose/tx.gather_scatter need to cx align,
    // convert 2d transpose to loops default now.
    // if (rank == 2)
    //   return convertToGatherScatter(op, adaptor, rewriter);

    if (rank == 3)
      return convertToGatherScatter(op, adaptor, rewriter);

    if (rank == 4 && perm[3] == 3) {
      return convertToGatherScatter(op, adaptor, rewriter);
    }

    if (rank == 4 && perm == ArrayRef<int64_t>({0, 2, 3, 1})) {
      return transposeChannel<tx::Nchw2nhwc>(op, adaptor, rewriter);
    }

    if (rank == 4 && perm == ArrayRef<int64_t>({0, 3, 1, 2})) {
      return transposeChannel<tx::Nhwc2nchw>(op, adaptor, rewriter);
    }

    // Default handling of remaining cases.
    // TODO:  Convert higher rank to tx.
    if (failed(linalg::linalgOpToLoops(rewriter, op)))
      return rewriter.notifyMatchFailure(op, " op not yet supported");
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mk.dot to tx.gemm Conversion Pattern
//===----------------------------------------------------------------------===//

class MKDotToTx81GemmOpConversion
    : public OpConversionPattern<linalg::MatmulOp> {

  void fp32ToTF32(ConversionPatternRewriter &rewriter, Location loc,
                  ValueRange sizes, Value spmAddr) const {
    // Warning for neural engine that fp32 is not supported
    llvm::errs()
        << "\nNeural engine not support FP32. Convert FP32 to TF32 for "
           "tx.Gemm Op\n";
    auto elemCount = calculateElemCount(rewriter, loc, sizes);
    rewriter.create<tx::FP32ToTF32Op>(
        loc, rewriter.getI64Type(), spmAddr, spmAddr,
        elemCount,                    // element_count
        rewriter.getI16IntegerAttr(0) // round_mode
    );
  }

  Value createChannelNorm(Location loc, Value op,
                          ConversionPatternRewriter &rewriter) const {
    auto memType = cast<MemRefType>(op.getType());
    auto shape = memType.getShape();
    int bitWidth = memType.getElementType().getIntOrFloatBitWidth();

    int alignBase = bitWidth == 8 ? 128 : 64;

    int c = shape.back();
    // Has been cx aligned
    bool noNeedChannelNorm =
        (c >= 4 && c <= alignBase && c == next_power_of_two_64(c));
    if (noNeedChannelNorm) {
      return op;
    }

    int cx = c / alignBase;
    int c0 = c % alignBase;
    int alignedC0 = c0 ? next_power_of_two_64(c0) : 0;
    if (c0 < 4 && c0 > 0) {
      // If c0 is not zero, we need to align it to 4
      alignedC0 = 4;
    }
    int alignedC = cx * alignBase + alignedC0;
    SmallVector<int64_t, 4> alignedShape(shape.begin(), shape.end());
    alignedShape.back() = alignedC;

    auto alignedMemType =
        MemRefType::get(alignedShape, memType.getElementType());
    auto alignedAlloc = rewriter.create<memref::AllocOp>(loc, alignedMemType);

    auto srcPtr = createAddressFromMemref(rewriter, loc, op);
    auto dstPtr = createAddressFromMemref(rewriter, loc, alignedAlloc);

    SmallVector<int64_t, 4> shape4D =
        reshapeReduceShapeTo4d(shape, shape.size() - 1);
    auto channelNorm = rewriter.create<tx::ChannelNormOp>(
        loc, TypeRange({}), srcPtr, dstPtr,
        rewriter.getDenseI64ArrayAttr(shape4D),
        rewriter.getI16IntegerAttr(alignedC0),
        rewriter.getI16IntegerAttr(bitWidth));
    return alignedAlloc;
  }

public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract dimensions from tensor types
    MemRefType aTensorType = cast<MemRefType>(op->getOperand(0).getType());
    MemRefType bTensorType = cast<MemRefType>(op->getOperand(1).getType());
    assert(aTensorType.getElementType() == bTensorType.getElementType() &&
           "a and b must have the same element type");
    MemRefType dstType = cast<MemRefType>(op.getOutputs()[0].getType());
    Data_Format srcFmt = getFormatCode(aTensorType);
    Data_Format dstFmt = getFormatCode(dstType);

    // Get converted operands
    auto loc = op->getLoc();

    auto a = adaptor.getInputs()[0];
    auto b = adaptor.getInputs()[1];

    auto aShape = aTensorType.getShape();
    auto bShape = bTensorType.getShape();

    // Matrix dimensions M, K, N for GEMM
    int32_t M = aShape[0];
    int32_t K = aShape[1];
    // Notice: operand is (N, K) now.
    int32_t N = bShape[0];

    // Create dimensions array attribute [M, K, N]
    auto dims = rewriter.getI32ArrayAttr({M, K, N});

    auto alignedA = createChannelNorm(loc, a, rewriter);
    auto alignedB = createChannelNorm(loc, b, rewriter);

    // Get operand ptr
    auto [aPtr, aSizes, aStrides] = createMetadata(rewriter, loc, alignedA);
    auto [bPtr, bSizes, bStrides] = createMetadata(rewriter, loc, alignedB);

    auto [dstPtr, dstSizes, dstStrides] =
        createMetadata(rewriter, loc, adaptor.getOutputs()[0]);

    // Assume input type is same. Tx neural engine not support fp32 for input
    if (aTensorType.getElementType().isF32()) {
      srcFmt = Data_Format::Fmt_TF32;
      fp32ToTF32(rewriter, loc, aSizes, aPtr);
      fp32ToTF32(rewriter, loc, bSizes, bPtr);
    }

    auto zero =
        rewriter.create<arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());

    // Check if N is a power of 2 and greater than 4
    if ((N & (N - 1)) != 0) { // Check if N is not a power of 2
      return rewriter.notifyMatchFailure(op, "N must be a power of 2");
    }
    if (N < 4) {
      return rewriter.notifyMatchFailure(op, "N must be greater than 4");
    }

    // Create GemmOp
    // TODO: Support bias when input is int8
    rewriter.create<tx::GemmOp>(
        loc, rewriter.getI64Type(),
        aPtr,                        // src_a (Matrix A in SPM)
        bPtr,                        // src_b (Matrix B in SPM)
        dstPtr,                      // src_bias. Unused for now.
        dstPtr,                      // dst,
        dims,                        // dimensions [M,K,N]
        rewriter.getBoolAttr(false), // en_psum. Used as accumulate buffer
        dstPtr, //  The address of psum in SPM, Always same to output
        rewriter.getBoolAttr(false), // trans_src_a
        // NOTE: (N, K) is thought not trans in hardware
        rewriter.getBoolAttr(false),                   // trans_src_b.
        rewriter.getI32IntegerAttr(1),                 // batch_src_a
        rewriter.getI32IntegerAttr(1),                 // batch_src_b
        rewriter.getI32IntegerAttr(ActFuncMode::None), // relu_mode.
        rewriter.getBoolAttr(false),                   // en_bias
        rewriter.getBoolAttr(false),                   // en_neg_scale
        zero,                                          // src_neg_scale
        rewriter.getBoolAttr(false),                   // en_pos_scale
        zero,                                          // src_pos_scale
        rewriter.getI32IntegerAttr(srcFmt),            // src_fmt
        rewriter.getI32IntegerAttr(dstFmt)             // dst_fmt
    );

    // DechannelNorm dst
    int bitWidth = dstType.getElementType().getIntOrFloatBitWidth();
    int alignBase = bitWidth == 32 ? 64 : 128;
    if (N > alignBase) {
      auto dechannelNorm = rewriter.create<tx::DechannelNormOp>(
          loc, TypeRange({}), dstPtr, dstPtr,
          rewriter.getDenseI64ArrayAttr({1, 1, M, N}),
          rewriter.getI16IntegerAttr(0) /*alignedC0*/,
          rewriter.getI16IntegerAttr(bitWidth));
    }

    // Op has no result value
    rewriter.eraseOp(op);

    return success();
  }
};

class GatherConvertPattern : public OpConversionPattern<mlir::mk::GatherOp> {
public:
  using OpConversionPattern<mlir::mk::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto indices = adaptor.getIndices();
    auto indicesType = cast<MemRefType>(indices.getType());
    auto shape = indicesType.getShape();

    auto axis = op.getAxis();

    int64_t numElems = indicesType.getNumElements();
    auto strides = computeStrides(shape);
    for (int64_t idx = 0; idx < numElems; idx += 1) {
      auto tensorIdx = delinearize(idx, strides);

      SmallVector<Value> idxIndex(tensorIdx.size());
      std::transform(tensorIdx.begin(), tensorIdx.end(), idxIndex.begin(),
                     [&](auto val) {
                       return rewriter.create<arith::ConstantIndexOp>(loc, val);
                     });
      // Read the index value from indices tensor
      Value indexValue =
          rewriter.create<memref::LoadOp>(loc, indices, idxIndex);

      // Read value from source using computed indices
      SmallVector<Value> inputIndex = idxIndex;
      assert(axis < inputIndex.size() && axis >= 0 &&
             "Axis index out of bounds");
      inputIndex[axis] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), indexValue);

      Value gatheredValue =
          rewriter.create<memref::LoadOp>(loc, adaptor.getSrc(), inputIndex);

      // Write value to destination
      rewriter.create<memref::StoreOp>(loc, gatheredValue, adaptor.getDst(),
                                       idxIndex);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

class MKSigmoidToTx81SigmoidOpConversion
    : public OpConversionPattern<mlir::mk::SigmoidOp> {
public:
  using OpConversionPattern<mlir::mk::SigmoidOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [input, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getSrc());
    auto [dst, dstSizes, dstStrides] =
        createMetadata(rewriter, loc, adaptor.getZeroes());
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    // Tx neural engine not support fp32 for input
    auto inputType = dyn_cast<MemRefType>(op.getSrc().getType());
    Data_Format srcFmt = getFormatCode(inputType);

    rewriter.create<tx::Sigmoid>(loc, rewriter.getI64Type(), input, dst,
                                 elemCount, rewriter.getI16IntegerAttr(srcFmt));
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename TxOpT>
class ArgMinMaxBaseConversion : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;

public:
  bool isArgMin;

  ArgMinMaxBaseConversion(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIter = ops.begin();
    Value indexSelectOp, valueSelectOp;
    if (failed(matchArgMinMax(currValue, currIndex, reduceValue, reduceIndex,
                              opsIter, indexSelectOp, valueSelectOp,
                              isArgMin))) {
      return failure();
    }

    // matching: linalg.yield %18, %19 : f32, i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIter << "\n");
    auto termOp = dyn_cast<linalg::YieldOp>(*opsIter++);
    if (termOp && termOp == block->getTerminator()) {
      auto opnds = termOp.getOperands();
      if (opnds != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
        return failure();
      }
    } else {
      return failure();
    }

    // Rewrite with tx81 operation
    auto loc = op.getLoc();
    auto dims = op.getDimensions();
    if (dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support one dim reduce.");
    auto dim = dims[0];

    auto input =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getInputs()[0]);
    auto value =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getInits()[0]);
    auto index =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getInits()[1]);
    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto valueType = dyn_cast<MemRefType>(adaptor.getInits()[0].getType());

    // TODO: Support any rank
    auto inputShape = inputType.getShape();
    if (inputShape.size() > 1)
      return rewriter.notifyMatchFailure(op, "Rank > 1 unsupported yet.");

    int64_t inputSize = inputShape.empty() ? 1 : inputShape[0];
    auto tx81Op = rewriter.create<TxOpT>(
        op->getLoc(), TypeRange{}, input,
        // TODO: get output value and index
        value, index, rewriter.getI32IntegerAttr(inputSize),
        rewriter.getI16IntegerAttr(getFormatCode(valueType)));

    rewriter.replaceOp(op, tx81Op);

    return success();
  }
};

struct ArgMinConversion : public ArgMinMaxBaseConversion<tx::ArgMinOp> {
  ArgMinConversion(MLIRContext *context) : ArgMinMaxBaseConversion(context) {
    isArgMin = true;
  }
};

struct ArgMaxConversion : public ArgMinMaxBaseConversion<tx::ArgMaxOp> {
  ArgMaxConversion(MLIRContext *context) : ArgMinMaxBaseConversion(context) {
    isArgMin = false;
  }
};

struct ElementwiseConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  template <typename TxOpT>
  LogicalResult convertUnaryOp(linalg::GenericOp op, OpAdaptor adapter,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adapter.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adapter.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    // Data format after conversion
    Data_Format srcFmt =
        insertConvertTypeOp(input, inputType, elemCount, rewriter, loc);
    Data_Format dstFmt =
        insertConvertTypeOp(output, outputType, elemCount, rewriter, loc);
    // Create the unary operation
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input, output, elemCount,
                           rewriter.getI16IntegerAttr(srcFmt));
    insertRestoreTypeOp(input, inputType, elemCount, rewriter, loc);
    insertRestoreTypeOp(output, outputType, elemCount, rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult convertBinaryOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    // Data format after conversion
    Data_Format srcFmt =
        insertConvertTypeOp(input0, inputType, elemCount, rewriter, loc);
    if (adaptor.getInputs()[0] != adaptor.getInputs()[1]) {
      // If input0 and input1 are not the same, we need to convert input1 type
      insertConvertTypeOp(input1, inputType, elemCount, rewriter, loc);
    }

    if (adaptor.getInputs()[0] != adaptor.getOutputs()[0] &&
        adaptor.getInputs()[1] != adaptor.getOutputs()[0]) {
      // If input and output are not the same, we need to convert output type
      Data_Format dstFmt =
          insertConvertTypeOp(output, inputType, elemCount, rewriter, loc);
    }

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCount,
                           rewriter.getI16IntegerAttr(0), // Round mode
                           rewriter.getI16IntegerAttr(srcFmt));

    insertRestoreTypeOp(input0, inputType, elemCount, rewriter, loc);
    if (adaptor.getInputs()[0] != adaptor.getInputs()[1]) {
      insertRestoreTypeOp(input1, inputType, elemCount, rewriter, loc);
    }
    if (adaptor.getInputs()[0] != adaptor.getOutputs()[0] &&
        adaptor.getInputs()[1] != adaptor.getOutputs()[0]) {
      insertRestoreTypeOp(output, inputType, elemCount, rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult NormalConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input, output,
                           elemCount);
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult RoundConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);
    // TODO: Fix attribute
    auto result =
        rewriter.create<TxOpT>(loc,
                               rewriter.getI64Type(),        // Result type
                               input,                        // Input
                               output,                       // Output
                               elemCount,                    // Element count
                               rewriter.getI16IntegerAttr(0) // Round mode
        );
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult BoolRelationVVOp(linalg::GenericOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    Data_Format srcFmt =
        insertConvertTypeOp(input0, inputType, elemCount, rewriter, loc);
    insertConvertTypeOp(input1, inputType, elemCount, rewriter, loc);

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCount,
                           rewriter.getI16IntegerAttr(srcFmt) // Format
    );

    insertRestoreTypeOp(input0, inputType, elemCount, rewriter, loc);
    insertRestoreTypeOp(input1, inputType, elemCount, rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult FmaConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto input2 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[2]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    auto mulResult = rewriter.create<tx::MulVVOp>(
        loc, rewriter.getI64Type(), input0, input1, output, elemCount,
        rewriter.getI16IntegerAttr(0), // Round mode
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));
    auto addResult = rewriter.create<tx::AddVVOp>(
        loc, rewriter.getI64Type(), output, input2, output, elemCount,
        rewriter.getI16IntegerAttr(0), // Round mode
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult SelectConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto input2 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[2]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    assert(inputType.getNumElements() % 8 == 0 &&
           "Total elements need be multiple of 8\n");

    Data_Format srcFmt =
        insertConvertTypeOp(input1, inputType, elemCount, rewriter, loc);
    insertConvertTypeOp(input2, inputType, elemCount, rewriter, loc);

    // Add zero const value
    auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                      rewriter.getI32Type());

    assert(adaptor.getInputs()[2] != adaptor.getOutputs()[0]);
    // Maskmove mask only support int8/fp, here mask is memref<i1>
    auto maskCast = rewriter.create<memref::AllocOp>(loc, inputType);
    auto maskCastAddr = createAddressFromMemref(rewriter, loc, maskCast);
    rewriter.create<tx::Bit2FpOp>(op.getLoc(), rewriter.getI64Type(), input0,
                                  maskCastAddr, elemCount,
                                  rewriter.getI16IntegerAttr(srcFmt));

    // Input1 and output are same address
    if (adaptor.getInputs()[1] == adaptor.getOutputs()[0]) {
      // Create memref::allocOp
      auto temp = rewriter.create<memref::AllocOp>(loc, inputType);
      auto tempAddr = createAddressFromMemref(rewriter, loc, temp);
      auto mid = rewriter.create<tx::AddVSOp>(
          op.getLoc(), rewriter.getI64Type(), input2, zero, tempAddr, elemCount,
          rewriter.getI16IntegerAttr(0), // round_mode
          rewriter.getI16IntegerAttr(srcFmt));

      rewriter.create<tx::MaskMoveOp>(loc, rewriter.getI64Type(), input1,
                                      tempAddr, elemCount, maskCastAddr,
                                      rewriter.getI32IntegerAttr(srcFmt));
      // Res = input2 + 0;
      rewriter.create<tx::AddVSOp>(op.getLoc(), rewriter.getI64Type(), tempAddr,
                                   zero, output, elemCount,
                                   rewriter.getI16IntegerAttr(0), // round_mode
                                   rewriter.getI16IntegerAttr(srcFmt));

    } else {
      insertConvertTypeOp(output, inputType, elemCount, rewriter, loc);

      // Res = input2 + 0;
      auto mid = rewriter.create<tx::AddVSOp>(
          op.getLoc(), rewriter.getI64Type(), input2, zero, output, elemCount,
          rewriter.getI16IntegerAttr(0), // round_mode
          rewriter.getI16IntegerAttr(srcFmt));

      // if input0 = 1, Res = input1;
      // if input0 = 0, Res = input2;
      rewriter.create<tx::MaskMoveOp>(loc, rewriter.getI64Type(), input1,
                                      output, elemCount, maskCastAddr,
                                      rewriter.getI32IntegerAttr(srcFmt));
      insertRestoreTypeOp(output, inputType, elemCount, rewriter, loc);
    }
    insertRestoreTypeOp(input1, inputType, elemCount, rewriter, loc);
    insertRestoreTypeOp(input2, inputType, elemCount, rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult convertMinMaxOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    auto lhs = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto rhs = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);

    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);

    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    auto fmt = getFormatCode(inputType);

    auto isANanBuffer = rewriter.create<memref::AllocOp>(loc, inputType);

    auto isANanBufferAddr =
        createAddressFromMemref(rewriter, loc, isANanBuffer);

    // auto isANan = UnEqualVV(lhs，lhs)
    // auto result = lhs
    // result = maskmove(isANan, rhs)
    // auto isBNan = UnEqualVV(rhs, rhs)
    // auto shouldApplyMinMax = EqualVS(isBNan, 0)
    // auto minMaxValue = maxvv/minvv(result, rhs)
    // result = maskmove(shouldApplyMinMax, minMaxValue)

    auto isANan =
        rewriter.create<tx::UnEqualVV>(loc,                   // loc
                                       rewriter.getI64Type(), // result type
                                       lhs,                   // input0
                                       lhs,                   // input1
                                       isANanBufferAddr,      // out
                                       elemCount,             // elem_count
                                       rewriter.getI16IntegerAttr(fmt) // fmt
        );

    auto constValue = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), 0, rewriter.getI32Type());
    rewriter.create<tx::AddVSOp>(op.getLoc(), rewriter.getI64Type(), lhs,
                                 constValue, output, elemCount,
                                 rewriter.getI16IntegerAttr(0), // round_mode
                                 rewriter.getI16IntegerAttr(fmt));

    rewriter.create<tx::MaskMoveOp>(loc, rewriter.getI64Type(), rhs, output,
                                    elemCount, isANanBufferAddr,
                                    rewriter.getI32IntegerAttr(fmt));

    auto isBNanBuffer = rewriter.create<memref::AllocOp>(loc, inputType);
    auto isBNanBufferAddr =
        createAddressFromMemref(rewriter, loc, isBNanBuffer);
    auto isBNan =
        rewriter.create<tx::UnEqualVV>(loc,                   // loc
                                       rewriter.getI64Type(), // result type
                                       rhs,                   // input0
                                       rhs,                   // input1
                                       isBNanBufferAddr,      // out
                                       elemCount,             // elem_count
                                       rewriter.getI16IntegerAttr(fmt) // fmt
        );

    auto shouldApplyMinMaxBuffer =
        rewriter.create<memref::AllocOp>(loc, inputType);
    auto shouldApplyMinMaxBufferAddr =
        createAddressFromMemref(rewriter, loc, shouldApplyMinMaxBuffer);
    auto shouldApplyMinMax = rewriter.create<tx::EqualVS>(
        op.getLoc(), rewriter.getI64Type(), isBNanBufferAddr, constValue,
        shouldApplyMinMaxBufferAddr, elemCount,
        rewriter.getI16IntegerAttr(fmt));

    auto minMaxValueBuffer = rewriter.create<memref::AllocOp>(loc, inputType);
    auto minMaxValueBufferAddr =
        createAddressFromMemref(rewriter, loc, isBNanBuffer);
    auto minMaxValue =
        rewriter.create<TxOpT>(loc, rewriter.getI64Type(), output, rhs,
                               minMaxValueBufferAddr, elemCount,
                               rewriter.getI16IntegerAttr(0), // Round mode
                               rewriter.getI16IntegerAttr(fmt));

    auto result = rewriter.create<tx::MaskMoveOp>(
        loc, rewriter.getI64Type(), minMaxValueBufferAddr, output, elemCount,
        shouldApplyMinMaxBufferAddr, rewriter.getI32IntegerAttr(fmt));

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  convertCeilAndFloorOp(linalg::GenericOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);
    auto *body = op.getBody();
    auto &operation = body->front();
    auto roundMode = llvm::dyn_cast<mlir::math::CeilOp>(&operation)
                         ? RND_MODE::RND_POS_INF
                         : RND_MODE::RND_NEG_INF;
    // TODO: Fix attribute
    // Use IEEE round to positive infinity mode
    auto fpToInt = rewriter.create<tx::FP32ToINT32Op>(
        loc,
        rewriter.getI64Type(),                // Result type
        input,                                // Input
        output,                               // Output
        elemCount,                            // Element count
        rewriter.getI16IntegerAttr(roundMode) // Round mode
    );
    auto intToFp = rewriter.create<tx::INT32ToFP32Op>(
        loc,
        rewriter.getI64Type(),        // Result type
        output,                       // Input
        output,                       // Output
        elemCount,                    // Element count
        rewriter.getI16IntegerAttr(0) // Round mode
    );

    rewriter.eraseOp(op);
    return success();
  }
  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto regionOps = getRegionOps<linalg::GenericOp>(op);

    // Check if the operation is elementwise
    if (op.getIteratorTypesArray().front() != utils::IteratorType::parallel)
      return rewriter.notifyMatchFailure(op, "Only support elementwise op.");

    // WORKAROUND: Select op input0 is bool(i1), cmp op result is bool(i1)
    // I64/F64 lowering to llvm
    if (regionOps.size() != 1 ||
        (dyn_cast<MemRefType>(op.getOutputs()[0].getType())
             .getElementType()
             .getIntOrFloatBitWidth() == 64) ||
        (dyn_cast<MemRefType>(op.getInputs()[0].getType())
             .getElementType()
             .getIntOrFloatBitWidth() == 64)) {
      if (failed(linalg::linalgOpToLoops(rewriter, op)))
        return rewriter.notifyMatchFailure(op,
                                           "Element-wise op not yet supported");
      rewriter.eraseOp(op);
      return success();
    }

    auto elemWiseOp = regionOps[0];
    auto resultType = elemWiseOp->getResult(0).getType();
    return llvm::TypeSwitch<Operation *, LogicalResult>(elemWiseOp)
        .Case<arith::AddIOp, arith::AddFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::AddVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::SubIOp, arith::SubFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::SubVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::MulIOp, arith::MulFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MulVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(
            [&](auto elemWiseOp) {
              return convertBinaryOp<tx::DivVVOp>(op, adaptor, rewriter);
            })
        .Case<arith::MaxSIOp, arith::MaxUIOp, arith::MaximumFOp>(
            [&](auto elemWiseOp) {
              return convertBinaryOp<tx::MaxVVOp>(op, adaptor, rewriter);
            })
        .Case<arith::MinSIOp, arith::MinUIOp, arith::MinimumFOp>(
            [&](auto elemWiseOp) {
              return convertBinaryOp<tx::MinVVOp>(op, adaptor, rewriter);
            })
        .Case<arith::MaxNumFOp>([&](auto elemWiseOp) {
          return convertMinMaxOp<tx::MaxVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::MinNumFOp>([&](auto elemWiseOp) {
          return convertMinMaxOp<tx::MinVVOp>(op, adaptor, rewriter);
        })
        .Case<math::AbsFOp, math::AbsIOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::AbsVVOp>(op, adaptor, rewriter);
        })
        .Case<math::CeilOp, math::FloorOp>([&](auto elemWiseOp) {
          return convertCeilAndFloorOp(op, adaptor, rewriter);
        })
        .Case<math::SqrtOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::SqrtVVOp>(op, adaptor, rewriter);
        })
        .Case<math::RsqrtOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::RsqrtVVOp>(op, adaptor, rewriter);
        })
        .Case<math::LogOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::LnOp>(op, adaptor, rewriter);
        })
        .Case<math::Log2Op>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::Log2Op>(op, adaptor, rewriter);
        })
        .Case<math::ExpOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::ExpOp>(op, adaptor, rewriter);
        })
        .Case<math::Exp2Op>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::Pow2Op>(op, adaptor, rewriter);
        })
        .Case<math::SinOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::SinOp>(op, adaptor, rewriter);
        })
        .Case<math::CosOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::CosOp>(op, adaptor, rewriter);
        })
        .Case<arith::ExtFOp>([&](auto elemWiseOp) {
          return NormalConvertOp<tx::FP16ToFP32Op>(op, adaptor, rewriter);
        })
        .Case<math::FmaOp>([&](auto elemWiseOp) {
          return FmaConvertOp(op, adaptor, rewriter);
        })
        .Case<arith::SelectOp>([&](auto elemWiseOp) {
          return SelectConvertOp(op, adaptor, rewriter);
        })
        .Case<arith::SIToFPOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType())
                                .getElementType();
          if (inputType.isInteger(16) && outputType.isF32()) {
            return RoundConvertOp<tx::INT16ToFP32Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(16) && outputType.isF16()) {
            return NormalConvertOp<tx::INT16ToFP16Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(32) && outputType.isF16()) {
            return RoundConvertOp<tx::INT32ToFP16Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(32) && outputType.isF32()) {
            return RoundConvertOp<tx::INT32ToFP32Op>(op, adaptor, rewriter);
          } else {
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for integer to "
                    "FP conversion");
          }
        })
        .Case<arith::FPToSIOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType())
                                .getElementType();
          if (inputType.isF16() && outputType.isInteger(8)) {
            return RoundConvertOp<tx::FP16ToINT8Op>(op, adaptor, rewriter);
          } else if (inputType.isF16() && outputType.isInteger(16)) {
            return RoundConvertOp<tx::FP16ToINT16Op>(op, adaptor, rewriter);
          } else if (inputType.isF16() && outputType.isInteger(32)) {
            return RoundConvertOp<tx::FP16ToINT32Op>(op, adaptor, rewriter);
          } else if (inputType.isF32() && outputType.isInteger(8)) {
            return RoundConvertOp<tx::FP32ToINT8Op>(op, adaptor, rewriter);
          } else if (inputType.isF32() && outputType.isInteger(16)) {
            return RoundConvertOp<tx::FP32ToINT16Op>(op, adaptor, rewriter);
          } else if (inputType.isF32() && outputType.isInteger(32)) {
            return RoundConvertOp<tx::FP32ToINT32Op>(op, adaptor, rewriter);
          } else {
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for fp to "
                    "integer conversion");
          }
        })
        .Case<arith::CmpIOp>([&](auto elemWiseOp) {
          // WORKAROUND: Tx8 bool relation op need elems to be multiple of 8.
          if (dyn_cast<MemRefType>(op.getOperandTypes()[0]).getNumElements() %
                  8 !=
              0) {
            if (failed(linalg::linalgOpToLoops(rewriter, op)))
              return rewriter.notifyMatchFailure(
                  op, "Element-wise op not yet supported");
            rewriter.eraseOp(op);
            return success();
          }
          arith::CmpIPredicate predicate = elemWiseOp.getPredicate();
          switch (predicate) {
          case arith::CmpIPredicate::eq:
            return BoolRelationVVOp<tx::BoolEqualVV>(op, adaptor, rewriter);
          case arith::CmpIPredicate::ne:
            return BoolRelationVVOp<tx::BoolUnEqualVV>(op, adaptor, rewriter);
          case arith::CmpIPredicate::sge:
            return BoolRelationVVOp<tx::BoolGreaterEqualVV>(op, adaptor,
                                                            rewriter);
          case arith::CmpIPredicate::sgt:
            return BoolRelationVVOp<tx::BoolGreaterVV>(op, adaptor, rewriter);
          case arith::CmpIPredicate::sle:
            return BoolRelationVVOp<tx::BoolLessEqualVV>(op, adaptor, rewriter);
          case arith::CmpIPredicate::slt:
            return BoolRelationVVOp<tx::BoolLessThenVV>(op, adaptor, rewriter);
          default:
            llvm_unreachable("Not yet supported");
            break;
          }
        })
        .Case<arith::CmpFOp>([&](auto elemWiseOp) {
          // WORKAROUND: Tx8 bool relation op need elems to be multiple of 8.
          if (dyn_cast<MemRefType>(op.getOperandTypes()[0]).getNumElements() %
                  8 !=
              0) {
            if (failed(linalg::linalgOpToLoops(rewriter, op)))
              return rewriter.notifyMatchFailure(
                  op, "Element-wise op not yet supported");
            rewriter.eraseOp(op);
            return success();
          }
          arith::CmpFPredicate predicate = elemWiseOp.getPredicate();
          switch (predicate) {
          case arith::CmpFPredicate::OEQ:
          case arith::CmpFPredicate::UEQ:
            return BoolRelationVVOp<tx::BoolEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::ONE:
          case arith::CmpFPredicate::UNE:
            return BoolRelationVVOp<tx::BoolUnEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OGE:
          case arith::CmpFPredicate::UGE:
            return BoolRelationVVOp<tx::BoolGreaterEqualVV>(op, adaptor,
                                                            rewriter);
          case arith::CmpFPredicate::OGT:
          case arith::CmpFPredicate::UGT:
            return BoolRelationVVOp<tx::BoolGreaterVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OLE:
          case arith::CmpFPredicate::ULE:
            return BoolRelationVVOp<tx::BoolLessEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OLT:
          case arith::CmpFPredicate::ULT:
            return BoolRelationVVOp<tx::BoolLessThenVV>(op, adaptor, rewriter);
          default:
            llvm_unreachable("Not yet supported");
            break;
          }
        })
        .Case<arith::TruncFOp>([&](auto elemWiseOp) {
          if (resultType.isF16())
            return RoundConvertOp<tx::FP32ToFP16Op>(op, adaptor, rewriter);
          else if (resultType.isBF16())
            return RoundConvertOp<tx::FP32ToBF16Op>(op, adaptor, rewriter);
          else
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for trunc "
                    "conversion");
        })
        .Default([&](auto elemWiseOp) {
          // WORKAROUND: Used to handle tl.arange(0, BLOCK_SIZE) which will
          // lower to linalg.generic + linalg.index + arith.index_cast and
          // other unsupported case now (eg: arith::extf)
          // TODO: Lower ops to tx81 if is supported

          // Affine dialect should handled before this pass. So here lower it
          // to scf.for
          if (failed(linalg::linalgOpToLoops(rewriter, op)))
            return rewriter.notifyMatchFailure(
                op, "Element-wise op not yet supported");
          rewriter.eraseOp(op);
          return success();
        });
  }
};

struct ReduceConversion : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;

private:
  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::MaximumFOp,
               arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
               arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
               arith::XOrIOp>(redOp);
  }

  template <typename Tx81Op>
  LogicalResult convertToReduceOp(linalg::ReduceOp op,
                                  typename linalg::ReduceOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto dims = op.getDimensions();
    if (dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support one dim reduce.");

    auto dim = dims[0];
    auto loc = op->getLoc();
    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto inputShape = inputType.getShape();
    auto newShape4D = reshapeReduceShapeTo4d(inputShape, dim);

    // Triton always assume shape is power of 2, we may not need channel norm
    int bitWidth = inputType.getElementType().getIntOrFloatBitWidth();
    int alignBase = bitWidth == 8 ? 128 : 64;

    auto input = op.getInputs()[0];
    auto srcPtr = createAddressFromMemref(rewriter, loc, input);

    int c = newShape4D.back();
    int alignedC = c;
    int alignedC0 = 0;
    // * < 4 aligned to 4;
    // * < align base aligned to next power of 2
    // * > align base aligned to nc'hwc_alignbase + nhwc_alignc0
    bool needChannelNorm =
        !(c >= 4 && c <= alignBase && c == next_power_of_two_64(c));
    bool reduceCDim = dim == inputShape.size() - 1;
    // Need cx aligned
    if (needChannelNorm) {

      int cx = c / alignBase;
      int c0 = c % alignBase;

      alignedC0 = c0 ? next_power_of_two_64(c0) : 0;
      if (c0 < 4 && c0 > 0) {
        // If c0 is not zero, we need to align it to 4
        alignedC0 = 4;
      }

      alignedC = cx * alignBase + alignedC0;
      SmallVector<int64_t, 4> alignedShape(newShape4D.begin(),
                                           newShape4D.end());
      alignedShape.back() = alignedC;

      auto alignedMemType =
          MemRefType::get(alignedShape, inputType.getElementType());
      auto alignedAlloc = rewriter.create<memref::AllocOp>(loc, alignedMemType);
      auto alignedPtr = createAddressFromMemref(rewriter, loc, alignedAlloc);

      auto channelNorm = rewriter.create<tx::ChannelNormOp>(
          loc, TypeRange({}), srcPtr, alignedPtr,
          rewriter.getDenseI64ArrayAttr(newShape4D),
          rewriter.getI16IntegerAttr(alignedC0),
          rewriter.getI16IntegerAttr(bitWidth));
      srcPtr = alignedPtr;
    }

    auto output = adaptor.getInits()[0];
    Value alignOutputPtr;
    bool needDeChannelNorm = reduceCDim || needChannelNorm;
    SmallVector<int64_t, 4> outputShape4D;
    if (needDeChannelNorm) {
      auto outputType = dyn_cast<MemRefType>(output.getType());
      SmallVector<int64_t, 4> alignedOutputShape4D;
      if (outputType.getRank() == 0) {
        outputShape4D =
            SmallVector<int64_t, 4>{1, 1, 1, outputType.getNumElements()};
        alignedOutputShape4D = SmallVector<int64_t, 4>{1, 1, 1, 4};
      } else {
        auto outputShape = outputType.getShape();
        outputShape4D = reshapeReduceShapeTo4d(outputShape, dim);
        alignedOutputShape4D =
            reduceCDim
                ? SmallVector<int64_t, 4>{1, 1, outputShape4D[2], 4}
                : SmallVector<int64_t, 4>{1, 1, outputShape4D[1], alignedC};
      }

      auto alignedMemType =
          MemRefType::get(alignedOutputShape4D, inputType.getElementType());
      auto alignedAlloc = rewriter.create<memref::AllocOp>(loc, alignedMemType);
      auto alignedPtr = createAddressFromMemref(rewriter, loc, alignedAlloc);
      alignOutputPtr = alignedPtr;
    } else {
      alignOutputPtr = createAddressFromMemref(rewriter, loc, output);
    }

    auto format = getFormatCode(inputType);
    auto reduceOp = rewriter.create<Tx81Op>(
        op->getLoc(), TypeRange{}, srcPtr, alignOutputPtr,
        rewriter.getUI32IntegerAttr(reduceCDim ? 0 /*reduce C dim*/
                                               : 1 /*reduce W dim*/),
        rewriter.getI64ArrayAttr(newShape4D),
        rewriter.getI16IntegerAttr(format));

    if (needDeChannelNorm) {
      auto outputPtr = createAddressFromMemref(rewriter, loc, output);
      alignedC0 = reduceCDim ? 4 : alignedC0;
      auto dechannelNorm = rewriter.create<tx::DechannelNormOp>(
          loc, TypeRange({}), alignOutputPtr, outputPtr,
          rewriter.getDenseI64ArrayAttr(outputShape4D),
          rewriter.getI16IntegerAttr(alignedC0) /*alignedC0*/,
          rewriter.getI16IntegerAttr(bitWidth));
    }

    rewriter.replaceOp(op, reduceOp);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reductionOps = getRegionOps(op);

    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f) or addf.");
    }
    auto redOp = reductionOps[0];

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    if (!isSupportedType(inputType)) {
      if (failed(linalg::linalgOpToLoops(rewriter, op)))
        return rewriter.notifyMatchFailure(op, "operation not supported yet.");
      rewriter.eraseOp(op);
      return success();
    }

    // TODO: Convert integer to float
    return llvm::TypeSwitch<Operation *, LogicalResult>(redOp)
        .Case<arith::AddIOp, arith::AddFOp>([&](auto redOp) {
          return convertToReduceOp<tx::ReduceSumOp>(op, adaptor, rewriter);
        })
        .Case<arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp,
              arith::MaxUIOp>([&](auto redOp) {
          return convertToReduceOp<tx::ReduceMaxOp>(op, adaptor, rewriter);
        })
        .Case<arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
              arith::MinUIOp>([&](auto redOp) {
          return convertToReduceOp<tx::ReduceMinOp>(op, adaptor, rewriter);
        })
        .Default([&](auto redOp) {
          // For other operation, we don't have specific tx81 op,
          // so we need to convert it to loops.
          if (failed(linalg::linalgOpToLoops(rewriter, op)))
            return rewriter.notifyMatchFailure(op,
                                               "operation not supported yet.");
          rewriter.eraseOp(op);
          return success();
        });
  }
};

struct BarrierConversion : public OpConversionPattern<mk::BarrierOp> {
  using OpConversionPattern<mk::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mk::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    rewriter.create<tx::BarrierOp>(loc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct PrintConversion : public OpConversionPattern<mk::PrintOp> {
  using OpConversionPattern<mk::PrintOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mk::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // printf scalar value.
    if (printScalar(op)) {
      if (op.getNumOperands() == 0) {
        createRuntimePrintScalarCall(rewriter, op.getPrefix(), std::nullopt);
      } else {
        createRuntimePrintScalarCall(rewriter, op.getPrefix(),
                                     adaptor.getOperands()[0], op.getHex(),
                                     op.getIsSigned()[0]);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // print memref value.
    createPrintMemrefCall(op, rewriter);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static std::string getFormatSubstr(Type type, bool hex = false,
                                     std::optional<int> width = std::nullopt,
                                     bool isSigned = false) {
    // If the `value` is a pointer, just return %p.
    if (isa<LLVM::LLVMPointerType>(type)) {
      return "%p";
    }
    // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in the
    // type (so 4 for fp16, 8 for int32, 16 for int64).
    if (hex) {
      // Ignore `width` for `hex` values, pad to typeWidth.
      std::string ret =
          "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
      if (type.getIntOrFloatBitWidth() > 32) {
        ret += "ll";
      }
      ret += "x";
      return ret;
    }

    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    }

    if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + (isSigned ? "lli" : "llu");
      else
        return prefix + (isSigned ? "i" : "u");
    }
    assert(false && "not supported type");
    return "";
  }

  // For printf, need to extend int32 or float64.
  static Value printfPromoteValue(RewriterBase &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    auto loc = UnknownLoc::get(context);
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    bool isUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (isUnsigned) {
        return b.zext(ui32_ty, value);
      } else {
        return b.sext(i32_ty, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      return b.fpext(f64_ty, value);
    }

    return value;
  }

  static LLVM::LLVMFuncOp
  getOrAddPrintFuncDecl(ConversionPatternRewriter &rewriter,
                        StringRef funcName = "__Print") {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType = {ptr_ty(ctx)};
    auto funcType =
        LLVM::LLVMFunctionType::get(i32_ty, argsType, /*isVarArg*/ true);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }

  static bool printScalar(mk::PrintOp op) {
    // Simply use printf if no operand or the operand is scalar.
    if (op.getNumOperands() == 0)
      return true;

    assert(op.getNumOperands() == 1);
    Type oprType = op.getOperands()[0].getType();
    return (oprType.isIntOrIndexOrFloat() || isa<triton::PointerType>(oprType));
  }

  static void createRuntimePrintScalarCall(ConversionPatternRewriter &rewriter,
                                           StringRef prefix,
                                           std::optional<Value> arg,
                                           bool hex = false,
                                           bool isSigned = false) {
    assert(!prefix.empty() && "printf with empty string not supported");
    auto loc = UnknownLoc::get(rewriter.getContext());
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << prefix;
    if (arg.has_value())
      os << getFormatSubstr(arg.value().getType(), hex, std::nullopt, isSigned);

    llvm::SmallString<64> formatStrNewline(formatStr);
    formatStrNewline.push_back('\n');
    formatStrNewline.push_back('\0');
    Value formatStrValue = LLVM::addStringToModule(
        loc, rewriter, "printfFormat_", formatStrNewline);

    SmallVector<Value> allArgs{formatStrValue};
    if (arg.has_value())
      allArgs.push_back(printfPromoteValue(rewriter, arg.value()));
    b.call(getOrAddPrintFuncDecl(rewriter), allArgs);
  }

  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtr, true);
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             StringRef funcName = "__Print") {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName,
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, funcName);
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }

  static void createPrintMemrefCall(mk::PrintOp op,
                                    ConversionPatternRewriter &rewriter) {
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    Type memElementType = memRefType.getElementType();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    std::string formatSpecifierStr = getFormatSubstr(
        memElementType, op.getHex(), std::nullopt, op.getIsSigned()[0]);
    formatSpecifierStr += " \0";
    auto prefix = op.getPrefix();
    std::string prefixNewline = "\n" + prefix.str();
    Value prefixValue = getOrCreateGlobalString(
        loc, rewriter, "frmt_prefix" + prefix.str(),
        StringRef(prefixNewline.c_str(), 128), parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec" + formatSpecifierStr,
        StringRef(formatSpecifierStr.c_str(), 8), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // print prefix firstly.
    rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                  prefixValue);

    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());

      if (i != e - 1)
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                      newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    Value elementLoad =
        rewriter.create<memref::LoadOp>(loc, op.getOperands()[0], loopIvs);
    if (elementLoad.getType() == rewriter.getF32Type())
      elementLoad = rewriter.create<mlir::LLVM::FPExtOp>(
          loc, rewriter.getF64Type(), elementLoad);
    else if (elementLoad.getType() == rewriter.getI8Type())
      elementLoad = rewriter.create<mlir::LLVM::SExtOp>(
          loc, rewriter.getI32Type(), elementLoad);
    rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Legalize magic kernel operations to be convertible to Tx81 operations
// patterns
//===----------------------------------------------------------------------===//
namespace {
struct ElementwiseRewrite : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  void initialize() {
    // Register conversions from SIOp to FPOp
    registerSIOpMapFPOp<arith::AddIOp, arith::AddFOp>();
    registerSIOpMapFPOp<arith::SubIOp, arith::SubFOp>();
    registerSIOpMapFPOp<arith::MulIOp, arith::MulFOp>();
    registerSIOpMapFPOp<arith::MaxSIOp, arith::MaximumFOp>();
    registerSIOpMapFPOp<arith::MinSIOp, arith::MinimumFOp>();
    registerSIOpMapFPOp<math::AbsIOp, math::AbsFOp>();
  }

  template <typename SIOp, typename FPOp> void registerSIOpMapFPOp() {
    OperationName SIOpName(SIOp::getOperationName(), getContext());
    assert(!SIOpMapFPOpFns.contains(SIOpName) &&
           "SIOp already registered for conversion to FPOp");
    SIOpMapFPOpFns[SIOpName] = [](OpBuilder &b, Location loc, ValueRange args) {
      // Create the floating-point operation
      Value fpVal = b.create<FPOp>(loc, b.getF32Type(), args.drop_back());
      b.create<linalg::YieldOp>(loc, fpVal);
    };
  }

  LogicalResult convertSIOpToFPOp(linalg::GenericOp op, Operation *elemWiseOp,
                                  PatternRewriter &rewriter) const {
    OperationName SIOpName = elemWiseOp->getName();
    if (!SIOpMapFPOpFns.contains(SIOpName))
      return failure();

    Location loc = op->getLoc();
    auto inputs = op.getInputs();
    auto output = op.getOutputs()[0];
    auto outputTy = cast<MemRefType>(output.getType());
    auto outputEleTy = outputTy.getElementType();

    assert(op.getOutputs().size() == 1 &&
           "Elementwise conversion only support single output");
    assert((outputEleTy.isInteger(16) || outputEleTy.isInteger(32) ||
            outputEleTy.isInteger(64)) &&
           "Output type must be integer type (16, 32 or 64 bits)");
    assert(
        llvm::all_of(
            inputs, [&outputTy](Value v) { return v.getType() == outputTy; }) &&
        "All inputs must have the same type as output");

    MemRefType fpMemrefTy =
        MemRefType::get(outputTy.getShape(), rewriter.getF32Type(),
                        outputTy.getLayout(), outputTy.getMemorySpace());
    auto rank = fpMemrefTy.getRank();
    auto id = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap> idMaps = {id, id};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    SmallVector<Value> fpInputs;
    for (auto input : inputs) {
      auto fpInput = rewriter.create<memref::AllocOp>(loc, fpMemrefTy);
      rewriter.create<linalg::GenericOp>(
          loc, ValueRange{input}, ValueRange{fpInput}, idMaps, iterators,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // Convert integer to float
            Value fpVal =
                b.create<arith::SIToFPOp>(loc, b.getF32Type(), args[0]);
            b.create<linalg::YieldOp>(loc, fpVal);
          });
      fpInputs.push_back(fpInput);
    }

    auto fpOutput = rewriter.create<memref::AllocOp>(loc, fpMemrefTy);
    rewriter.create<linalg::GenericOp>(
        loc, fpInputs, ValueRange{fpOutput}, op.getIndexingMapsArray(),
        op.getIteratorTypesArray(), SIOpMapFPOpFns.at(SIOpName));
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, ValueRange{fpOutput}, ValueRange{output}, idMaps, iterators,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Convert float to integer
          Value intVal = b.create<arith::FPToSIOp>(loc, outputEleTy, args[0]);
          b.create<linalg::YieldOp>(loc, intVal);
        });
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto regionOps = getRegionOps<linalg::GenericOp>(op);
    if (regionOps.size() != 1)
      return failure();

    return convertSIOpToFPOp(op, regionOps[0], rewriter);
  }

private:
  // Map from SIOp to FPOp conversion functions
  llvm::DenseMap<OperationName,
                 std::function<void(OpBuilder &, Location, ValueRange)>>
      SIOpMapFPOpFns;
};
} // namespace

void mlir::triton::populateMKToTx81CanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ElementwiseRewrite>(patterns.getContext());
}

void mlir::triton::populateMKToTx81ConversionPatterns(
    RewritePatternSet &patterns) {

  MKToTx81TypeConverter typeConverter;

  // Add type conversion patterns
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);

  // clang-format off
  patterns.add<MemoryCopyConvertPattern,
               ReduceConversion,
               TransposeOpConversion,
               LinalgFillOpConversion,
               MKDotToTx81GemmOpConversion,
               MKSigmoidToTx81SigmoidOpConversion,
               ArgMinConversion,
               ArgMaxConversion,
               GatherConvertPattern,
               ElementwiseConversion,
               BarrierConversion,
               PrintConversion>(
      patterns.getContext());
  // clang-format on
}
