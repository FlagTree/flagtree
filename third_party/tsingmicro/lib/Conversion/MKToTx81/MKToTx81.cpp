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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "mk-to-tx81"

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
  if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
    for (auto dim : memrefType.getShape())
      dims.push_back(static_cast<int32_t>(dim));
  } else if (auto tensorType = mlir::dyn_cast<TensorType>(type)) {
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
  auto elemType = mlir::cast<MemRefType>(memref.getType()).getElementType();
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

static std::tuple<Value, ValueRange, ValueRange>
createMetadata(ConversionPatternRewriter &rewriter, Location loc,
               Value operand) {
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
  Value indexBasePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), stridedMetadata.getBaseBuffer());
  auto elemType = mlir::cast<MemRefType>(operand.getType()).getElementType();
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
  nhwcStrides.pop_back();
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

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->hasAttr("srcSpm") && op->hasAttr("dstSpm") &&
           "Can't get memory space attribute\n");
    bool isSrcSPM = mlir::cast<IntegerAttr>(op->getAttr("srcSpm")).getInt();
    bool isDstSPM = mlir::cast<IntegerAttr>(op->getAttr("dstSpm")).getInt();

    // DDR to DDR
    if (!isSrcSPM && !isDstSPM)
      return rewriter.notifyMatchFailure(
          op, "Can not copy memory from DDR to DDR.\n");

    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getSource());
    auto [dstPtr, dstSizes, dstStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getTarget());

    auto inputType = dyn_cast<MemRefType>(op.getSource().getType());
    // SPM to SPM
    if (isSrcSPM && isDstSPM) {
      // FIXME: Only support 1d for now, take sizes[0] as elemCount.
      auto elemCount = calculateElemCount(rewriter, op->getLoc(), srcSizes);

      // WORKAROUND: Assume no mask.
      auto constValue = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), 0, rewriter.getI32Type());

      rewriter.create<tx::AddVSOp>(
          op->getLoc(), rewriter.getI64Type(), srcPtr, constValue, dstPtr,
          elemCount,                                           // Element count
          rewriter.getI16IntegerAttr(0),                       // Round mode
          rewriter.getI16IntegerAttr(getFormatCode(inputType)) // Format
      );
    } else if (isDstSPM) {
      auto nhwcShape = padSizesToNHWC(rewriter, op->getLoc(), srcSizes);
      auto nhwcStrides = padStridesToNHWC(rewriter, op->getLoc(), srcStrides);

      auto rdmaOp = rewriter.create<tx::RdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          nhwcShape,                                           // NHWC shape
          nhwcStrides,                                         // NHWC stride
          rewriter.getI32IntegerAttr(getFormatCode(inputType)) // Format
      );
    } else {
      auto nhwcShape = padSizesToNHWC(rewriter, op->getLoc(), dstSizes);
      auto nhwcStrides = padStridesToNHWC(rewriter, op->getLoc(), dstSizes);

      auto wdmaOp = rewriter.create<tx::WdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          nhwcShape,                                           // NHWC shape
          nhwcStrides,                                         // NHWC stride
          rewriter.getI32IntegerAttr(getFormatCode(inputType)) // Format
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
    assert(bitWidth == 16 ||
           bitWidth == 32 && "Only support 16/32 fill value\n");

    // AddVS value need has fmt with input fmt and only support float type
    Data_Format fmt = bitWidth == 16 ? Fmt_FP16 : Fmt_FP32;

    if (inputType.isInteger()) {
      auto floatType =
          bitWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
      fillValue =
          rewriter.create<arith::SIToFPOp>(op.getLoc(), floatType, fillValue);
    }

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

//===----------------------------------------------------------------------===//
// mk.dot to tx.gemm Conversion Pattern
//===----------------------------------------------------------------------===//

class MKDotToTx81GemmOpConversion
    : public OpConversionPattern<mlir::mk::DotOp> {

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

public:
  using OpConversionPattern<mlir::mk::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract dimensions from tensor types
    MemRefType aTensorType = mlir::cast<MemRefType>(op.getA().getType());
    MemRefType bTensorType = mlir::cast<MemRefType>(op.getB().getType());
    assert(aTensorType.getElementType() == bTensorType.getElementType() &&
           "a and b must have the same element type");
    MemRefType zeroTensorType =
        mlir::cast<MemRefType>(op.getZeroes().getType());
    Data_Format srcFmt = getFormatCode(aTensorType);
    Data_Format dstFmt = getFormatCode(zeroTensorType);

    // Get converted operands
    auto loc = op.getLoc();

    auto aShape = aTensorType.getShape();
    auto bShape = bTensorType.getShape();

    // Matrix dimensions M, K, N for GEMM
    int32_t M = aShape[0];
    int32_t K = aShape[1];
    int32_t N = bShape[1];

    // Create dimensions array attribute [M, K, N]
    auto dims = rewriter.getI32ArrayAttr({M, K, N});

    // Get operand ptr
    auto [aPtr, aSizes, aStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getA());
    auto [bPtr, bSizes, bStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getB());
    auto [cPtr, cSizes, cStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getC());
    // Assume input type is same. Tx neural engine not support fp32 for input
    if (aTensorType.getElementType().isF32()) {
      srcFmt = Data_Format::Fmt_TF32;
      fp32ToTF32(rewriter, op->getLoc(), aSizes, aPtr);
      fp32ToTF32(rewriter, op->getLoc(), bSizes, bPtr);
      fp32ToTF32(rewriter, op->getLoc(), cSizes, cPtr);
    }

    auto dst = createAddressFromMemref(rewriter, loc, adaptor.getZeroes());

    auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                      rewriter.getI64Type());

    // Create GemmOp
    rewriter.create<tx::GemmOp>(
        op.getLoc(), rewriter.getI64Type(),
        aPtr,                        // src_a (Matrix A in SPM)
        bPtr,                        // src_b (Matrix B in SPM)
        cPtr,                        // src_bias (optional accumulation)
        dst,                         // dst,
        dims,                        // dimensions [M,K,N]
        rewriter.getBoolAttr(false), // en_psum
        dst,                         // WORKAROUND: psum_addr (using dst buffer)
        rewriter.getBoolAttr(false), // trans_src_a
        // NOTE: (N, K) is thought not trans in hardware
        rewriter.getBoolAttr(true),                    // trans_src_b.
        rewriter.getI32IntegerAttr(1),                 // batch_src_a
        rewriter.getI32IntegerAttr(1),                 // batch_src_b
        rewriter.getI32IntegerAttr(ActFuncMode::None), // relu_mode.
        rewriter.getBoolAttr(op.getC() != nullptr),    // en_bias
        rewriter.getBoolAttr(false),                   // en_neg_scale
        zero,                                          // src_neg_scale
        rewriter.getBoolAttr(false),                   // en_pos_scale
        zero,                                          // src_pos_scale
        rewriter.getI32IntegerAttr(srcFmt),            // src_fmt
        rewriter.getI32IntegerAttr(dstFmt)             // dst_fmt
    );
    // Op has no result value
    rewriter.eraseOp(op);

    return success();
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
    insertConvertTypeOp(input1, inputType, elemCount, rewriter, loc);
    insertConvertTypeOp(output, inputType, elemCount, rewriter, loc);

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCount,
                           rewriter.getI16IntegerAttr(0), // Round mode
                           rewriter.getI16IntegerAttr(srcFmt));

    insertRestoreTypeOp(input0, inputType, elemCount, rewriter, loc);
    insertRestoreTypeOp(input1, inputType, elemCount, rewriter, loc);
    insertRestoreTypeOp(output, inputType, elemCount, rewriter, loc);

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

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(
        loc, rewriter.getI64Type(), input0, input1, output, elemCount,
        rewriter.getI16IntegerAttr(getFormatCode(inputType)) // Format
    );

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

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto regionOps = getRegionOps<linalg::GenericOp>(op);

    // Check if the operation is elementwise
    if (op.getIteratorTypesArray().front() != utils::IteratorType::parallel)
      return rewriter.notifyMatchFailure(op, "Only support elementwise op.");

    if (regionOps.size() != 1) {
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
        .Case<arith::MaxSIOp, arith::MaxUIOp, arith::MaximumFOp,
              arith::MaxNumFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MaxVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::MinSIOp, arith::MinUIOp, arith::MinimumFOp,
              arith::MinNumFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MinVVOp>(op, adaptor, rewriter);
        })
        .Case<math::AbsFOp, math::AbsIOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::AbsVVOp>(op, adaptor, rewriter);
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
        .Case<arith::SIToFPOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType = mlir::cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = mlir::cast<MemRefType>(op.getOutputs()[0].getType())
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
          auto inputType = mlir::cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = mlir::cast<MemRefType>(op.getOutputs()[0].getType())
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
// FIXME: Now BoolLessThenOp run fail on board. Need more op information from
// Tx81
#if 0
        .Case<arith::CmpIOp>([&](auto elemWiseOp) {
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
#endif
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

          // Affine dialect should handled before this pass. So here lower it to
          // scf.for
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
               arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(
        redOp);
  }

  template <typename Tx81Op>
  LogicalResult convertToReduceOp(linalg::ReduceOp op,
                                  typename linalg::ReduceOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto dims = op.getDimensions();
    if (dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support one dim reduce.");
    auto dim = dims[0];
    auto input =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getInputs()[0]);
    auto output =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getInits()[0]);
    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto inputShape = inputType.getShape();
    // TODO: Support any rank
    if (inputShape.size() > 1)
      return rewriter.notifyMatchFailure(op, "Rank > 1 unsupported yet.");

    if (dim && dim >= inputShape.size())
      return rewriter.notifyMatchFailure(op,
                                         "Dimensions attribute > input rank !");

    int64_t inputSize = inputShape.empty() ? 1 : inputShape[0];

    SmallVector<int64_t, 4> reduceShape = {1, 1, 1, inputSize};
    auto format = getFormatCode(inputType);
    auto reduceOp = rewriter.create<Tx81Op>(
        op->getLoc(), TypeRange{}, input, output,
        rewriter.getUI32IntegerAttr(dim), rewriter.getI64ArrayAttr(reduceShape),
        rewriter.getI16IntegerAttr(format));
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
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return failure();
        });
  }
};

} // namespace

void mlir::triton::populateMKToTx81CanonicalizationPatterns(
    RewritePatternSet &patterns) {}

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
               LinalgFillOpConversion,
               MKDotToTx81GemmOpConversion,
               ElementwiseConversion>(
      patterns.getContext());
  // clang-format on
}
