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
#include "Tx81/instr_def.h"
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
Data_Format getFormatCode(Type type) {
  if (type.isF32()) {
    return Fmt_FP32;
  } else if (type.isF16()) {
    return Fmt_FP16;
  } else if (type.isBF16()) {
    return Fmt_BF16;
  } else if (type.isInteger(8)) {
    return Fmt_INT8;
  }

  // Default to F32 format
  return Fmt_FP32;
}

// Get element count from shape
int32_t getElementCount(ArrayRef<int64_t> shape) {
  int32_t elementCount = 1;
  for (auto dim : shape) {
    if (dim > 0) {
      elementCount *= dim;
    }
  }
  return elementCount;
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

Value createAddressFromMemref(ConversionPatternRewriter &rewriter, Location loc,
                              Value memref) {
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value indexBasePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), stridedMetadata.getBaseBuffer());
  Value offset = stridedMetadata.getOffset();
  Value offsetPtr = rewriter.create<arith::AddIOp>(loc, indexBasePtr.getType(),
                                                   indexBasePtr, offset);
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
  Value offset = stridedMetadata.getOffset();
  Value offsetPtr = rewriter.create<arith::AddIOp>(loc, indexBasePtr.getType(),
                                                   indexBasePtr, offset);
  Value i64SPMPtr = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), offsetPtr);

  // FIXME: For multi-dimensional(rank > 2), strides need to be multiplied.
  return {i64SPMPtr, stridedMetadata.getSizes(), stridedMetadata.getStrides()};
}

static SmallVector<Value> padSizesToNHWC(ConversionPatternRewriter &rewriter,
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
static SmallVector<Value> padStridesToNHWC(ConversionPatternRewriter &rewriter,
                                           Location loc, ValueRange strides) {
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

class MemoryCopyConvertPattern : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  // Workaround: Avoid analyzing control flow as much as possible。
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
    bool isSrcSPM = isOperandMemorySpaceSPM(adaptor.getSource());
    bool isDstSPM = isOperandMemorySpaceSPM(adaptor.getTarget());

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
          elemCount,                     // Element count
          rewriter.getI16IntegerAttr(0), // Round mode
          rewriter.getI16IntegerAttr(getFormatCode(
              inputType)) // Format (5 = f32, assuming f32 for now)
      );
    } else if (isDstSPM) {
      auto nhwcShape = padSizesToNHWC(rewriter, op->getLoc(), srcSizes);
      auto nhwcStrides = padStridesToNHWC(rewriter, op->getLoc(), srcStrides);

      auto rdmaOp = rewriter.create<tx::RdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          nhwcShape,   // NHWC shape
          nhwcStrides, // NHWC stride
          rewriter.getI32IntegerAttr(getFormatCode(
              inputType)) // Format (5 = f32, assuming f32 for now)
      );
    } else {
      auto nhwcShape = padSizesToNHWC(rewriter, op->getLoc(), dstSizes);
      auto nhwcStrides = padStridesToNHWC(rewriter, op->getLoc(), dstSizes);

      auto wdmaOp = rewriter.create<tx::WdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          nhwcShape,   // NHWC shape
          nhwcStrides, // NHWC stride
          rewriter.getI32IntegerAttr(getFormatCode(
              inputType)) // Format (5 = f32, assuming f32 for now)
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

    // Convert the fill value to int64
    if (fillValue.getType().isF32()) {
      // If it's a float constant, bitcast it to int
      fillValue = rewriter.create<arith::BitcastOp>(
          op.getLoc(), rewriter.getI32Type(), fillValue);
    } else if (fillValue.getType().isF16()) {
      auto extf = rewriter.create<arith::ExtFOp>(
          op.getLoc(), rewriter.getF32Type(), fillValue);
      fillValue = rewriter.create<arith::BitcastOp>(
          op.getLoc(), rewriter.getI32Type(), extf);
    }

    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), srcSizes);

    // Create a MemsetOp to fill the SPM buffer
    // TODO: Support format code for different element types
    auto memsetOp = rewriter.create<tx::MemsetOp>(
        op.getLoc(), rewriter.getI64Type(), srcPtr, fillValue, elemCount,
        rewriter.getI32ArrayAttr({}), // Strides (empty for simple fill)
        rewriter.getI32ArrayAttr({}), // Iterations (empty for simple fill)
        rewriter.getI16IntegerAttr(5) // Format (5 = f32, assuming f32 for now)
    );

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// mk.dot to tx.gemm Conversion Pattern
//===----------------------------------------------------------------------===//

class MKDotToTx81GemmOpConversion
    : public OpConversionPattern<mlir::mk::DotOp> {
public:
  using OpConversionPattern<mlir::mk::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Extract dimensions from tensor types
    MemRefType aTensorType = cast<MemRefType>(op.getA().getType());
    MemRefType bTensorType = cast<MemRefType>(op.getB().getType());

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
    auto a = createAddressFromMemref(rewriter, loc, adaptor.getA());
    auto b = createAddressFromMemref(rewriter, loc, adaptor.getB());
    auto c = createAddressFromMemref(rewriter, loc, adaptor.getC());
    auto zeros = createAddressFromMemref(rewriter, loc, adaptor.getZeroes());

    // Create GemmOp
    rewriter.create<tx::GemmOp>(
        op.getLoc(), rewriter.getI64Type(),
        a,                           // src_a (Matrix A in SPM)
        b,                           // src_b (Matrix B in SPM)
        c,                           // src_bias (optional accumulation)
        zeros,                       // zeroes,
        dims,                        // dimensions [M,K,N]
        rewriter.getBoolAttr(false), // en_psum
        zeros, // WORKAROUND: psum_addr (using zeroes buffer)
        rewriter.getBoolAttr(false),                // trans_src_a
        rewriter.getBoolAttr(false),                // trans_src_b
        rewriter.getI32IntegerAttr(1),              // batch_src_a
        rewriter.getI32IntegerAttr(1),              // batch_src_b
        rewriter.getBoolAttr(false),                // en_leaky_relu
        rewriter.getBoolAttr(op.getC() != nullptr), // en_bias
        rewriter.getBoolAttr(false),                // en_neg_scale
        rewriter
            .create<arith::ConstantIntOp>(op.getLoc(), 0, rewriter.getI64Type())
            .getResult(),            // src_neg_scale
        rewriter.getBoolAttr(false), // en_pos_scale
        rewriter
            .create<arith::ConstantIntOp>(op.getLoc(), 0, rewriter.getI64Type())
            .getResult(),              // src_pos_scale
        rewriter.getI32IntegerAttr(3), // src_fmt (3 = f32)
        rewriter.getI32IntegerAttr(3)  // dst_fmt (3 = f32)
    );
    // Op has no result value
    rewriter.eraseOp(op);

    return success();
  }
};

struct ElementwiseConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

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
    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(
        loc, rewriter.getI64Type(), input0, input1, output, elemCount,
        rewriter.getI16IntegerAttr(0), // Round mode
        rewriter.getI16IntegerAttr(
            getFormatCode(inputType)) // Format (5 = f32, assuming f32 for now)
    );
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

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto regionOps = getRegionOps<linalg::GenericOp>(op);

    // Check if the operation is elementwise
    if (op.getIteratorTypesArray().front() != utils::IteratorType::parallel)
      return rewriter.notifyMatchFailure(op, "Only support elementwise op.");

    auto elemWiseOp = regionOps[0];
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
        .Case<arith::ExtFOp>([&](auto elemWiseOp) {
          return NormalConvertOp<tx::FP16ToFP32Op>(op, adaptor, rewriter);
        })
        .Case<arith::SIToFPOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType =
              cast<MemRefType>(op.getInputs()[0].getType()).getElementType();
          auto outputType =
              cast<MemRefType>(op.getOutputs()[0].getType()).getElementType();
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
          auto inputType =
              cast<MemRefType>(op.getInputs()[0].getType()).getElementType();
          auto outputType =
              cast<MemRefType>(op.getOutputs()[0].getType()).getElementType();
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
        .Default([&](auto elemWiseOp) {
          // WORKAROUND: Used to handle tl.arange(0, BLOCK_SIZE) which will
          // lower to linalg.generic + linalg.index + arith.index_cast and
          // other unsupported case now (eg: arith::extf)
          // TODO: Lower ops to tx81 if is supported
          if (failed(linalg::linalgOpToAffineLoops(rewriter, op)))
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
