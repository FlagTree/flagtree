//===--------------------- Tx81ToLLVM.cpp ---------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to convert operations from tx dialect to
// LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Tx81ToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tx81-to-llvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//
// Crt func name
const char addVVFuncName[] = "__AddVV";
const char subVVFuncName[] = "__SubVV";
const char mulVVFuncName[] = "__MulVV";
const char divVVFuncName[] = "__DivVV";
const char absVVFuncName[] = "__AbsVV";
const char rsqrtVVFuncName[] = "__RsqrtVV";
const char sqrtVVFuncName[] = "__SqrtVV";
const char lnFuncName[] = "__Ln";
const char log2FuncName[] = "__Log2";
const char expFuncName[] = "__Exp";
const char pow2FuncName[] = "__Pow2";
const char sinFuncName[] = "__Sin";
const char cosFuncName[] = "__Cos";
const char addVSFuncName[] = "__AddVS";
const char subVSFuncName[] = "__SubVS";
const char mulVSFuncName[] = "__MulVS";
const char divVSFuncName[] = "__DivVS";
const char reduceSumFuncName[] = "__ReduceSum";
const char reduceMaxFuncName[] = "__ReduceMax";
const char reduceMinFuncName[] = "__ReduceMin";
const char fp16ToFp32FuncName[] = "__FP16_FP32";
const char int16ToFp16FuncName[] = "__INT16_FP16";
const char int16ToFp32FuncName[] = "__INT16_FP32";
const char int32ToFp16FuncName[] = "__INT32_FP16";
const char int32ToFp32FuncName[] = "__INT32_FP32";
const char fp16ToInt8FuncName[] = "__FP16_INT8";
const char fp16ToInt16FuncName[] = "__FP16_INT16";
const char fp16ToInt32FuncName[] = "__FP16_INT32";
const char fp32ToInt8FuncName[] = "__FP32_INT8";
const char fp32ToInt16FuncName[] = "__FP32_INT16";
const char fp32ToInt32FuncName[] = "__FP32_INT32";
const char boolEqualVVFuncName[] = "__BoolEqualVV";
const char boolUnEqualVVFuncName[] = "__BoolUnEqualVV";
const char boolGreaterEqualVVFuncName[] = "__BoolGreaterEqualVV";
const char boolGreaterVVFuncName[] = "__BoolGreaterVV";
const char boolLessEqualVVFuncName[] = "__BoolLessEqualVV";
const char boolLessVVFuncName[] = "__BoolLessThenVV";
const char fp32ToFp16FuncName[] = "__FP32_FP16";
const char fp32ToBf16FuncName[] = "__FP32_BF16";
const char fp32ToTF32FuncName[] = "__FP32_TF32";
const char andVVFuncName[] = "__AndVV";
const char orVVFuncName[] = "__OrVV";
const char xorVVFuncName[] = "__XorVV";
const char MaxVVFuncName[] = "__MaxVV";
const char MinVVFuncName[] = "__MinVV";

// Function to declare Tx81 runtime function
Value declareTx81Function(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes) {
  // Check if the function already exists
  Operation *funcOp = module.lookupSymbol(name);
  if (funcOp)
    return builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), name);

  // Create function type
  Type funcType = LLVM::LLVMFunctionType::get(resultType, argumentTypes,
                                              /*isVarArg=*/false);

  // Create a function declaration
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(module.getBody());

  builder.create<LLVM::LLVMFuncOp>(loc, name, funcType,
                                   LLVM::Linkage::External);

  builder.restoreInsertionPoint(ip);

  // Return function pointer
  return builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), name);
}

static Value adjustElemCountType(ConversionPatternRewriter &rewriter,
                                 Location loc, Value elemCount) {
  Value newElemCount = elemCount;
  if (isa<IndexType>(elemCount.getType())) {
    newElemCount = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), elemCount);
  } else if (isa<IntegerType>(elemCount.getType())) {
    auto elemCountType = dyn_cast<IntegerType>(elemCount.getType());
    if (elemCountType.isInteger(64))
      newElemCount = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI32Type(), elemCount);
  }
  return newElemCount;
}

static Value castIndexToInt32(ConversionPatternRewriter &rewriter, Location loc,
                              Value indexOp) {
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                             indexOp);
}

//===----------------------------------------------------------------------===//
// Arith Operation Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert constant operations to LLVM constants
struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the constant value
    auto constAttr = op.getValue();

    // Get the result type
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Handle different attribute types
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constAttr)) {
      // Convert integer attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, intAttr);
      return success();
    } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(constAttr)) {
      // Convert float attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, floatAttr);
      return success();
    } else if (auto boolAttr = mlir::dyn_cast<BoolAttr>(constAttr)) {
      // Convert bool attribute to i1
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, resultType,
          rewriter.getIntegerAttr(resultType, boolAttr.getValue()));
      return success();
    }

    return failure();
  }
};

// Convert arith.index_cast to appropriate LLVM conversions
struct IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get source and result types
    auto srcType = adaptor.getIn().getType();
    auto dstType = getTypeConverter()->convertType(op.getResult().getType());

    // Convert from index to specific integer type
    if (mlir::isa<LLVM::LLVMPointerType>(srcType) &&
        mlir::isa<IntegerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Convert from specific integer type to index
    if (mlir::isa<IntegerType>(srcType) &&
        mlir::isa<LLVM::LLVMPointerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Handle integer to integer casts
    if (mlir::isa<IntegerType>(srcType) && mlir::isa<IntegerType>(dstType)) {
      unsigned srcWidth = mlir::cast<IntegerType>(srcType).getWidth();
      unsigned dstWidth = mlir::cast<IntegerType>(dstType).getWidth();

      if (srcWidth < dstWidth) {
        // Sign extend if source is signed, zero extend otherwise
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, adaptor.getIn());
      } else if (srcWidth > dstWidth) {
        // Truncate
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, dstType,
                                                   adaptor.getIn());
      } else {
        // Same width, just pass through
        rewriter.replaceOp(op, adaptor.getIn());
      }
      return success();
    }

    return failure();
  }
};

// Convert arith.addi to LLVM add
struct AddIOpConversion : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

// Convert arith.muli to LLVM mul
struct MulIOpConversion : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tx81 Operation Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert tx81.rdma to LLVM call to crt __Rdma function
struct RdmaOpConversion : public OpConversionPattern<tx::RdmaOp> {
  using OpConversionPattern<tx::RdmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::RdmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Rdma runtime function if not already declared
    /*
    void __Rdma(uint64_t *src, uint64_t *dst, int shape_n, int shape_h, int
    shape_w, int shape_c, int stride_n, int stride_h, int stride_w, int
    *strides, uint32_t fmt)
    */
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // target
        i32Ty,   // shape_n
        i32Ty,   // shape_h
        i32Ty,   // shape_w
        i32Ty,   // shape_c
        i32Ty,   // stride_n
        i32Ty,   // stride_h
        i32Ty,   // stride_w
        i32Ty    // fmt
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(), "__Rdma",
                                        i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    // Get the operands
    Value target = adaptor.getTarget();
    target = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, target);

    ValueRange shape = adaptor.getShape();
    Value shape0 = castIndexToInt32(rewriter, op->getLoc(), shape[0]);
    Value shape1 = castIndexToInt32(rewriter, op->getLoc(), shape[1]);
    Value shape2 = castIndexToInt32(rewriter, op->getLoc(), shape[2]);
    Value shape3 = castIndexToInt32(rewriter, op->getLoc(), shape[3]);

    ValueRange strides = adaptor.getStrides();
    Value stride0 = castIndexToInt32(rewriter, op->getLoc(), strides[0]);
    Value stride1 = castIndexToInt32(rewriter, op->getLoc(), strides[1]);
    Value stride2 = castIndexToInt32(rewriter, op->getLoc(), strides[2]);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __Rdma
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{i8PtrTy}, "__Rdma", // funcPtr,
        ValueRange{src, target, shape0, shape1, shape2, shape3, stride0,
                   stride1, stride2, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.wdma to LLVM call to __Wdma function
struct WdmaOpConversion : public OpConversionPattern<tx::WdmaOp> {
  using OpConversionPattern<tx::WdmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::WdmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Wdma runtime function if not already declared
    /*
    void __Wdma(uint64_t *src, uint64_t *dst, int shape_n, int shape_h, int
    shape_w, int shape_c, int stride_n, int stride_h, int stride_w, int
    *strides, uint32_t fmt)
    */
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // target
        i32Ty,   // shape_n
        i32Ty,   // shape_h
        i32Ty,   // shape_w
        i32Ty,   // shape_c
        i32Ty,   // stride_n
        i32Ty,   // stride_h
        i32Ty,   // stride_w
        i32Ty    // fmt
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(), "__Wdma",
                                        i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    // Get the operands
    Value target = adaptor.getTarget();

    // Need to bitcast src to i8*
    target = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, target);

    ValueRange shape = adaptor.getShape();
    Value shape0 = castIndexToInt32(rewriter, op->getLoc(), shape[0]);
    Value shape1 = castIndexToInt32(rewriter, op->getLoc(), shape[1]);
    Value shape2 = castIndexToInt32(rewriter, op->getLoc(), shape[2]);
    Value shape3 = castIndexToInt32(rewriter, op->getLoc(), shape[3]);

    ValueRange strides = adaptor.getStrides();
    Value stride0 = castIndexToInt32(rewriter, op->getLoc(), strides[0]);
    Value stride1 = castIndexToInt32(rewriter, op->getLoc(), strides[1]);
    Value stride2 = castIndexToInt32(rewriter, op->getLoc(), strides[2]);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __Wdma
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Wdma", // funcPtr,
        ArrayRef<Value>{src, target, shape0, shape1, shape2, shape3, stride0,
                        stride1, stride2, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.mask_move to LLVM call to __MaskMove function
struct MaskMoveOpConversion : public OpConversionPattern<tx::MaskMoveOp> {
  using OpConversionPattern<tx::MaskMoveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::MaskMoveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __MaskMove runtime function if not already declared
    // Signature: void* __MaskMove(void* source, void* target, uint32_t
    // elem_count, int32_t* masks, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy,  // source
        i8PtrTy,  // target
        i32Ty,    // elem_count
        i32PtrTy, // masks
        i32Ty     // fmt
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        "__MaskMove", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value target = adaptor.getTarget();

    // Need to bitcast src to i8*
    target = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, target);
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op->getLoc(), elemCount);

    // Handle mask arrays
    // For simplicity, we'll create empty arrays
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(op.getLoc(), i32PtrTy);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __MaskMove
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__MaskMove", // funcPtr,
        ArrayRef<Value>{src, target, elemCount, nullPtr, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.binary op to LLVM call
template <typename Tx81Op, const char *funcPrefix>
struct ReduceOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __ReduceSum(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
    // uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value srcB = adaptor.getSrc();
    Value dst = adaptor.getDst();
    // Need to bitcast src to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert dim attribute to Value
    Value dim = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDim()));

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, dst, dim, shape_n, shape_h, shape_w, shape_c,
                        fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert tx81.elementwise op to LLVM call
template <typename Tx81Op, const char *funcPrefix>
struct ElementWiseOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;
  // using OpConversionPattern<Tx81Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy,

                                      i32Ty,   i32Ty,   i32Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct UnaryOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Abs(void* src, void* dst, uint32_t elem_count,
    // uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    // Need to bitcast src to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    Value out = adaptor.getOut();
    // Need to bitcast out to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// FIXME: Use trait to refactor the BinaryVSOpConversion and
// ElementWiseOpConversion
template <typename Tx81Op, const char *funcPrefix>
struct BinaryVSOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i32Ty, i8PtrTy,
                                      i32Ty,   i32Ty, i32Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);

    Value srcB = adaptor.getValue();

    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct BinaryLogicVVOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __XorVV(void* a, void* b, void* out, uint32_t
    // elem_count, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src0_addr
        i8PtrTy, // src1_addr
        i8PtrTy, // dst_addr
        i32Ty,   // elem_count
        i32Ty    // fmt
    };

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename BoolRelationVVOp, const char *funcPrefix>
struct BoolRelationVVOpConversion
    : public OpConversionPattern<BoolRelationVVOp> {
  using OpConversionPattern<BoolRelationVVOp>::OpConversionPattern;
  using OpAdaptor = typename BoolRelationVVOp::Adaptor;
  // using OpConversionPattern<Tx81Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BoolRelationVVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __BoolLessEqualVV(uint64_t *src0, uint64_t *src1,
    // uint64_t *dst, uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.NormalConvertOp op to LLVM
template <typename NormalConvertOp, const char *funcPrefix>
struct NormalConvertOpConversion : public OpConversionPattern<NormalConvertOp> {
  using OpConversionPattern<NormalConvertOp>::OpConversionPattern;
  using OpAdaptor = typename NormalConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(NormalConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __FP16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.RoundConvertOp op to LLVM
template <typename RoundConvertOp, const char *funcPrefix>
struct RoundConvertOpConversion : public OpConversionPattern<RoundConvertOp> {
  using OpConversionPattern<RoundConvertOp>::OpConversionPattern;
  using OpAdaptor = typename RoundConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(RoundConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __INT16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count, RND_MODE round);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getRndMode()));

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount, rnd_mode});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.gemm to LLVM call to __Gemm function
struct GemmOpConversion : public OpConversionPattern<tx::GemmOp> {
  using OpConversionPattern<tx::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Gemm runtime function if not already declared
    // Signature: void __Gemm(int64_t* srcA, int64_t *srcB, int64_t * srcBias,
    // int64_t *dst, int32_t *dims, bool enPsum, int64_t *psum, bool enTransA,
    // bool enTransB, int64_t batchSizeA, int64_t batchSizeB, bool enLeakyRelu,
    // bool enBias,bool enNegScale, int64_t *negScale, bool enPosScale, int64_t
    // *posScale, int64_t srcFmt, int64_t dstFmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i1Ty = rewriter.getI1Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy,  // srcA
        i8PtrTy,  // srcB
        i8PtrTy,  // srcBias
        i8PtrTy,  // dst
        i32PtrTy, // dims
        i1Ty,     // enPsum
        i8PtrTy,  // psum
        i1Ty,     // enTransA
        i1Ty,     // enTransB
        i32Ty,    // batchSizeA
        i32Ty,    // batchSizeB
        i32Ty,    // reluMode
        i1Ty,     // enBias
        i1Ty,     // enNegScale
        i8PtrTy,  // negScale
        i1Ty,     // enPosScale
        i8PtrTy,  // posScale
        i32Ty,    // srcFmt
        i32Ty     // dstFmt
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(), "__Gemm",
                                        i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getSrcA();
    Value srcB = adaptor.getSrcB();
    Value srcBias = adaptor.getSrcBias();
    Value dst = adaptor.getDst();

    Value psumAddr = adaptor.getPsumAddr();
    Value srcNegScale = adaptor.getSrcNegScale();
    Value srcPosScale = adaptor.getSrcPosScale();

    // Bitcast all pointers to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    srcBias = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcBias);
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);
    psumAddr =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, psumAddr);
    srcNegScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcNegScale);
    srcPosScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcPosScale);

    // Handle dims array - need to convert from attribute to runtime array
    auto dimsAttr = op.getDims();
    SmallVector<int32_t, 3> dimsValues;
    for (auto dimAttr : dimsAttr)
      dimsValues.push_back(mlir::cast<IntegerAttr>(dimAttr).getInt());

    // Allocate memory for the dims array
    Value dimsArraySize = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(dimsValues.size()));

    // Use alloc to allocate memory for dims array
    auto dimsArrayI32Ptr = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), i32PtrTy, rewriter.getI32Type(), dimsArraySize,
        /*alignment=*/0);

    // Store each dimension in the array
    for (size_t i = 0; i < dimsValues.size(); i++) {
      // Create the index
      Value idx = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), i64Ty, rewriter.getI32IntegerAttr(i));

      // Create GEP to get pointer to array element
      Value elemPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), i64PtrTy, i32Ty, dimsArrayI32Ptr, ArrayRef<Value>{idx});

      // Create the dimension value
      Value dimValue = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(dimsValues[i]));

      // Store the value
      rewriter.create<LLVM::StoreOp>(op.getLoc(), dimValue, elemPtr);
    }

    // Convert boolean attributes
    Value transA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcA()));
    Value transB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcB()));
    Value enPSum = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPsum()));
    Value reluMode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getReluMode()));
    Value enBias = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnBias()));
    Value enNegScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnNegScale()));
    Value enPosScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPosScale()));

    // Convert integer attributes
    Value batchA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcA()));
    Value batchB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcB()));
    Value srcFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getSrcFmt()));
    Value dstFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDstFmt()));

    // Create the call to __Gemm
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Gemm", // funcPtr,
        ArrayRef<Value>{srcA, srcB, srcBias, dst, dimsArrayI32Ptr, enPSum,
                        psumAddr, transA, transB, batchA, batchB, reluMode,
                        enBias, enNegScale, srcNegScale, enPosScale,
                        srcPosScale, srcFmt, dstFmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.memset to LLVM call to __Memset function
struct MemsetOpConversion : public OpConversionPattern<tx::MemsetOp> {
  using OpConversionPattern<tx::MemsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::MemsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: void* __Memset(void* dst, int64_t value, uint32_t elem_count,
    //                    int32_t* strides, int32_t* iterations, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i8PtrTy, // Spm addr
        i32Ty,   // value
        i32Ty,   // shape_n/iterator_2
        i32Ty,   // shape_h/iterator_1
        i32Ty,   // shape_w/iterator_0
        i32Ty,   // shape_c/elem_count
        i32Ty,   // stride_n
        i32Ty,   // stride_h
        i32Ty,   // stride_w,
        i16Ty    // fmt
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        "__Memset", i8PtrTy, argTypes);

    // Get operands
    Value src = adaptor.getSrc();
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value value = adaptor.getValue();

    // Handle strides and iterations arrays
    ValueRange shape = adaptor.getShape();
    Value iteration2 = castIndexToInt32(rewriter, op->getLoc(), shape[0]);
    Value iteration1 = castIndexToInt32(rewriter, op->getLoc(), shape[1]);
    Value iteration0 = castIndexToInt32(rewriter, op->getLoc(), shape[2]);
    Value elemCount = castIndexToInt32(rewriter, op->getLoc(), shape[3]);

    ValueRange strides = adaptor.getStrides();
    Value stride2 = castIndexToInt32(rewriter, op->getLoc(), strides[0]);
    Value stride1 = castIndexToInt32(rewriter, op->getLoc(), strides[1]);
    Value stride0 = castIndexToInt32(rewriter, op->getLoc(), strides[2]);

    // Convert fmt attribute to Value
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Memset", // funcPtr,
        ArrayRef<Value>{src, value, elemCount, stride0, iteration0, stride1,
                        iteration1, stride2, iteration2, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Conversion pattern for linalg.fill operation with tensor arguments
struct LinalgFillOpConversion : public OpConversionPattern<linalg::FillOp> {
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LinalgFillOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpConversionPattern<linalg::FillOp>(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The operation should have tensor as output
    if (op.getOutputs().size() != 1) {
      return rewriter.notifyMatchFailure(op, "expects single output tensor");
    }

    // Check if the output is a tensor type
    Value outputTensor = op.getOutputs()[0];
    auto tensorType = mlir::dyn_cast<RankedTensorType>(outputTensor.getType());
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "expects ranked tensor type");
    }

    // Check for static shape
    if (!tensorType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "dynamic shapes not yet supported");
    }

    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    Value value = adaptor.getInputs()[0];

    // Get the element type
    Type elemType = tensorType.getElementType();

    // Convert the tensor type to the LLVM pointer type
    auto llvmPtrType = mlir::dyn_cast<LLVM::LLVMPointerType>(
        typeConverter->convertType(tensorType));
    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert tensor type to LLVM pointer type");
    }

    // Calculate total number of elements
    int64_t totalElements = 1;
    for (int64_t dim : tensorType.getShape()) {
      totalElements *= dim;
    }

    // Get index type
    auto indexType = rewriter.getI64Type();

    // Implement the following steps:
    // 1. Allocate memory for the tensor
    // 2. Fill it using memset if applicable
    // 3. Return the pointer as the result

    // Calculate element size in bytes
    int64_t elemSizeInBytes = 0;
    if (auto intType = mlir::dyn_cast<IntegerType>(elemType)) {
      elemSizeInBytes =
          (intType.getWidth() + 7) / 8; // Round up to nearest byte
    } else if (auto floatType = mlir::dyn_cast<FloatType>(elemType)) {
      elemSizeInBytes =
          (floatType.getWidth() + 7) / 8; // Round up to nearest byte
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    // Calculate total size in bytes
    auto totalSizeInBytes = totalElements * elemSizeInBytes;
    auto totalSizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, indexType, rewriter.getI64IntegerAttr(totalSizeInBytes));

    // Allocate memory
    auto mallocFunc =
        getOrInsertMalloc(rewriter, op->getParentOfType<ModuleOp>());
    auto allocated = rewriter.create<LLVM::CallOp>(
        loc, LLVM::LLVMPointerType::get(context), mallocFunc,
        ArrayRef<Value>{totalSizeVal});

    auto llvmVoidPtr = LLVM::LLVMPointerType::get(context);

    // Cast the allocated memory to the appropriate pointer type
    auto castPtr = rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType,
                                                    allocated.getResult());

    // Check if we can use memset for filling
    bool useMemset = false;
    Value byteValue;

    // For memset to work correctly, we need to have a consistent byte pattern
    if (auto constOp = value.getDefiningOp<LLVM::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue())) {
        // For integer constants
        auto intVal = intAttr.getInt();
        // Check if all bytes in the pattern are the same
        bool allBytesEqual = true;
        uint8_t firstByte = intVal & 0xFF;
        for (unsigned i = 1; i < elemSizeInBytes; i++) {
          if (((intVal >> (i * 8)) & 0xFF) != firstByte) {
            allBytesEqual = false;
            break;
          }
        }

        if (allBytesEqual) {
          useMemset = true;
          byteValue = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getIntegerType(8),
              rewriter.getIntegerAttr(rewriter.getIntegerType(8), firstByte));
        }
      } else if (auto floatAttr =
                     mlir::dyn_cast<FloatAttr>(constOp.getValue())) {
        // For floating point constants
        if (floatAttr.getValue().isZero()) {
          // Zero float can use memset with zero byte value
          useMemset = true;
          byteValue = rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getIntegerType(8), rewriter.getI8IntegerAttr(0));
        }
      }
    }

    if (useMemset) {
      // Use memset for filling
      auto memsetFunc =
          getOrInsertMemset(rewriter, op->getParentOfType<ModuleOp>());
      rewriter.create<LLVM::CallOp>(
          loc, llvmVoidPtr, memsetFunc,
          ArrayRef<Value>{castPtr, byteValue, totalSizeVal});
    } else {
      // Create a loop to manually fill the tensor with the value
      // We'll use SCF dialect for structured loops
      auto llvmElemType = typeConverter->convertType(elemType);

      // Create loop initialization
      auto zero = rewriter.create<LLVM::ConstantOp>(
          loc, indexType, rewriter.getI64IntegerAttr(0));
      auto upperBound = rewriter.create<LLVM::ConstantOp>(
          loc, indexType, rewriter.getI64IntegerAttr(totalElements));
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, indexType, rewriter.getI64IntegerAttr(1));

      // Create the fill loop
      auto loopOp =
          rewriter.create<scf::ForOp>(loc, zero, upperBound, one, ValueRange{});

      // Set insertion point inside the loop
      rewriter.setInsertionPointToStart(loopOp.getBody());

      // Calculate pointer for the current element
      auto currentPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(context),
          LLVM::LLVMPointerType::get(context), castPtr,
          ArrayRef<Value>({loopOp.getInductionVar()}));

      // Store the fill value to the current memory location
      rewriter.create<LLVM::StoreOp>(loc, value, currentPtr);

      // Reset insertion point after the loop
      rewriter.setInsertionPointAfter(loopOp);
    }

    // Replace the original op with the casted pointer
    rewriter.replaceOp(op, castPtr);
    return success();
  }

private:
  // Helper to get or insert malloc function declaration
  FlatSymbolRefAttr getOrInsertMalloc(PatternRewriter &rewriter,
                                      ModuleOp module) const {
    auto context = rewriter.getContext();
    auto mallocName = "malloc";
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(mallocName)) {
      return SymbolRefAttr::get(rewriter.getContext(), mallocName);
    }

    // Create malloc function declaration
    auto llvmVoidPtr = LLVM::LLVMPointerType::get(context);
    auto mallocType =
        LLVM::LLVMFunctionType::get(llvmVoidPtr, {rewriter.getI64Type()},
                                    /*isVarArg=*/false);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), mallocName, mallocType);

    return SymbolRefAttr::get(rewriter.getContext(), mallocName);
  }

  // Helper to get or insert memset function declaration
  FlatSymbolRefAttr getOrInsertMemset(PatternRewriter &rewriter,
                                      ModuleOp module) const {
    auto context = rewriter.getContext();
    auto memsetName = "memset";
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(memsetName)) {
      return SymbolRefAttr::get(rewriter.getContext(), memsetName);
    }

    // Create memset function declaration
    auto voidPtrType = LLVM::LLVMPointerType::get(context);
    auto memsetType = LLVM::LLVMFunctionType::get(
        context,
        voidPtrType, // memset returns the destination pointer
        ArrayRef<Type>{voidPtrType, rewriter.getI8Type(),
                       rewriter.getI64Type()},
        /*isVarArg=*/false);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), memsetName, memsetType);

    return SymbolRefAttr::get(rewriter.getContext(), memsetName);
  }
};

// Conversion pattern for tensor.empty operation
class TensorEmptyOpConversion : public OpConversionPattern<tensor::EmptyOp> {
public:
  using OpConversionPattern<tensor::EmptyOp>::OpConversionPattern;

  TensorEmptyOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<tensor::EmptyOp>(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the result tensor type
    TensorType resultType = op.getType();

    // Verify we can handle this tensor type
    if (!resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "dynamic shapes not yet supported");
    }

    // Convert the tensor type to LLVM pointer type
    auto llvmPtrType = mlir::dyn_cast<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(resultType));

    if (!llvmPtrType) {
      return rewriter.notifyMatchFailure(
          op, "failed to convert tensor type to LLVM pointer type");
    }

    // Get element type
    Type elementType = resultType.getElementType();

    // Create LLVM operations to allocate memory
    // 1. Calculate the total allocation size in bytes
    auto loc = op.getLoc();
    int64_t totalElements = 1;
    for (int64_t dim : resultType.getShape()) {
      totalElements *= dim;
    }

    auto elementSize = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(getElementTypeSize(elementType)));

    auto totalSize = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(totalElements));

    auto allocSize = rewriter.create<LLVM::MulOp>(loc, rewriter.getI64Type(),
                                                  totalSize, elementSize);

    // 2. Allocate memory using malloc
    auto mallocFunc =
        getOrInsertMalloc(rewriter, op->getParentOfType<ModuleOp>());
    auto allocated = rewriter.create<LLVM::CallOp>(loc, llvmPtrType, mallocFunc,
                                                   ArrayRef<Value>{allocSize});

    // Replace the tensor.empty operation with our allocation
    rewriter.replaceOp(op, allocated.getResult());
    return success();
  }

private:
  // Helper to get element type size in bytes
  int64_t getElementTypeSize(Type type) const {
    if (auto floatType = mlir::dyn_cast<FloatType>(type)) {
      return floatType.getWidth() / 8;
    } else if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
      return intType.getWidth() / 8;
    }
    // Default for other types
    return 1;
  }

  // Helper to get or insert malloc function declaration
  FlatSymbolRefAttr getOrInsertMalloc(PatternRewriter &rewriter,
                                      ModuleOp module) const {
    auto mallocName = "malloc";
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(mallocName)) {
      return SymbolRefAttr::get(rewriter.getContext(), mallocName);
    }

    // Create malloc function declaration
    auto llvmVoidPtr = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto mallocType =
        LLVM::LLVMFunctionType::get(llvmVoidPtr, {rewriter.getI64Type()},
                                    /*isVarArg=*/false);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), mallocName, mallocType);

    return SymbolRefAttr::get(rewriter.getContext(), mallocName);
  }
};

// Convert tt.get_program_id to LLVM call to __get_pid function
// Think this as Tx81 special action. May can separate to a single pass or use
// tx81.get_program_id op
struct GetProgramIDConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      mlir::triton::getMaxEnumValForProgramIDDim() + 1;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: uint32_t __get_pid(uint32_t);
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i32Ty, // x: 0/y: 1/z: 2,
    };

    // Declare the function
    Value funcPtr = declareTx81Function(module, rewriter, op.getLoc(),
                                        "__get_pid", i32Ty, argTypes);

    // Get operands
    auto axis = (uint32_t)op.getAxis();

    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    // Convert fmt attribute to Value
    Value src = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(axis));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), i32Ty,
                                              "__get_pid", // funcPtr,
                                              ArrayRef<Value>{src});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// The conversion pass
class Tx81ToLLVMPass : public Tx81ToLLVMBase<Tx81ToLLVMPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, tx::Tx81Dialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Setup LLVM lowering options object which should live across the call to
    // applyFull/PartialConversion.
    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = false;

    // Setup conversion target
    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           func::FuncDialect, math::MathDialect>();
    // Handle the tx81 op to llvm.call and support kcore load/store op's spm
    // offset
    target.addIllegalDialect<triton::TritonDialect, linalg::LinalgDialect,
                             tensor::TensorDialect, affine::AffineDialect,
                             tx::Tx81Dialect>();

    // Setup rewrite patterns
    RewritePatternSet patterns(context);

    // NOTE: LLVMTypeConverter should be enough for MLIR core dialects.
    LLVMTypeConverter llvmTypeConverter(context, options);

    // Add the Tx81 to LLVM conversion patterns
    // clang-format off
    patterns.add</*ConstantOpConversion, IndexCastOpConversion,
                 AddIOpConversion, MulIOpConversion,*/
                 NormalConvertOpConversion<tx::FP16ToFP32Op, fp16ToFp32FuncName>,
                 NormalConvertOpConversion<tx::INT16ToFP16Op, int16ToFp16FuncName>,
                 RoundConvertOpConversion<tx::FP16ToINT8Op,fp16ToInt8FuncName>,
                 RoundConvertOpConversion<tx::FP16ToINT16Op,fp16ToInt16FuncName>,
                 RoundConvertOpConversion<tx::FP16ToINT32Op,fp16ToInt32FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT8Op,fp16ToInt8FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT16Op,fp32ToInt16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT32Op,fp32ToInt32FuncName>,
                 RoundConvertOpConversion<tx::INT16ToFP32Op,int16ToFp32FuncName>,
                 RoundConvertOpConversion<tx::INT32ToFP32Op,int32ToFp32FuncName>,
                 RoundConvertOpConversion<tx::INT32ToFP16Op,int32ToFp16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT32Op,fp32ToInt32FuncName>,
                 RoundConvertOpConversion<tx::FP32ToFP16Op, fp32ToFp16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToBF16Op,fp32ToBf16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToTF32Op, fp32ToTF32FuncName>,
                 ReduceOpConversion<tx::ReduceSumOp,reduceSumFuncName>,
                 ReduceOpConversion<tx::ReduceMaxOp,reduceMaxFuncName>,
                 ReduceOpConversion<tx::ReduceMinOp,reduceMinFuncName>,
                 ElementWiseOpConversion<tx::AddVVOp, addVVFuncName>,
                 ElementWiseOpConversion<tx::SubVVOp, subVVFuncName>,
                 ElementWiseOpConversion<tx::MulVVOp, mulVVFuncName>,
                 ElementWiseOpConversion<tx::DivVVOp, divVVFuncName>,
                 ElementWiseOpConversion<tx::MaxVVOp, MaxVVFuncName>,
                 ElementWiseOpConversion<tx::MinVVOp, MinVVFuncName>,
                 UnaryOpConversion<tx::AbsVVOp, absVVFuncName>,
                 UnaryOpConversion<tx::RsqrtVVOp, rsqrtVVFuncName>,
                 UnaryOpConversion<tx::SqrtVVOp, sqrtVVFuncName>,
                 UnaryOpConversion<tx::LnOp, lnFuncName>,
                 UnaryOpConversion<tx::Log2Op, log2FuncName>,
                 UnaryOpConversion<tx::ExpOp, expFuncName>,
                 UnaryOpConversion<tx::Pow2Op, pow2FuncName>,
                 UnaryOpConversion<tx::SinOp, sinFuncName>,
                 UnaryOpConversion<tx::CosOp, cosFuncName>,
                 BinaryVSOpConversion<tx::AddVSOp, addVSFuncName>,
                 BinaryVSOpConversion<tx::SubVSOp, subVSFuncName>,
                 BinaryVSOpConversion<tx::MulVSOp, mulVSFuncName>,
                 BinaryVSOpConversion<tx::DivVSOp, divVSFuncName>,
                 BoolRelationVVOpConversion<tx::BoolEqualVV, boolEqualVVFuncName>,
                 BoolRelationVVOpConversion<tx::BoolUnEqualVV, boolUnEqualVVFuncName>,
                 BoolRelationVVOpConversion<tx::BoolGreaterEqualVV, boolGreaterEqualVVFuncName>,
                 BoolRelationVVOpConversion<tx::BoolGreaterVV, boolGreaterVVFuncName>,
                 BoolRelationVVOpConversion<tx::BoolLessEqualVV, boolLessEqualVVFuncName>,
                 BoolRelationVVOpConversion<tx::BoolLessThenVV, boolLessVVFuncName>,
                 BinaryLogicVVOpConversion<tx::AndVV, andVVFuncName>,
                 BinaryLogicVVOpConversion<tx::OrVV, orVVFuncName>,
                 BinaryLogicVVOpConversion<tx::XorVV, xorVVFuncName>,
                 RdmaOpConversion,
                 WdmaOpConversion,
                 MaskMoveOpConversion,
                 GemmOpConversion,
                 MemsetOpConversion,
                 GetProgramIDConversion>(
        context);
    // clang-format on

    // Add call op conversion
    populateCallOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Add return op conversion
    populateReturnOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTx81ToLLVMPass() {
  return std::make_unique<Tx81ToLLVMPass>();
}
