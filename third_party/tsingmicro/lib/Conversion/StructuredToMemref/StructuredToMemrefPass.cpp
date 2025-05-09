//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/OneToNTypeConversion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <optional>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREF
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static MemRefType getMemrefTypeForScalarPtr(triton::PointerType ptrType,
                                            MLIRContext *context) {
  SmallVector<int64_t> strides{1};
  auto layout = StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

  auto elemType = ptrType.getPointeeType();
  auto memrefType = MemRefType::get({1}, elemType, layout);
  return memrefType;
}

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    // Used for converting memref<*> back to tt.ptr type, these ops will then be
    // handled when we convert addptr op later.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs,
                                   Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return getMemrefTypeForScalarPtr(ptrType, context);
    });

    // A tensor of pointers can be passed in as scf.for's init-args, in such
    // cases, we convert the type to a memref with dynamic offsets and
    // strides.
    addConversion(
        [context](RankedTensorType tensorType) -> std::optional<MemRefType> {
          if (auto ptrType = llvm::dyn_cast<triton::PointerType>(
                  tensorType.getElementType())) {
            auto layout = StridedLayoutAttr::get(
                context, ShapedType::kDynamic,
                SmallVector<int64_t>(tensorType.getRank(),
                                     ShapedType::kDynamic));
            Type elemType = ptrType.getPointeeType();
            return MemRefType::get(tensorType.getShape(), elemType, layout);
          }

          return std::nullopt;
        });

    // Convert the current memref type to a memref type with dynamic offsets and
    // strides through another reinterpret_cast with the same offsets.
    // Canonicalization will simplify this sequence by removing the inital
    // reinterpret_cast.
    addTargetMaterialization([&](OpBuilder &builder, MemRefType memrefType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      auto reinterpretCast =
          inputs[0].getDefiningOp<memref::ReinterpretCastOp>();
      return builder.create<memref::ReinterpretCastOp>(
          loc, memrefType, inputs[0], reinterpretCast.getMixedOffsets()[0],
          reinterpretCast.getMixedSizes(), reinterpretCast.getMixedStrides());
    });
  }
};

struct ScalarAddptrConverter
    : public OneToNOpConversionPattern<triton::AddPtrOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }

    auto loc = op->getLoc();

    auto offsetIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getOffset());

    auto ptrInfo = adaptor.getPtr();
    assert(ptrInfo.size() == 2);
    auto ptr = ptrInfo[0];
    auto offset = ptrInfo[1];

    auto newOffset = rewriter.create<arith::AddIOp>(loc, offset, offsetIndex);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(op.getPtr().getType()),
            rewriter.getContext()),
        ptr, getAsOpFoldResult(newOffset) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    rewriter.replaceOp(op, SmallVector<Value>{castOp.getResult(), newOffset},
                       adaptor.getResultMapping());

    return success();
  }
};

static std::optional<SmallVector<Value>>
buildCastAndOffsetOps(OpBuilder &builder, TypeRange resultTypes, Value input,
                      Location loc) {
  assert(resultTypes.size() == 2 && isa<MemRefType>(resultTypes[0]) &&
         isa<IndexType>(resultTypes[1]) &&
         "Unexpected result types when converting addptr");
  assert(isa<triton::PointerType>(input.getType()) &&
         "Unexpected input type when converting addptr");

  // There are only two types of ops that can produce a result of type tt.ptr
  // 1) tt.addptr, this is already handled by ScalarAddptrConverter
  // 2) unrealized_conversion_cast, which are inserted during the conversion
  //    of function arguments.
  //    We assert that there can only be input that comes from
  //    unrealized_conversion_cast.
  auto castOp = input.getDefiningOp<UnrealizedConversionCastOp>();
  assert(castOp && "Unexpected defining op for input of type tt.ptr");

  // Compute the memref type
  auto buffer = castOp.getOperand(0);
  auto bufferType = cast<UnrankedMemRefType>(buffer.getType());
  auto layout =
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, {1});
  auto memrefType = MemRefType::get({1}, bufferType.getElementType(), layout);

  // Create ops to convert the triton input type to a pair of {memref, index}
  auto cast = builder.create<memref::ReinterpretCastOp>(
      loc, memrefType, buffer, 0 /*offset*/, ArrayRef<int64_t>{(1)} /*sizes*/,
      ArrayRef<int64_t>{(1)} /*strides*/);
  auto zero = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));

  return SmallVector<Value>{cast, zero};
}

static Value buildCastOp(OpBuilder &builder, Type resultType,
                         ValueRange inputs, Location loc) {
  assert(isa<triton::PointerType>(resultType));
  assert(inputs.size() && isa<MemRefType>(inputs[0].getType()) &&
         isa<IndexType>(inputs[1].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
      .getResult(0);
}

class StructuredToMemrefPass
    : public triton::impl::StructuredToMemrefBase<StructuredToMemrefPass> {
  using StructuredToMemrefBase<StructuredToMemrefPass>::StructuredToMemrefBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  LogicalResult convertArgsToMemrefType() {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;

    // Update function signatures and calls to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) && typeConverter.isLegal(op.getOperandTypes());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::CallOp>(
        patterns, typeConverter);

    return applyPartialConversion(moduleOp, target, std::move(patterns));
  }

  // We leverage the 1->N conversion infrastructure to convert tt.addptr for
  // scalar to memref.reinterpret_cast.
  //
  // A tt.addptr has the following form:
  //
  // %new_ptr = tt.addptr %ptr %offset
  //
  // where %new_ptr and %ptr have tt.ptr type, and %offset is of index type.
  //
  // With this form, there can be a chain of tt.addptr where we keep adding
  // offsets to an existing pointer:
  //
  // %ptr_1 = tt.addptr %arg0 %offset
  // %ptr_2 = tt.addptr %ptr_1 %offset
  // %ptr_3 = tt.addptr %ptr_2 %offset
  //
  // Now, we want to lower each tt.addptr to a memref.reinterpret_cast so that
  // the pointers can be used by affine.load and affine.store (lowered from
  // tt.load and tt.store).
  //
  // A memref.reinterpret_cast op also takes an offset and returns a memref in a
  // similar fashion to tt.addptr:
  //
  //  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes:
  //  [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset:
  //  ?>>
  //
  // However, since the semantic of memref.reinterpret_cast is different,
  // the following lowering would be incorrect for the sequence of tt.addptr
  // above:
  //
  //  %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_2 = memref.reinterpret_cast %cast_1 to offset [%offset]
  //  %cast_3 = memref.reinterpret_cast %cast_2 to offset [%offset]
  //
  // The above sequence is equivalent to:
  //
  //  %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_2 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_3 = memref.reinterpret_cast %arg0 to offset [%offset]
  //
  // In other word, memref.reinterpret_cast ignores the current offset of the
  // input buffer.
  //
  // Therefore, we have to manually track the offset for each addptr by lowering
  // to the following form:
  //
  // %offset_1 = arith.addi %cst_0 %offset
  // %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset_1]
  //
  // %offset_2 = arith.addi %offset_1 %offset
  // %cast_2 = memref.reinterpret_cast %arg0 to offset [%offset_2]
  //
  // %offset_3 = arith.addi %offset_2 %offset
  // %cast_3 = memref.reinterpret_cast %arg0 to offset [%offset_3]
  //
  // Each tt.addptr is lowered to a pair of arith.addi that accumulates the
  // current offset before using that offset to the reinterpret_cast.
  LogicalResult convertAddPtrToReinterpretCast() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->2 type conversion here, where a triton pointer type
    // maps to a pair of {memref, index} type for the the buffer and offset.
    converter.addConversion(
        [context](triton::PointerType ptrType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          types = SmallVector<Type>{getMemrefTypeForScalarPtr(ptrType, context),
                                    IndexType::get(context)};
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a pair of {memref,
    // index} type back to the original triton pointer type.
    // These are used when there are ops that still need to use the original
    // pointer type. For instance, we convert the result of tt.addptr from
    // tt.ptr type to a pair of {memref, index}, but the original ptr result is
    // still being used by another tt.load or tt.store.
    converter.addArgumentMaterialization(buildCastOp);
    converter.addSourceMaterialization(buildCastOp);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a pair of {memref, index} type.
#if 0  // FIXME: Incompatible MILR interface
    converter.addTargetMaterialization(buildCastAndOffsetOps);
#endif

    patterns.add<ScalarAddptrConverter>(converter, context);

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      return failure();
    }

    return success();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    if (failed(convertArgsToMemrefType())) {
      signalPassFailure();
      return;
    }

    if (failed(convertAddPtrToReinterpretCast())) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalDialect<tts::TritonStructuredDialect>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](Operation *op) {
      auto resType = op->getResultTypes()[0];
      return !isa<triton::PointerType>(resType);
    });

    LoopTypeConverter loopTypeConverter(patterns.getContext());

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        loopTypeConverter, patterns, target);

    triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                         loopTypeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
