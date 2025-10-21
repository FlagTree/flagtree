#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace arith;

namespace mlir {

namespace aipu {

#define GEN_PASS_DEF_CONVERTI64TOI32
#include "Passes/Passes.h.inc"

LogicalResult checkConstInt(Value value, long val) {
  if (auto constOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      if (intAttr.getValue() == val) {
        return success();
      }
    }
  }
  return failure();
}

LogicalResult checkExtI32ToI64(Value value) {
  if (auto op = value.getDefiningOp<ExtUIOp>()) {
    if (op.getResult().getType() == IntegerType::get(op.getContext(), 64) &&
        op.getOperand().getType() == IntegerType::get(op.getContext(), 32)) {
      return success();
    }
  }
  if (auto op = value.getDefiningOp<ExtSIOp>()) {
    if (op.getResult().getType() == IntegerType::get(op.getContext(), 64) &&
        op.getOperand().getType() == IntegerType::get(op.getContext(), 32)) {
      return success();
    }
  }
  return failure();
}

struct ReplaceJoinI32ToI64 : public OpRewritePattern<OrIOp> {
  using OpRewritePattern<OrIOp>::OpRewritePattern;

  // We're looking for the following sequence of MLIR ops:
  //
  //   %28 = arith.extui %26 : i32 to i64
  //   %29 = arith.shli %28, %c32_i64 : i64
  //   %30 = arith.extui %27 : i32 to i64
  //   %31 = arith.ori %29, %30 : i64
  //
  // This pattern is generated from the following high-level Triton code:
  //
  //   def join_i32_to_i64(hi: tl.uint32, lo: tl.uint32):
  //       hi = hi.to(tl.uint64) << 32
  //       x = hi | lo.to(tl.uint64)
  //       return x
  //
  // match and rewrite with the origin low value

  LogicalResult matchAndRewrite(OrIOp orOp,
                                PatternRewriter &rewriter) const override {
    if (orOp.getResult().getType() != IntegerType::get(orOp.getContext(), 64)) {
      return failure();
    }
    auto or_lhs = orOp.getRhs();
    if (failed(checkExtI32ToI64(or_lhs))) {
      return failure();
    }
    Value hi_i64 = orOp.getLhs();
    auto shliOp = hi_i64.getDefiningOp<ShLIOp>();
    if (!shliOp)
      return failure();
    if (failed(checkConstInt(shliOp.getRhs(), 32))) {
      return failure();
    }
    if (failed(checkExtI32ToI64(shliOp.getLhs()))) {
      return failure();
    }

    auto loc = orOp.getLoc();
    auto extuiOp = or_lhs.getDefiningOp<ExtUIOp>();
    auto origin_lo = extuiOp.getOperand();
    rewriter.replaceOp(orOp, origin_lo);
    return success();
  }
};

struct ReplaceSplitI64ToI32 : public OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  // We're looking for the following sequence of MLIR ops:
  //
  //   %c32_i64 = arith.constant 32 : i64
  //   %c4294967295_i64 = arith.constant 4294967295 : i64
  //   %1 = arith.extsi %arg0 : i32 to i64
  //   -------------------------------------------------|
  //   `matchLow`                                       |
  //    %3 = arith.andi %1, %c4294967295_i64 : i64      |  1.
  //   -------------------------------------------------|
  //   `matchHigh`                                      |
  //    %2 = arith.shrsi %1, %c32_i64 : i64             |
  //    %3 = arith.andi %2, %c4294967295_i64 : i64      |  2.
  //    ------------------------------------------------|
  //    %4 = arith.trunci %3 : i64 to i32
  //
  // This pattern corresponds to the following high-level Triton code:
  //
  //   def split_i64_to_i32(x: i64):
  //      lo = (x & 0xFFFFFFFF).to(tl.uint32)
  //      hi = ((x >> 32) & 0xFFFFFFFF).to(tl.uint32)
  //
  // match and rewrite : low with origin_x, high with zero
  LogicalResult matchAndRewrite(TruncIOp truncOp,
                                PatternRewriter &rewriter) const override {
    if (truncOp.getResult().getType() !=
            IntegerType::get(truncOp.getContext(), 32) ||
        truncOp.getOperand().getType() !=
            IntegerType::get(truncOp.getContext(), 64)) {
      return failure();
    }
    Value src = truncOp.getOperand();

    auto andiOp = src.getDefiningOp<AndIOp>();
    if (!andiOp)
      return failure();

    if (failed(checkConstInt(andiOp.getRhs(), 0xFFFFFFFF))) {
      return failure();
    }
    Value andLhs = andiOp.getLhs();
    // match low
    if (succeeded(checkExtI32ToI64(andLhs))) {
      // rewrite
      auto extsiOp = andLhs.getDefiningOp<ExtSIOp>();
      auto origin_x = extsiOp.getOperand();
      rewriter.replaceOp(truncOp, origin_x);
      return success();
    }
    // match high
    auto shrsiOp = andLhs.getDefiningOp<ShRSIOp>();
    auto shruiOp = andLhs.getDefiningOp<ShRUIOp>();
    if (!(shrsiOp || shruiOp))
      return failure();
    if (shrsiOp) {
      if (failed(checkConstInt(shrsiOp.getRhs(), 32))) {
        return failure();
      }
    }
    if (shruiOp) {
      if (failed(checkConstInt(shruiOp.getRhs(), 32))) {
        return failure();
      }
    }
    // rewrite
    Value zero = rewriter.create<ConstantIntOp>(truncOp.getLoc(), 0, 32);
    rewriter.replaceOp(truncOp, zero);
    return success();
  }
};

class I64ToI32TypeConverter : public TypeConverter {
public:
  I64ToI32TypeConverter(MLIRContext *ctx) {
    addConversion([ctx](Type type) -> Type { return type; });
    addConversion([ctx](mlir::IntegerType type) -> Type {
      if (type.getWidth() == 64) {
        return mlir::IntegerType::get(type.getContext(), 32);
      }
      return type;
    });
    addConversion([ctx, this](MemRefType type) -> Type {
      Type oldElemTy = type.getElementType();
      Type newElemTy = this->convertType(oldElemTy);

      if (newElemTy != oldElemTy) {
        return MemRefType::get(type.getShape(), newElemTy, type.getLayout(),
                               type.getMemorySpace());
      }
      return type;
    });
    addConversion([this](UnrankedMemRefType type) -> Type {
      Type oldElemTy = type.getElementType();
      Type newElemTy = this->convertType(oldElemTy);

      if (newElemTy != oldElemTy) {
        return UnrankedMemRefType::get(newElemTy, type.getMemorySpace());
      }
      return type;
    });
    addConversion([ctx, this](VectorType type) -> Type {
      Type oldElemTy = type.getElementType();

      Type newElemTy = this->convertType(oldElemTy);

      if (newElemTy != oldElemTy) {

        return VectorType::get(type.getShape(), newElemTy);
      }

      return type;
    });
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addSourceMaterialization([&](OpBuilder &builder, VectorType resultType,
                                 ValueRange inputs, Location loc) {
      return builder.create<mlir::arith::ExtSIOp>(loc, resultType, inputs[0]);
    });

    addTargetMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::VectorType resultType,
                                 mlir::ValueRange inputs, mlir::Location loc) {
      return builder.create<mlir::arith::TruncIOp>(loc, resultType, inputs[0]);
    });
    addSourceMaterialization([&](OpBuilder &builder, IntegerType resultType,
                                 ValueRange inputs, Location loc) {
      return builder.create<mlir::arith::ExtSIOp>(loc, resultType, inputs[0]);
    });

    addTargetMaterialization([&](mlir::OpBuilder &builder,
                                 mlir::IntegerType resultType,
                                 mlir::ValueRange inputs, mlir::Location loc) {
      return builder.create<mlir::arith::TruncIOp>(loc, resultType, inputs[0]);
    });
  }
};

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto valueAttr = op.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      if (intAttr.getType().isInteger(64)) {
        auto newValue = mlir::IntegerAttr::get(rewriter.getI32Type(),
                                               intAttr.getValue().trunc(32));
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newValue);
        return mlir::success();
      }
    } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
      Type oldType = denseAttr.getType();
      if (auto shapedType = dyn_cast<VectorType>(oldType)) {
        Type elementType = shapedType.getElementType();
        if (elementType.isInteger(64)) {
          auto newType =
              VectorType::get(shapedType.getShape(), rewriter.getI32Type());

          auto mappedAttr = denseAttr.mapValues(
              rewriter.getI32Type(),
              [&](const APInt &value) { return value.trunc(32); });

          rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, mappedAttr);
          return success();
        }
      }
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, valueAttr);
    return mlir::success();
  }
};

struct ScfIfOpConversion : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, scf::IfOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getResultTypes().empty()) {
      return failure();
    }
    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes()) {
      if (resultType.isInteger(64)) {
        resultTypes.push_back(IntegerType::get(op.getContext(), 32));
      } else {
        resultTypes.push_back(resultType);
      }
    }

    bool hasElseRegion = !op.getElseRegion().empty();
    // In case the scf.if produces results, the “else” region must also have
    // exactly 1 block.
    if (!hasElseRegion) {
      return failure();
    }

    auto ifOp = rewriter.create<scf::IfOp>(
        op.getLoc(), resultTypes, adaptor.getCondition(), hasElseRegion);
    rewriter.inlineRegionBefore(op.getThenRegion(),
                                &ifOp.getThenRegion().back());
    rewriter.eraseBlock(&ifOp.getThenRegion().back());

    rewriter.inlineRegionBefore(op.getElseRegion(),
                                &ifOp.getElseRegion().back());
    rewriter.eraseBlock(&ifOp.getElseRegion().back());

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

template <typename OpTy>
struct GenericConvert : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type, 4> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    auto newOp = rewriter.create<OpTy>(op->getLoc(), resultTypes,
                                       adaptor.getOperands(), op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertI64ToI32 : impl::ConvertI64ToI32Base<ConvertI64ToI32> {
  using ConvertI64ToI32Base::ConvertI64ToI32Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet low_high_match_patterns(context);
    low_high_match_patterns.add<ReplaceJoinI32ToI64, ReplaceSplitI64ToI32>(
        context);
    (void)applyPatternsGreedily(module, std::move(low_high_match_patterns));

    I64ToI32TypeConverter converter(context);
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    target.addLegalOp<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                      arith::IndexCastOp>();
    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) { return converter.isLegal(op); });
    patterns.add<
        // const
        ConstantOpConversion,
        // arith
        GenericConvert<arith::AddIOp>, GenericConvert<arith::AddUIExtendedOp>,
        GenericConvert<arith::AndIOp>, GenericConvert<arith::CmpIOp>,
        GenericConvert<arith::DivSIOp>, GenericConvert<arith::DivUIOp>,
        GenericConvert<arith::FloorDivSIOp>, GenericConvert<arith::MaxSIOp>,
        GenericConvert<arith::MaxUIOp>, GenericConvert<arith::MinSIOp>,
        GenericConvert<arith::MinUIOp>, GenericConvert<arith::MulIOp>,
        GenericConvert<arith::MulSIExtendedOp>, GenericConvert<arith::SIToFPOp>,
        GenericConvert<arith::UIToFPOp>, GenericConvert<arith::FPToSIOp>,
        GenericConvert<arith::FPToUIOp>, GenericConvert<arith::MulUIExtendedOp>,
        GenericConvert<arith::OrIOp>, GenericConvert<arith::RemSIOp>,
        GenericConvert<arith::RemUIOp>, GenericConvert<arith::SelectOp>,
        GenericConvert<arith::ShLIOp>, GenericConvert<arith::ShRSIOp>,
        GenericConvert<arith::ShRUIOp>, GenericConvert<arith::SubIOp>,
        GenericConvert<arith::XOrIOp>,
        // memref
        GenericConvert<memref::CopyOp>, GenericConvert<memref::LoadOp>,
        GenericConvert<memref::AllocOp>, GenericConvert<memref::AllocaOp>,
        GenericConvert<memref::CastOp>, GenericConvert<memref::CollapseShapeOp>,
        GenericConvert<memref::DmaStartOp>, GenericConvert<memref::DmaWaitOp>,
        GenericConvert<memref::ExpandShapeOp>,
        GenericConvert<memref::SubViewOp>, GenericConvert<memref::ReshapeOp>,
        GenericConvert<memref::StoreOp>, GenericConvert<memref::TransposeOp>,
        GenericConvert<memref::ViewOp>,
        GenericConvert<memref::ReinterpretCastOp>,
        // vector
        GenericConvert<vector::BroadcastOp>,
        GenericConvert<vector::TransferReadOp>,
        GenericConvert<vector::TransferWriteOp>,
        // scf
        ScfIfOpConversion, GenericConvert<scf::YieldOp>>(converter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace aipu

} // namespace mlir
