//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "../../include/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <limits>

#define DEBUG_TYPE "TritonNPU-Utils"

namespace mlir {

namespace ConverterUtils {

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(order);
  SmallVector<int64_t> originalShape(sourceType.getShape());
  SmallVector<int64_t> transposedShape(sourceRank);
  for (size_t i = 0; i < sourceRank; i++) {
    transposedShape[i] = originalShape[perm[i]];
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;
  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = mlir::TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

memref::SubViewOp makeSubViewOp(Value src,
                                llvm::SmallVectorImpl<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<memref::SubViewOp>(loc, dyn_cast<MemRefType>(dstType),
                                            src, offsets, sizes, strides);
}

void getShapeInfo(Value val, llvm::SmallVectorImpl<OpFoldResult> &shapes,
                  ConversionPatternRewriter &rewriter) {
  if (isa<BlockArgument>(val)) {
    auto blockArg = dyn_cast<BlockArgument>(val);
    auto blockOp = blockArg.getOwner()->getParentOp();
    if (isa<scf::ForOp>(blockOp)) {
      auto forOp = dyn_cast<scf::ForOp>(blockOp);
      auto operand = forOp.getTiedLoopInit(blockArg)->get();
      getShapeInfo(operand, shapes, rewriter);
    } else {
      emitError(val.getLoc())
          << "getShapeInfo() only support ReinterpretCastOp "
             "and scf.for's block argument, but got : "
          << val << "\n";
    }
    return;
  }

  if (isa<triton::PointerType>(val.getType())) {
    val = rewriter.getRemappedValue(val);
  }

  if (!isa<memref::ReinterpretCastOp>(val.getDefiningOp())) {
    emitError(val.getLoc()) << "getShapeInfo() only support ReinterpretCastOp "
                               "and scf.for's block argument, but got : "
                            << val << "\n";
    return;
  }
  auto castOp = dyn_cast<memref::ReinterpretCastOp>(val.getDefiningOp());
  auto tensorPtrAttr = castOp->getAttr("tensor_ptr_attr");
  if (tensorPtrAttr) {
    shapes = castOp.getConstifiedMixedSizes();
  } else {
    getShapeInfo(castOp.getSource(), shapes, rewriter);
  }
  return;
}

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 Value adaptorPtr, const Location &loc,
                 ConversionPatternRewriter &rewriter) {
  SmallVector<OpFoldResult> parTensorShapes;
  getShapeInfo(adaptorPtr, parTensorShapes, rewriter);
  auto extractOp =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, adaptorPtr);

  OpFoldResult baseOffset = extractOp.getConstifiedMixedOffset();
  SmallVector<OpFoldResult> strides = extractOp.getConstifiedMixedStrides();

  SmallVector<OpFoldResult> boundarySizes = extractOp.getConstifiedMixedSizes();
  auto dims = boundarySizes.size();
  OpFoldResult currentStride = rewriter.getIndexAttr(1);
  for (int i = dims - 1; i >= 0; i--) {
    auto offset = divOpFoldResult(baseOffset, currentStride, loc, rewriter);
    offset = remOpFoldResult(offset, parTensorShapes[i], loc, rewriter);
    if (llvm::find(boundaryCheck, i) != boundaryCheck.end()) {
      OpFoldResult subOfr =
          subOpFoldResult(parTensorShapes[i], offset, loc, rewriter);
      boundarySizes[i] =
          minOpFoldResult(boundarySizes[i], subOfr, loc, rewriter);
    }
    currentStride =
        mulOpFoldResult(currentStride, parTensorShapes[i], loc, rewriter);
  }
  return boundarySizes;
}

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1 &&
             "Size of source broadcast dimension must be 1");
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "Cannot identify broadcast dimension");
  return broadcastDims;
}

// Dimensions of collapesd tensor is all unbroadcast dims
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) {
  SmallVector<int64_t> unbroadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] == srcShape[i]) {
      unbroadcastDims.emplace_back(srcShape[i]);
    }
  }
  return unbroadcastDims;
}

} // namespace ConverterUtils

namespace triton {

mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn) {
  LLVM_DEBUG(llvm::dbgs() << "[findFirstMatchingOperandDef] Current op: "
                          << *rootOp << "\n");
  mlir::Value lhs = nullptr;
  mlir::Value rhs = nullptr;
  if (auto op = dyn_cast<triton::AddPtrOp>(rootOp)) {
    lhs = op.getPtr();
    rhs = op.getOffset();
  } else if (auto op = dyn_cast<arith::AddIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::SubIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::MulIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::DivSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::RemSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<triton::SplatOp>(rootOp)) {
    lhs = op.getSrc();
  } else if (auto op = dyn_cast<triton::MakeRangeOp>(rootOp)) {
  } else {
    rootOp->emitRemark("Backtracing encounters unsupported Operation");
    return nullptr;
  }
  // Backtrace operands
  if (!lhs) {
    return nullptr;
  }
  auto lhsDef = lhs.getDefiningOp();
  mlir::Operation *targetOp;
  if (lhsDef) {
    if (condFn(lhsDef)) {
      targetOp = lhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(lhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  if (!rhs) {
    return nullptr;
  }
  auto rhsDef = rhs.getDefiningOp();
  if (rhsDef) {
    if (condFn(rhsDef)) {
      targetOp = rhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(rhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  return nullptr;
}

void traverseBackwardUpdateOperandChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    OpBuilder &builder) {

  if (!op)
    return;

  if (conditionFn(op)) {
    actionFn(builder, op);
  }

  for (Value operand : op->getOperands()) {
    // TODO: handle BlockArgument
    if (Operation *defOp = operand.getDefiningOp()) {
      traverseBackwardUpdateOperandChainIf(defOp, conditionFn, actionFn,
                                           builder);
    }
  }
}

// Note: rootOp will also be processed.
void traverseBackwardUpdateOperandChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<void(OpBuilder &, Operation *)> actionFn) {

  OpBuilder builder(rootOp->getContext());

  traverseBackwardUpdateOperandChainIf(rootOp, conditionFn, actionFn, builder);
}

void traverseForwardUpdateUserChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  if (!op) {
    return;
  }

  if (stopFn(op)) {
    stopOps.insert(op);
    return;
  }

  if (conditionFn(op)) {
    actionFn(builder, op);
  }

  for (auto res : op->getResults()) {
    for (auto userOp : res.getUsers()) {
      traverseForwardUpdateUserChainIf(userOp, conditionFn, stopFn, actionFn,
                                       builder, stopOps);
    }
  }
}

// Note: rootOp will also be processed.
void traverseForwardUpdateUserChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  OpBuilder builder(rootOp->getContext());

  traverseForwardUpdateUserChainIf(rootOp, conditionFn, stopFn, actionFn,
                                   builder, stopOps);
}

bool isMetaUse(Operation *op) { return op->hasAttr("MetaUse"); }

bool isMixUse(Operation *op) { return op->hasAttr("MixUse"); }

IndirectLoadInterfaceOpType getIndirectLoadInterfaceOpType(Operation *op) {
  auto ty = IndirectLoadInterfaceOpType::Undefined;
  if (isMetaUse(op)) {
    if (isa<triton::LoadOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Load;
    } else if (isa<arith::FPToSIOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Calc;
    }
  }
  return ty;
}

bool opIsIndirectLoad(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Load;
}

bool opIsIndirectCalc(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Calc;
}

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder) {

  if (currentDim >= totalDims) {
    bodyBuilder(builder, loc, ivs, initArgs);
    return nullptr;
  }

  auto loop = builder.create<scf::ForOp>(
      loc, LBs[currentDim], UBs[currentDim], steps[currentDim], initArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange iterArgs) {
        ivs.push_back(iv);
        auto innerLoop = createNestedLoops(nestedBuilder, nestedLoc,
                                           currentDim + 1, totalDims, LBs, UBs,
                                           steps, ivs, iterArgs, bodyBuilder);
        if (innerLoop) {
          nestedBuilder.create<scf::YieldOp>(loc, innerLoop.getResults());
        }
      });

  return loop;
}

ModuleOp getModuleOpFromOperation(Operation *op) {
  Operation *parent = op;
  while (parent != nullptr && !isa<ModuleOp>(parent)) {
    parent = parent->getParentOp(); // 向上查找
  }
  return cast<ModuleOp>(parent); // 如果没找到会抛出异常
}

} // namespace triton

std::optional<int64_t> makeIntAttr(const OpFoldResult &ofr) {
  if (isa<Attribute>(ofr) && isa<IntegerAttr>(ofr.get<Attribute>()))
    return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
  return std::nullopt;
}

bool hasConstantZero(const OpFoldResult &ofr) {
  auto intAttr = makeIntAttr(ofr);
  if (intAttr.has_value())
    return !intAttr.value();

  auto val = dyn_cast<Value>(ofr);
  assert(val && "Provided ofr must can be cast to Value");

  auto ConstOp = val.getDefiningOp<arith::ConstantOp>();
  if (!ConstOp)
    return false;

  intAttr = makeIntAttr(ConstOp.getValue());
  return intAttr.has_value() && !intAttr.value();
}

Value opFoldResultToIndex(const OpFoldResult &ofr, const Location &loc,
                          OpBuilder &b) {
  if (auto val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIndex() && "Provided ofr shoule be type of Index");
    return val;
  }

  auto intAttr = makeIntAttr(ofr);
  if (intAttr.has_value()) {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(intAttr.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

SmallVector<Value> opFoldResultToIndex(ArrayRef<OpFoldResult> ofrs,
                                       const Location &loc, OpBuilder &b) {
  return llvm::map_to_vector<4>(ofrs, [&](OpFoldResult ofr) -> Value {
    return opFoldResultToIndex(ofr, loc, b);
  });
}

Value createConstIntOp(const Location &loc, OpBuilder &b, int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIndexAttr(value)).getResult();
}

// TODO: imply these function below
OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);

  if (!lhsInt && rhsInt && rhsInt.value() == 0) {
    return lhs;
  }
  if (!rhsInt && lhsInt && lhsInt.value() == 0) {
    return rhs;
  }

  if (lhsInt && rhsInt) {
    return b.getIndexAttr(lhsInt.value() + rhsInt.value());
  }

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  } else {
    assert(isa<IndexType>(lhsValue.getType()));
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  } else {
    assert(isa<IndexType>(rhsValue.getType()));
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);

  if (!lhsInt && rhsInt && rhsInt.value() == 0) {
    return lhs;
  }

  if (lhsInt && rhsInt) {
    return b.getIndexAttr(lhsInt.value() - rhsInt.value());
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::SubIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);

  if (lhsInt) {
    if (lhsInt.value() == 0) {
      return lhs;
    }
    if (lhsInt.value() == 1) {
      return rhs;
    }
  }
  if (rhsInt) {
    if (rhsInt.value() == 0) {
      return rhs;
    }
    if (rhsInt.value() == 1) {
      return lhs;
    }
  }

  if (lhsInt && rhsInt) {
    return b.getIndexAttr(lhsInt.value() * rhsInt.value());
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const Value &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsConstFlag = false;

  auto rhsConstInt = std::numeric_limits<int64_t>::max();
  auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
  if (rhsOp) {
    rhsConstFlag = true;
    rhsConstInt = dyn_cast<IntegerAttr>(rhsOp.getValue()).getInt();
  }

  if (lhsInt && rhsConstFlag) {
    return b.getIndexAttr(lhsInt.value() * rhsConstInt);
  }

  if (lhsInt) {
    if (lhsInt.value() == 0) {
      return lhs;
    }
    if (lhsInt.value() == 1) {
      return rhs;
    }
  }
  if (rhsConstFlag) {
    if (rhsConstInt == 0) {
      return rhsOp.getResult();
    }
    if (rhsConstInt == 1) {
      return lhs;
    }
  }

  if (lhsInt && !rhsConstFlag) {
    auto lhsValue = createConstIntOp(loc, b, lhsInt.value());
    return b.create<arith::MulIOp>(loc, lhsValue, rhs).getResult();
  }
  assert(!lhsInt);
  return b.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs).getResult();
}

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);
  if (lhsInt) {
    if (lhsInt.value() == 0) {
      return lhs;
    }
  }
  if (rhsInt) {
    if (rhsInt.value() == 0) {
      emitError(loc) << "cannot div 0!";
      return OpFoldResult();
    }
    if (rhsInt.value() == 1) {
      return lhs;
    }
  }

  if (lhsInt && rhsInt) {
    return b.getIndexAttr(lhsInt.value() / rhsInt.value());
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::DivSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);
  if (lhsInt && lhsInt.value() == 0) {
    return lhs;
  }
  if (lhsInt && rhsInt) {
    return b.getIndexAttr(lhsInt.value() % rhsInt.value());
  }
  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::RemSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);
  if (lhsInt && rhsInt) {
    return b.getIndexAttr(std::min(lhsInt.value(), rhsInt.value()));
  }
  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::MinSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = makeIntAttr(lhs);
  auto rhsInt = makeIntAttr(rhs);
  if (lhsInt && rhsInt) {
    return b.getIndexAttr(std::max(lhsInt.value(), rhsInt.value()));
  }
  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt) {
    lhsValue = createConstIntOp(loc, b, lhsInt.value());
  }
  if (rhsInt) {
    rhsValue = createConstIntOp(loc, b, rhsInt.value());
  }
  return b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue).getResult();
}

LogicalResult
addReduceWithIndexAttrIfNeeded(ConversionPatternRewriter &rewriter,
                               linalg::ReduceOp reduceOp) {
  // To verify whether the operation of the reduceOp is ReduceWithIndex
  // TODO: maybe a better way of judging?
  auto ctx = reduceOp.getContext();
  Block &body = reduceOp.getCombiner().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());

  auto yieldValue = yieldOp.getValues();
  if (yieldValue.size() == 0) {
    return failure();
  }

  auto opIter = reduceOp.getBody()->without_terminator().begin();
  auto cmpMaskOp = dyn_cast<arith::CmpFOp>(*opIter);
  const StringRef reduceRef = "reduce_mode";
  if (cmpMaskOp) {
    if (cmpMaskOp.getPredicate() == arith::CmpFPredicate::OGT) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("max_with_index"));
    } else if (cmpMaskOp.getPredicate() == arith::CmpFPredicate::OLT) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("min_with_index"));
    }
  }

  auto cmpMaskIOp = dyn_cast<arith::CmpIOp>(*opIter);
  if (cmpMaskIOp) {
    if (cmpMaskIOp.getPredicate() == arith::CmpIPredicate::sgt) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("max_with_index"));
    } else if (cmpMaskIOp.getPredicate() == arith::CmpIPredicate::slt) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("min_with_index"));
    }
  }

  return success();
}

} // namespace mlir
