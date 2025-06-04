#include "TritonToLinalg/MaskAnalysis.h"
// #include "triton-shared/Analysis/opFoldResultutils.h"
#include "Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "mask-analysis"

namespace mlir {

namespace triton {

LogicalResult MaskState::parse(Value operand, const Location &loc,
                               OpBuilder &builder) {
  if (isa<IntegerType>(operand.getType())) {
    return parseIntScalar(operand, loc, builder);
  }
  auto definingOp = operand.getDefiningOp();
  LLVM_DEBUG({
    llvm::dbgs() << "[MaskState]==> parse op\n"
                 << *definingOp << "\n[MaskState]<==\n";
  });
  return TypeSwitch<Operation *, LogicalResult>(definingOp)
      .Case<arith::ConstantOp>(
          [&](auto op) { return this->parseConstant(op, loc, builder); })
      .Case<arith::AddIOp>(
          [&](auto op) { return this->parseAdd(op, loc, builder); })
      .Case<arith::AndIOp>(
          [&](auto op) { return this->parseAnd(op, loc, builder); })
      .Case<arith::CmpIOp>(
          [&](auto op) { return this->parseCmp(op, loc, builder); })
      .Case<triton::MakeRangeOp>(
          [&](auto op) { return this->parseMakeRange(op, loc, builder); })
      .Case<triton::BroadcastOp>(
          [&](auto op) { return this->parseBroadcast(op, loc, builder); })
      .Case<triton::SplatOp>(
          [&](auto op) { return this->parseSplat(op, loc, builder); })
      .Case<triton::ExpandDimsOp>(
          [&](auto op) { return this->parseExpandDims(op, loc, builder); })
      .Case<arith::ExtSIOp>(
          [&](auto op) { return this->parse(op.getIn(), loc, builder); })
      .Case<arith::DivSIOp>(
          [&](auto op) { return this->parseDiv(op, loc, builder); })
      .Default([&](Operation *op) { return failure(); });
}

// extractSlice
tensor::ExtractSliceOp MaskState::getExtractSlice(Value source,
                                                  const Location &loc,
                                                  OpBuilder &builder) const {
  auto sourceRType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  auto dstRType = tensor::ExtractSliceOp::inferResultType(sourceRType, offsets,
                                                          dims, strides);
  return builder.create<tensor::ExtractSliceOp>(loc, dstRType, source, offsets,
                                                dims, strides);
}

tensor::InsertSliceOp MaskState::getInsertSlice(Value source, Value dest,
                                                const Location &loc,
                                                OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  return builder.create<tensor::InsertSliceOp>(loc, source, dest, offsets, dims,
                                               strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location &loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);
  return builder.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                           source, offsets, dims, strides);
}

static memref::SubViewOp createSubview(Value src, const Location &loc,
                                       OpBuilder &builder,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return builder.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), src,
                                           offsets, sizes, strides);
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2,
                                 const Location &loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 = builder.create<memref::DimOp>(loc, block1, 1).getResult();
  OpFoldResult subviewCol1 =
      minOpFoldResult(col1, subviewColFull, loc, builder);
  OpFoldResult subviewCol2 =
      subOpFoldResult(subviewColFull, subviewCol1, loc, builder);
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sbv1 = createSubview(block1, loc, builder, offsets,
                            {subviewRowFull, subviewCol1}, strides);
  auto sbv2 = createSubview(block2, loc, builder, offsets,
                            {subviewRowFull, subviewCol2}, strides);
  return {sbv1, sbv2};
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getStackedSubviews(Value block1, Value block2, const Location &loc,
                              OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult row1 = builder.create<memref::DimOp>(loc, block1, 0).getResult();
  OpFoldResult subviewRow1 =
      minOpFoldResult(row1, subviewRowFull, loc, builder);
  OpFoldResult subviewRow2 =
      subOpFoldResult(subviewRowFull, subviewRow1, loc, builder);
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sbv1 = createSubview(block1, loc, builder, offsets,
                            {subviewRow1, subviewColFull}, strides);
  auto sbv2 = createSubview(block2, loc, builder, offsets,
                            {subviewRow2, subviewColFull}, strides);
  return {sbv1, sbv2};
}

// addstatescalar
LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar,
                                        const Location &loc,
                                        OpBuilder &builder) {
  start = addOpFoldResult(state.start, scalar, loc, builder);
  end = addOpFoldResult(state.end, scalar, loc, builder);
  dims = state.dims;
  offsets = state.offsets;
  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }
  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar) {
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  } else {
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
  }
}

LogicalResult MaskState::divStateScalar(const MaskState &state,
                                        const OpFoldResult scalar,
                                        const Location &loc,
                                        OpBuilder &builder) {
  start = divOpFoldResult(state.start, scalar, loc, builder);
  end = divOpFoldResult(state.end, scalar, loc, builder);
  dims = state.dims;
  offsets = state.offsets;
  return success();
}

LogicalResult MaskState::divStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (!lhsState.scalar && rhsState.scalar) {
    if (isZeroIndex(rhsState.scalar)) {
      InFlightDiagnostic diag =
          emitError(loc)
          << "Unsupported scenario where rhs is zero constant in divide!";
      return failure();
    }

    return divStateScalar(lhsState, rhsState.scalar, loc, builder);
  }

  InFlightDiagnostic diag = emitError(loc)
                            << "Supported scenario where only rhs is a scalar";
  return failure();
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState,
                                   const Location &loc, OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsOffset = lhsState.offsets[i];
    auto rhsOffset = rhsState.offsets[i];
    auto newOffset = maxOpFoldResult(lhsOffset, rhsOffset, loc, builder);
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    auto lhsEnd = addOpFoldResult(lhsOffset, lhsDim, loc, builder);
    auto rhsEnd = addOpFoldResult(rhsOffset, rhsDim, loc, builder);
    auto newEnd = minOpFoldResult(lhsEnd, rhsEnd, loc, builder);
    auto newDim = subOpFoldResult(newEnd, newOffset, loc, builder);

    offsets.push_back(newOffset);
    dims.push_back(newDim);
  }
  return success();
}

// Helper func for MaskState::parse()
LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location &loc,
                                       OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }
  return success();
}

// parseIntScalar
LogicalResult MaskState::parseIntScalar(Value scalar, const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  Value castOp;

  if (scalar.getType().isInteger(1)) {
    castOp = builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(),
                                                  scalar);
  } else {
    castOp =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  }
  this->scalar = castOp;
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder))) {
    return failure();
  }
  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseDiv(arith::DivSIOp divOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(divOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(divOp.getRhs(), loc, builder))) {
    return failure();
  }
  return this->divStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());
  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)) ||
      !lhsState.isMask()) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)) ||
      !rhsState.isMask()) {
    return failure();
  }
  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location &loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sge &&
      cmpOp.getPredicate() != arith::CmpIPredicate::eq) {
    LLVM_DEBUG({ llvm::dbgs() << "Unsupported cmpi predicate\n"; });
    return failure();
  }
  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder))) {
    return failure();
  }

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder))) {
    return failure();
  }

  if (!(!lhsState.scalar && rhsState.scalar)) {
    cmpOp->emitRemark("[MaskState] Unsupported cmpi scenario");
    return failure();
  }

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = makeIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitError(loc)
                                  << "Unsupported cmpi with more than one  "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }

  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

  this->offsets = lhsState.offsets;
  this->dims = lhsState.dims;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::slt: {
    auto realBound =
        maxOpFoldResult(lhsState.start, rhsState.scalar, loc, builder);
    auto newEnd = minOpFoldResult(lhsState.end, realBound, loc, builder);
    auto newDim = subOpFoldResult(newEnd, lhsState.start, loc, builder);

    this->dims[cmpDim] = newDim;
    break;
  }
  case arith::CmpIPredicate::sge: {
    auto realBound =
        maxOpFoldResult(lhsState.start, rhsState.scalar, loc, builder);
    auto newStart = minOpFoldResult(lhsState.end, realBound, loc, builder);
    auto newOffset = subOpFoldResult(newStart, lhsState.start, loc, builder);
    auto newDim = subOpFoldResult(lhsState.end, newStart, loc, builder);

    this->offsets[cmpDim] = newOffset;
    this->dims[cmpDim] = newDim;
    break;
  }
  case arith::CmpIPredicate::eq: {
    auto newOffset =
        subOpFoldResult(rhsState.scalar, lhsState.start, loc, builder);
    auto newDim = builder.getIndexAttr(1);

    this->offsets[cmpDim] = newOffset;
    this->dims[cmpDim] = newDim;
    break;
  }
  default:
    return failure();
  }
  return success();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));
  this->offsets.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location &loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder))) {
    return failure();
  }
  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i]) {
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp,
                                    const Location &loc, OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  auto splatAsMask = [&](Operation *userOp) -> bool {
    return TypeSwitch<Operation *, bool>(userOp)
        .Case<arith::AndIOp>([&](arith::AndIOp andOp) { return true; })
        .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
          return selectOp.getCondition() == dst;
        })
        .Case<triton::LoadOp>(
            [&](triton::LoadOp loadOp) { return loadOp.getMask() == dst; })
        .Case<triton::StoreOp>(
            [&](triton::StoreOp storeOp) { return storeOp.getMask() == dst; })
        .Default([&](Operation *op) { return false; });
  };

  if (src.getType().isInteger(1) && !splatOp->use_empty() &&
      llvm::all_of(splatOp->getUsers(), splatAsMask)) {
    for (auto s : dstShape) {
      auto currentDim =
          mulOpFoldResult(builder.getIndexAttr(s), this->scalar, loc, builder);
      this->dims.push_back(currentDim);
      this->offsets.push_back(builder.getIndexAttr(0));
    }

    this->scalar = nullptr;
    return success();
  }

  for (auto s : dstShape) {
    this->dims.push_back(builder.getIndexAttr(s));
    this->offsets.push_back(builder.getIndexAttr(0));
  }
  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location &loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder))) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "Expect changed dimention to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));
  this->offsets.insert(this->offsets.begin() + axis, builder.getIndexAttr(0));

  return success();
}

void MaskState::eraseInsertedOps(Operation *rawOp, PatternRewriter &rewriter) {
  auto moduleOp = rawOp->getParentOfType<ModuleOp>();
  SmallVector<Operation *> worklist;
  moduleOp->walk([&](Operation *op) {
    if (isOpTriviallyDead(op))
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value value : op->getOperands()) {
      if (auto defOp = value.getDefiningOp())
        worklist.push_back(defOp);
    }
    LLVM_DEBUG({
      llvm::dbgs() << "[MaskState]==> inserted op: \n"
                   << *op << "\n[MaskState]<== is removed\n";
    });
    rewriter.eraseOp(op);
  }
}

} // namespace triton

} // namespace mlir
