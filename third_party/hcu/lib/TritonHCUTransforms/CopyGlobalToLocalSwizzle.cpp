// clang-format off
#include <iterator>
#include <utility>
#include <algorithm>
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "llvm/Support/Debug.h"
// clang-format on
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using triton::AddPtrOp;
using triton::gpu::AsyncCopyGlobalToLocalOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::SharedEncodingAttr;
using triton::gpu::SliceEncodingAttr;
RankedTensorType getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

Value getNoSwizzleOffset(Value offset, bool dstRow) {
  auto offsetInner = offset;
  if (auto arg = dyn_cast<BlockArgument>(offset)) {
    auto blockOp = dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
    unsigned argIndex = arg.getArgNumber();
    offsetInner = blockOp.getInitArgs()[argIndex - 1];
  }
  auto offsetDefOp = offsetInner.getDefiningOp(); // trunci, addi
  if (isa<mlir::arith::TruncIOp>(offsetDefOp)) {
    offsetDefOp = offsetDefOp->getOperand(0).getDefiningOp();
  }
  auto offsetOp = dyn_cast<mlir::arith::AddIOp>(offsetDefOp);
  auto rhs = offsetOp.getRhs();
  auto lhs = offsetOp.getLhs();
  auto rhsValue = rhs.getDefiningOp()->getOperand(0);
  auto rhsShape = dyn_cast<RankedTensorType>(rhsValue.getType()).getShape();
  auto colOff = rhsShape[0] == 1 ? rhs : lhs;
  auto rowOff = rhsShape[1] == 1 ? rhs : lhs;
  return dstRow ? rowOff : colOff;
}
bool getNoSwizzleMask(mlir::arith::AndIOp rowColMaskDef, bool dstRow) {
  Value rMask = rowColMaskDef.getRhs();
  auto rMaskDef = rMask.getDefiningOp();
  if (auto rMaskOp = dyn_cast<triton::BroadcastOp>(rMaskDef)) {
    auto rMaskShape =
        cast<RankedTensorType>(rMaskOp.getSrc().getType()).getShape();
    if ((rMaskShape[0] == 1 && dstRow) || (rMaskShape[1] == 1 && !dstRow))
      return 0;
  }
  return 1;
}

Value changeMaskInner(Location loc, mlir::PatternRewriter &rewriter,
                      Value swizzleMask, Value colOff, Attribute inLy) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = nullptr;
  getBackwardSlice(swizzleMask, &slice, opt);
  slice.insert(swizzleMask.getDefiningOp());
  SmallVector<std::pair<mlir::arith::CmpIPredicate, Value>> predToVal;
  Value colM = colOff;
  for (Operation *op : slice) {
    if (auto cmpOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
      auto val = cmpOp.getRhs();
      if (auto cmpTy = dyn_cast<RankedTensorType>(val.getType())) {
        auto cmpLy = cmpTy.getEncoding();
        if (auto sliceLy = dyn_cast<SliceEncodingAttr>(cmpLy))
          val =
              rewriter.create<triton::ExpandDimsOp>(loc, val, sliceLy.getDim());
      }
      auto cmpTy = getNewType(val.getType(), inLy);
      val = rewriter.create<ConvertLayoutOp>(loc, cmpTy, val);
      unsigned int bitWidth = cast<RankedTensorType>(colOff.getType())
                                  .getElementType()
                                  .getIntOrFloatBitWidth();
      if (bitWidth == 64) {
        auto colOffShape = dyn_cast<RankedTensorType>(val.getType()).getShape();
        auto colOffEncoding =
            dyn_cast<RankedTensorType>(val.getType()).getEncoding();
        auto newRowType = RankedTensorType::get(
            colOffShape, rewriter.getI64Type(), colOffEncoding);
        val = rewriter.create<arith::ExtSIOp>(loc, newRowType, val);
      }
      auto cVal =
          rewriter.create<triton::BroadcastOp>(loc, colOff.getType(), val);
      predToVal.push_back(
          std::make_pair(cmpOp.getPredicate(), cVal.getResult()));
    } else if (auto addOp = dyn_cast<mlir::arith::AddIOp>(op)) {
      auto val = addOp.getLhs();
      auto rVal = addOp.getRhs();
      auto rDef = rVal.getDefiningOp();
      if (isa<arith::ConstantOp, triton::SplatOp>(rDef))
        val = rVal;
      if (auto addTy = dyn_cast<RankedTensorType>(val.getType())) {
        auto addLy = addTy.getEncoding();
        if (auto sliceLy = dyn_cast<SliceEncodingAttr>(addLy)) {
          val =
              rewriter.create<triton::ExpandDimsOp>(loc, val, sliceLy.getDim());
        }
        auto addNTy = getNewType(val.getType(), inLy);
        val = rewriter.create<ConvertLayoutOp>(loc, addNTy, val);
        unsigned int bitWidth = cast<RankedTensorType>(colOff.getType())
                                    .getElementType()
                                    .getIntOrFloatBitWidth();
        if (bitWidth == 64) {
          auto colOffShape =
              dyn_cast<RankedTensorType>(val.getType()).getShape();
          auto colOffEncoding =
              dyn_cast<RankedTensorType>(val.getType()).getEncoding();
          auto newRowType = RankedTensorType::get(
              colOffShape, rewriter.getI64Type(), colOffEncoding);
          val = rewriter.create<arith::ExtSIOp>(loc, newRowType, val);
        }
        val = rewriter.create<triton::BroadcastOp>(loc, colOff.getType(), val);
        colM = rewriter.create<mlir::arith::AddIOp>(loc, val, colM);
      }
    } else if (auto subOp = dyn_cast<mlir::arith::SubIOp>(op)) {
      if (auto subTy = dyn_cast<RankedTensorType>(subOp.getLhs().getType())) {
        auto subLy = subTy.getEncoding();
        Value val;
        if (auto sliceLy = dyn_cast<SliceEncodingAttr>(subLy))
          val = rewriter.create<triton::ExpandDimsOp>(loc, addOp.getLhs(),
                                                      sliceLy.getDim());
        auto subNTy = getNewType(val.getType(), inLy);
        val = rewriter.create<ConvertLayoutOp>(loc, subNTy, val);
        val = rewriter.create<triton::BroadcastOp>(loc, colOff.getType(), val);
        colM = rewriter.create<mlir::arith::SubIOp>(loc, val, colM);
      }
    }
  }

  SmallVector<Value> res;
  for (auto kv : predToVal) {
    Operation *op =
        rewriter.create<mlir::arith::CmpIOp>(loc, kv.first, colM, kv.second);
    res.push_back(op->getResult(0));
  }
  auto newColMask = res[0];
  for (int i = 1; i < res.size(); ++i)
    newColMask = rewriter.create<mlir::arith::AndIOp>(loc, newColMask, res[i]);

  return newColMask;
}

Value changeMask(Location loc, mlir::PatternRewriter &rewriter, Value mask,
                 Value colOff, RankedTensorType inTy, bool dstRow) {
  Operation *currentOp = mask.getDefiningOp();
  auto inLy = dyn_cast<BlockedEncodingAttr>(inTy.getEncoding());
  if (isa<mlir::arith::ConstantOp, triton::SplatOp>(currentOp)) {
    auto maskLy = dyn_cast<RankedTensorType>(mask.getType()).getEncoding();
    auto naiveLy = dyn_cast<BlockedEncodingAttr>(maskLy);
    return (naiveLy.getSizePerThread() == inLy.getSizePerThread() &&
            naiveLy.getThreadsPerWarp() == inLy.getThreadsPerWarp() &&
            naiveLy.getWarpsPerCTA() == inLy.getWarpsPerCTA() &&
            naiveLy.getOrder() == inLy.getOrder())
               ? mask
               : rewriter.create<ConvertLayoutOp>(loc, inTy, mask);
  } else {
    auto maskDef = dyn_cast<mlir::arith::AndIOp>(currentOp);
    auto rowColMask = maskDef.getRhs();
    auto offSetMask = maskDef.getLhs();
    auto newOffsetMask =
        rewriter.create<ConvertLayoutOp>(loc, inTy, offSetMask);
    if (auto arg = dyn_cast<BlockArgument>(rowColMask)) {
      auto newRowColMask =
          rewriter.create<ConvertLayoutOp>(loc, inTy, rowColMask);
      return rewriter.create<mlir::arith::AndIOp>(loc, newOffsetMask,
                                                  newRowColMask);
    }
    auto rowColDef = rowColMask.getDefiningOp();
    if (isa<mlir::arith::AndIOp>(rowColDef)) {
      auto rowColMaskDef = dyn_cast<mlir::arith::AndIOp>(rowColDef);

      // 判断哪个为rowMask -哪个colMask --
      bool rowColMaskDim = getNoSwizzleMask(rowColMaskDef, dstRow);
      auto rowMask =
          rowColMaskDim ? rowColMaskDef.getRhs() : rowColMaskDef.getLhs();
      auto colMask =
          rowColMaskDim ? rowColMaskDef.getLhs() : rowColMaskDef.getRhs();

      auto rowMaskType = dyn_cast<RankedTensorType>(rowMask.getType());
      Value newRowMask =
          rowMaskType.getEncoding() == inTy.getEncoding()
              ? rowMask
              : rewriter.create<ConvertLayoutOp>(loc, inTy, rowMask);

      auto swizzleMask = colMask;
      auto newColMask =
          changeMaskInner(loc, rewriter, swizzleMask, colOff, inLy);
      auto newMask =
          rewriter.create<mlir::arith::AndIOp>(loc, newRowMask, newColMask);
      newMask =
          rewriter.create<mlir::arith::AndIOp>(loc, newOffsetMask, newMask);

      return newMask;
    } else {
      // 单个维度的mask
      if (isa<BroadcastOp>(rowColDef)) {
        auto rowOrColDef = dyn_cast<BroadcastOp>(rowColDef);
        auto rowOrColValue = rowOrColDef.getSrc();
        auto rowOrColShape =
            dyn_cast<RankedTensorType>(rowOrColValue.getType()).getShape();
        if ((dstRow && rowOrColShape[0] == 1) ||
            (!dstRow && rowOrColShape[1] == 1)) {
          auto newColMask =
              changeMaskInner(loc, rewriter, rowOrColValue, colOff, inLy);
          auto newMask = rewriter.create<mlir::arith::AndIOp>(
              loc, newOffsetMask, newColMask);
          return newMask;
        }
      }
      // 直接返回原来的mask
      auto maskLy = dyn_cast<RankedTensorType>(mask.getType()).getEncoding();
      auto naiveLy = dyn_cast<BlockedEncodingAttr>(maskLy);
      return (naiveLy.getSizePerThread() == inLy.getSizePerThread() &&
              naiveLy.getThreadsPerWarp() == inLy.getThreadsPerWarp() &&
              naiveLy.getWarpsPerCTA() == inLy.getWarpsPerCTA() &&
              naiveLy.getOrder() == inLy.getOrder())
                 ? mask
                 : rewriter.create<ConvertLayoutOp>(loc, inTy, mask);
    }
  }
}

void applySwizzle(Location loc, mlir::Operation *op,
                  mlir::PatternRewriter &rewriter, RankedTensorType ppType,
                  RankedTensorType newTensorType, Value row, Value col,
                  Value rowOff, Value basePtr, triton::AddPtrOp addptr,
                  unsigned int outVec, unsigned int perPhase,
                  unsigned int maxPhase, bool isConvert, bool dstRow,
                  unsigned int inVec, unsigned int bitWidth) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = nullptr;
  getBackwardSlice(row, &slice, opt);

  Operation *makeRange = nullptr;
  bool isLoadOrWhere = false;
  for (auto op : slice) {
    if (isa<tt::LoadOp, mlir::arith::SelectOp>(op))
      isLoadOrWhere = true;
    if (isa<tt::MakeRangeOp>(op))
      makeRange = op;
  }
  if (isLoadOrWhere && makeRange) {
    auto rowDef = dyn_cast<tt::ExpandDimsOp>(row.getDefiningOp());
    auto axisVal = rowDef.getAxis();
    row = rewriter.create<tt::ExpandDimsOp>(loc, makeRange->getResult(0),
                                            axisVal);
  }

  auto copyOp = dyn_cast<AsyncCopyGlobalToLocalOp>(op);
  if (bitWidth == 64) {
    auto rowShape = dyn_cast<RankedTensorType>(row.getType()).getShape();
    auto colShape = dyn_cast<RankedTensorType>(col.getType()).getShape();
    auto rowEncoding = dyn_cast<RankedTensorType>(row.getType()).getEncoding();
    auto newRowType =
        RankedTensorType::get(rowShape, rewriter.getI64Type(), rowEncoding);
    auto newColType =
        RankedTensorType::get(colShape, rewriter.getI64Type(), rowEncoding);
    row = rewriter.create<arith::ExtSIOp>(loc, newRowType, row);
    col = rewriter.create<arith::ExtSIOp>(loc, newColType, col);
  }
  auto ppTypeR = getNewType(dyn_cast<RankedTensorType>(row.getType()),
                            ppType.getEncoding());
  auto ppTypeC = getNewType(dyn_cast<RankedTensorType>(col.getType()),
                            ppType.getEncoding());

  auto rowb = rewriter.create<triton::BroadcastOp>(
      loc, ppType,
      (isConvert ? rewriter.create<ConvertLayoutOp>(loc, ppTypeR, row) : row));
  auto ccol = rewriter.create<triton::BroadcastOp>(
      loc, ppType,
      (isConvert ? rewriter.create<ConvertLayoutOp>(loc, ppTypeC, col) : col));

  auto outVecConst =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, outVec, bitWidth);
  auto outVecConstant =
      rewriter.create<triton::SplatOp>(loc, ppType, outVecConst);

  auto phaseConst =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, perPhase, bitWidth);
  auto maxPhaseConst =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, maxPhase, bitWidth);
  auto splatPhase = rewriter.create<triton::SplatOp>(loc, ppType, phaseConst);
  auto splatMaxPhase =
      rewriter.create<triton::SplatOp>(loc, ppType, maxPhaseConst);
  auto rowPhase = rewriter.create<mlir::arith::RemUIOp>(
      loc, rewriter.create<mlir::arith::DivUIOp>(loc, rowb, splatPhase),
      splatMaxPhase);

  auto coloff = rewriter.create<mlir::arith::XOrIOp>(
      loc, rewriter.create<mlir::arith::DivUIOp>(loc, ccol, outVecConstant),
      rowPhase);
  auto colOffSwizzled =
      rewriter.create<mlir::arith::MulIOp>(loc, coloff, outVecConstant);

  auto colOffInVec =
      rewriter.create<mlir::arith::RemUIOp>(loc, ccol, outVecConstant);
  auto colFinalOff =
      rewriter.create<mlir::arith::AddIOp>(loc, colOffSwizzled, colOffInVec);

  auto newOffset = rewriter.create<mlir::arith::AddIOp>(
      loc,
      (isConvert ? rewriter.create<ConvertLayoutOp>(loc, ppType, rowOff)
                 : rowOff),
      colFinalOff);
  auto newOffsetVal = bitWidth == 32 ? newOffset.getResult()
                                     : rewriter.create<mlir::arith::TruncIOp>(
                                           loc, newTensorType, newOffset);

  auto oldMask = copyOp.getMask();
  if (oldMask) {
    auto maskType =
        getNewType(dyn_cast<RankedTensorType>(copyOp.getMask().getType()),
                   newTensorType.getEncoding());
    auto newMask =
        changeMask(loc, rewriter, oldMask, colFinalOff, maskType, dstRow);
    copyOp.setOperand(2, newMask);
  }

  Value newAddPtr;
  Operation *newCopy;
  if (!isConvert) {
    newAddPtr = rewriter.create<triton::AddPtrOp>(loc, addptr.getType(),
                                                  basePtr, newOffsetVal);
    newCopy = rewriter.create<AsyncCopyGlobalToLocalOp>(
        loc, newAddPtr, copyOp.getResult(), copyOp.getMask(), copyOp.getOther(),
        copyOp.getCache(), copyOp.getEvict(), copyOp.getIsVolatile(), true,
        inVec);
  } else {

    auto defPtrOp = basePtr.getDefiningOp();
    auto splatop = dyn_cast<triton::SplatOp>(defPtrOp);
    auto srcPtr = splatop.getSrc();
    auto newPtrType = RankedTensorType::get(ppType.getShape(), srcPtr.getType(),
                                            ppType.getEncoding());
    auto newSplatPtr =
        rewriter.create<triton::SplatOp>(loc, newPtrType, srcPtr);
    newAddPtr = rewriter.create<triton::AddPtrOp>(loc, newPtrType, newSplatPtr,
                                                  newOffsetVal);
    Value newOther;
    if (copyOp.getOther()) {
      auto otherType =
          getNewType(dyn_cast<RankedTensorType>(copyOp.getOther().getType()),
                     newTensorType.getEncoding());
      auto newOther =
          rewriter.create<ConvertLayoutOp>(loc, otherType, copyOp.getOther());
    }
    newCopy = rewriter.create<AsyncCopyGlobalToLocalOp>(
        loc, newAddPtr, copyOp.getResult(), copyOp.getMask(), newOther,
        copyOp.getCache(), copyOp.getEvict(), copyOp.getIsVolatile(), true,
        outVec);
  }

  rewriter.replaceAllUsesWith(op->getResult(0), newCopy->getResult(0));
}
class GlobalToLocalSwizzle : public mlir::RewritePattern {

public:
  GlobalToLocalSwizzle(mlir::MLIRContext *context)
      : mlir::RewritePattern(ttg::AsyncCopyGlobalToLocalOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto ctx = rewriter.getContext();

    auto copyOp = dyn_cast<AsyncCopyGlobalToLocalOp>(op);
    if (!copyOp || copyOp.getIsSwizzle())
      return mlir::failure();
    Value src = copyOp.getSrc();
    auto dstTy = copyOp.getResult().getType();
    auto dstLyt = dstTy.getEncoding();
    auto dstSharedLayout = dyn_cast<SharedEncodingAttr>(dstLyt);
    unsigned maxPhase = dstSharedLayout.getMaxPhase();
    if (maxPhase == 1)
      return mlir::failure();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcLyt = cast<RankedTensorType>(src.getType()).getEncoding();
    auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
    auto srcLayout = dyn_cast<BlockedEncodingAttr>(srcLyt);

    auto dstOrder = dstSharedLayout.getOrder();
    auto srcOrder = srcLayout.getOrder();

    auto outVec = dstSharedLayout.getVec();
    auto perPhase = dstSharedLayout.getPerPhase();

    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    unsigned inVec =
        triton::gpu::getUniqueContigPerThread(srcLayout, srcShape)[dstOrder[0]];

    Operation *currentOp = src.getDefiningOp(); // addptrOp
    auto addptr = dyn_cast<triton::AddPtrOp>(currentOp);
    Value row, col, rowOff;

    bool srcRow = srcOrder[0] == 1 ? 1 : 0;
    bool dstRow = dstOrder[0] == 1 ? 1 : 0;

    Value offset = addptr.getOffset();
    Value basePtr = addptr.getPtr();
    Value res = addptr.getResult();

    auto backSlice = [&](Value offset, SetVector<Operation *> &slice) {
      SmallVector<Value> backList = {offset};
      while (!backList.empty()) {
        Value val = backList.pop_back_val();
        if (Operation *def = val.getDefiningOp()) {
          if (slice.insert(def)) {
            for (Value operand : def->getOperands())
              backList.push_back(operand);
          }
        } else if (auto arg = dyn_cast<BlockArgument>(val)) {
          Operation *parentOp = arg.getOwner()->getParentOp();
          if (slice.insert(parentOp)) {
            for (auto input : parentOp->getOperands())
              backList.push_back(input);
          }
        }
      }
    };

    SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = nullptr;
    if (auto arg = dyn_cast<BlockArgument>(offset)) {
      unsigned argIndex = arg.getArgNumber();
      auto blockOp = dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
      offset = blockOp.getInitArgs()[argIndex - 1];
    }
    getBackwardSlice(offset, &slice, opt);

    for (Operation *bop : slice) {
      if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(bop)) {
        uint32_t axisNum = expandOp.getAxis();
        if ((axisNum == 1 && dstRow) || (axisNum == 0 && !dstRow))
          row = expandOp.getResult();
        if ((axisNum == 1 && !dstRow) || (axisNum == 0 && dstRow))
          col = expandOp.getResult();
      }
    }
    // when m = 1 OR n = 1 ,  all other rows or columns are 0
    // swizzle is not executed to reduce vgpr usage
    if (!row || !col) {
      SetVector<Operation *> cumSlice;
      backSlice(offset, cumSlice);
      for (Operation *bop : cumSlice) {
        if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(bop)) {
          uint32_t axisNum = expandOp.getAxis();
          if (!row && ((axisNum == 1 && dstRow) || (axisNum == 0 && !dstRow)))
            row = expandOp.getResult();
          if (!col && ((axisNum == 1 && !dstRow) || (axisNum == 0 && dstRow)))
            col = expandOp.getResult();
        }
      }
      if (!row || !col) {
        return mlir::failure();
      }
    }
    rowOff = getNoSwizzleOffset(offset, dstRow);
    auto loc = op->getLoc();
    RankedTensorType ppType, newTensorType;
    auto rowOffType =
        cast<RankedTensorType>(rowOff.getType()).getElementType(); // i32
    auto rowShape = cast<RankedTensorType>(offset.getType()).getShape();
    auto rowSlice2d = cast<RankedTensorType>(offset.getType()).getEncoding();
    unsigned bitWidth = rowOffType.getIntOrFloatBitWidth();
    if (inVec <= outVec) {
      ppType = RankedTensorType::get(rowShape, rowOffType, rowSlice2d);
      newTensorType =
          RankedTensorType::get(rowShape, rewriter.getI32Type(), rowSlice2d);
      applySwizzle(loc, op, rewriter, ppType, newTensorType, row, col, rowOff,
                   basePtr, addptr, outVec, perPhase, maxPhase, false, dstRow,
                   inVec, bitWidth);

    } else {
      SmallVector<unsigned> sizePerThread(srcTy.getRank(), 1);
      sizePerThread[srcOrder[0]] = outVec;
      auto warpsPerCTA =
          dyn_cast<BlockedEncodingAttr>(rowSlice2d).getWarpsPerCTA();
      auto numWarps = accumulate(warpsPerCTA.begin(), warpsPerCTA.end(), 1,
                                 std::multiplies<int>());
      auto CTALayout = triton::gpu::getCTALayout(rowSlice2d);
      auto newLayout = BlockedEncodingAttr::get(
          ctx, rowShape, sizePerThread, srcOrder, numWarps, 64, CTALayout);

      ppType = RankedTensorType::get(rowShape, rowOffType, newLayout);
      newTensorType =
          RankedTensorType::get(rowShape, rewriter.getI32Type(), newLayout);

      applySwizzle(loc, op, rewriter, ppType, newTensorType, row, col, rowOff,
                   basePtr, addptr, outVec, perPhase, maxPhase, true, dstRow,
                   inVec, bitWidth);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h"

struct TritonHCUGlobalToLocalSwizzlePass
    : public TritonHCUGlobalToLocalSwizzleBase<
          TritonHCUGlobalToLocalSwizzlePass> {
public:
  TritonHCUGlobalToLocalSwizzlePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    mlir::RewritePatternSet patterns(context);
    patterns.add<GlobalToLocalSwizzle>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonHCUGlobalToLocalSwizzlePass() {
  return std::make_unique<TritonHCUGlobalToLocalSwizzlePass>();
}
