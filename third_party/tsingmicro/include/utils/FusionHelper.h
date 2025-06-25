#ifndef TRITON_FUSION_PATTERNS
#define TRITON_FUSION_PATTERNS

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;

namespace {

//===--------------------------- Match ArgMinMax --------------------------===//

// We're looking for an op that looks like this:
//
// %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
// ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
// -------------------------------------------------
// `matchTieBreakValue`                                |
//   %11 = arith.cmpf oeq, %arg9, %arg11 : f32         |
//   %12 = arith.cmpi slt, %arg10, %arg12 : i32        |   1.
//   %13 = arith.andi %11, %12 : i1                    |
// -------------------------------------------------   |-> `matchShouldUpdate`
// `matchUpdateCondition`                              |
//   %14 = arith.cmpf ogt, %arg9, %arg11 : f32         |   2.
// -------------------------------------------------   |
//   %15 = arith.ori %14, %13 : i1                     |
// -------------------------------------------------
//   %16 = arith.select %15, %arg9, %arg11 : f32
//   %17 = arith.select %15, %arg10, %arg12 : i32

static LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                         Value reduceValue, Value reduceIndex,
                                         mlir::Block::iterator &it,
                                         Value &tileBreakValue) {
  // Match the following (section 1. of the above)
  //
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
  //   %13 = arith.andi %11, %12 : i1
  //
  // which is equivalent to the following python code
  //
  //   tie = value1 == value2 and index1 < index2

  // matching: %11 = arith.cmpf oeq, %arg9, %arg11 : f32
  auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it++);
  if (!eqCmpOp || eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ ||
      currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
    return failure();
  }

  // matching: %12 = arith.cmpi slt, %arg10, %arg12 : i32
  auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
  if (!sltCmpOp || sltCmpOp.getPredicate() != arith::CmpIPredicate::slt ||
      currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
    return failure();
  }

  // matching: %13 = arith.andi %11, %12 : i1
  auto andOp = dyn_cast<arith::AndIOp>(*it++);
  if (!andOp || andOp.getLhs() != eqCmpOp || andOp.getRhs() != sltCmpOp) {
    return failure();
  }

  tileBreakValue = andOp;
  return success();
}

static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                           Value reduceValue, Value reduceIndex,
                                           mlir::Block::iterator &it,
                                           Value &comparisonResult,
                                           bool isArgMin) {
  // %14 = arith.cmpf olt(ogt), %arg9, %arg11 : f32
  auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
  if (!cmpOp) {
    return failure();
  }

  auto predicate =
      isArgMin ? arith::CmpFPredicate::OLT : arith::CmpFPredicate::OGT;
  if (cmpOp.getPredicate() != predicate || currValue != cmpOp.getLhs() ||
      reduceValue != cmpOp.getRhs()) {
    return failure();
  }

  comparisonResult = cmpOp;
  return success();
}

static LogicalResult
matchShouldUpdateValue(Value currValue, Value currIndex, Value reduceValue,
                       Value reduceIndex, mlir::Block::iterator &it,
                       Value &shouldUpdate, bool isArgMin) {
  Value tieResult;
  if (failed(matchTieBreakResult(currValue, currIndex, reduceValue, reduceIndex,
                                 it, tieResult))) {
    return failure();
  }

  Value comparisonResult;
  if (failed(matchComparisonResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, comparisonResult,
                                   isArgMin))) {
    return failure();
  }

  // matching: %15 = arith.ori %14, %13 : i1
  auto orOp = dyn_cast<arith::OrIOp>(*it++);
  if (!orOp || orOp.getLhs() != comparisonResult ||
      orOp.getRhs() != tieResult) {
    return failure();
  }

  shouldUpdate = orOp;
  return success();
}

LogicalResult matchSelect(mlir::Block::iterator &opsIt, Value curr,
                          Value reduce, Value shouldUpdate, Value &result) {
  auto selectOp = dyn_cast<arith::SelectOp>(*opsIt++);
  if (!selectOp) {
    return failure();
  }

  if (selectOp.getCondition() != shouldUpdate ||
      curr != selectOp.getTrueValue() || reduce != selectOp.getFalseValue()) {
    return failure();
  }

  result = selectOp;

  return success();
}

LogicalResult matchArgMinMax(Value currValue, Value currIndex,
                             Value reduceValue, Value reduceIndex,
                             mlir::Block::iterator &opsIt, Value &indexResult,
                             Value &valueResult, bool isArgMin) {
  Value shouldUpdate;
  if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                    reduceIndex, opsIt, shouldUpdate,
                                    isArgMin))) {
    return failure();
  }

  // matching: %16 = arith.select %15, %arg9, %arg11 : f32
  Value valueSelectOp;
  if (failed(matchSelect(opsIt, currValue, reduceValue, shouldUpdate,
                         valueSelectOp))) {
    return failure();
  }

  // matching:%17 = arith.select %15, %arg10, %arg12 : i32
  Value indexSelectOp;
  if (failed(matchSelect(opsIt, currIndex, reduceIndex, shouldUpdate,
                         indexSelectOp))) {
    return failure();
  }

  indexResult = indexSelectOp;
  valueResult = valueSelectOp;

  return success();
}

} // namespace

#endif
