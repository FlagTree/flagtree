#ifndef TRITONNPU_UTILS_UTILS_H
#define TRITONNPU_UTILS_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

#include <functional>
#include <optional>

namespace mlir {

namespace ConverterUtils {

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order);

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n);

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter);

memref::SubViewOp makeSubViewOp(Value src,
                                llvm::SmallVectorImpl<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter);

void getShapeInfo(Value val, llvm::SmallVectorImpl<OpFoldResult> &shapes,
                  ConversionPatternRewriter &rewriter);

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 Value adaptorPtr, const Location &loc,
                 ConversionPatternRewriter &rewriter);

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst);

SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst);

} // namespace ConverterUtils

class ConversionPatternRewriter;

namespace triton {

enum class IndirectLoadInterfaceOpType { Undefined = 0, Load = 1, Calc = 2 };

// Traceback from rootOp to find the targetOp with the specified condition
mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn);

void traverseBackwardUpdateOperandChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder);

void traverseBackwardUpdateOperandChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<void(OpBuilder &, Operation *)> actionFn);

void traverseForwardUpdateUserChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    llvm::SmallPtrSet<Operation *, 16> &stopOps);

void traverseForwardUpdateUserChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    llvm::SmallPtrSet<Operation *, 16> &stopOps);

// UseAnalysis will tag operations whose results are used only as meta-data
// with "MetaUse" tag.
bool isMetaUse(Operation *op);

bool isMixUse(Operation *op);

IndirectLoadInterfaceOpType getIndirectLoadInterfaceOpType(Operation *op);

bool opIsIndirectLoad(Operation *op);

bool opIsIndirectCalc(Operation *op);

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder);

ModuleOp getModuleOpFromOperation(Operation *op);

} // namespace triton

class OpBuilder;

std::optional<int64_t> makeIntAttr(const OpFoldResult &ofr);

bool hasConstantZero(const OpFoldResult &ofr);

Value opFoldResultToIndex(const OpFoldResult &ofr, const Location &loc,
                          OpBuilder &b);

SmallVector<Value> opFoldResultToIndex(ArrayRef<OpFoldResult> ofrs,
                                       const Location &loc, OpBuilder &b);

Value createConstIntOp(const Location &loc, OpBuilder &b, int64_t value);

OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const Value &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

LogicalResult
addReduceWithIndexAttrIfNeeded(ConversionPatternRewriter &rewriter,
                               linalg::ReduceOp reduceOp);

} // namespace mlir

#endif // TRITONNPU_UTILS_UTILS_H
