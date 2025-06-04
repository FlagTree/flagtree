#ifndef TRITON_ADAPTER_TRITONOPCONVERTER_H
#define TRITON_ADAPTER_TRITONOPCONVERTER_H

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "triton-to-linalg"

#include "llvm/Support/Debug.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

/*
Convert `tt.precise_div` operation to `arith.divf` operation.
tensor_x / tensor_y

```ttir
  %11 = tt.precise_divf %7, %10 : tensor<100xf32>
```

converts to:

```mlir
  %11 = arith.divf %7, %10 : tensor<100xf32>
```
*/
struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
public:
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/*
 * Rewrite arith.select with contiguouse mask to
 * tensor.extract_slice/insert_slice.
 */
class SelectConverter : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override;
};

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
class BitcastCanonicalizer : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::BitcastOp bitcastOp,
                                PatternRewriter &rewriter) const override;
};

template <typename MathOp>
class ScalarMathCanonicalizer : public OpRewritePattern<MathOp> {
public:
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer expects single scalar output.");
    }
    if (!op->getResult(0).getType().isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles scalar load scene.");
    }
    if (auto linalgOp = op->template getParentOfType<triton::ReduceOp>()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles op not within tt.reduce.");
    }
    auto loc = op.getLoc();
    llvm::SmallVector<Value> inputs;
    for (auto input : op->getOperands()) {
      auto blkTy = RankedTensorType::get({(int64_t)1}, input.getType());
      auto inputSplat = rewriter.create<triton::SplatOp>(loc, blkTy, input);
      inputs.push_back(inputSplat.getResult());
    }
    auto blkOp = rewriter.create<MathOp>(loc, inputs);
    Value offset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, blkOp.getResult(), offset);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

class DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplatConverter : public OpConversionPattern<triton::SplatOp> {
public:
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
public:
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
public:
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ClampFConverter : public OpConversionPattern<triton::ClampFOp> {
public:
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
public:
  explicit ReduceConverter(MLIRContext *context)
      : OpConversionPattern<triton::ReduceOp>(context) {}

  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const;

  bool isReductionOpSupported(Operation *redOp) const;

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const;

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const;

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const;

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const;

  LogicalResult
  convertToLinalgReduceExtended(ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const;

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ExternElementwiseClOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class JoinConverter : public OpConversionPattern<triton::JoinOp> {
public:
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplitConverter : public OpConversionPattern<triton::SplitOp> {
public:
  using OpConversionPattern<triton::SplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class CatConverter : public OpConversionPattern<triton::CatOp> {
public:
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class GatherConverter : public OpConversionPattern<triton::GatherOp> {
private:
  static constexpr llvm::StringRef gatherFuncNameBase = "triton_gather";

public:
  using OpConversionPattern<triton::GatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class YieldConverter : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class LoopConverter : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AdvanceConverter : public OpConversionPattern<triton::AdvanceOp> {
public:
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class MakeTensorPtrConverter
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
private:
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;

  void populateVectorAsIndex(SmallVector<OpFoldResult> &vec,
                             Operation::operand_range ops,
                             ConversionPatternRewriter &rewriter,
                             Location loc) const;

  memref::ReinterpretCastOp
  createRedundantOp(triton::MakeTensorPtrOp op,
                    ConversionPatternRewriter &rewriter, BlockData &data) const;

  OpFoldResult
  accumulatePotentialOffsetOnBase(triton::MakeTensorPtrOp op, Value base,
                                  OpFoldResult offset,
                                  ConversionPatternRewriter &rewriter) const;

public:
  explicit MakeTensorPtrConverter(MLIRContext *context)
      : OpConversionPattern<triton::MakeTensorPtrOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TransposeConverter : public OpConversionPattern<triton::TransOp> {
public:
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
public:
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonMulhiuiConverter : public OpConversionPattern<triton::MulhiUIOp> {
public:
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonPreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
public:
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class AssertConverter : public OpRewritePattern<triton::AssertOp> {
public:
  using OpRewritePattern<triton::AssertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AssertOp op,
                                PatternRewriter &rewriter) const override;
};

class DevicePrintConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

private:
  static constexpr llvm::StringRef printFuncNameBase = "triton_print";
  static constexpr llvm::StringRef prefixAttrName = "prefix";
  static constexpr llvm::StringRef hexAttrName = "hex";

public:
  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // end of namespace TTOpConverters

#endif
