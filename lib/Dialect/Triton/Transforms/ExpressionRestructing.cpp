#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"


#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

using namespace mlir;
using llvm::ArrayRef;
namespace mlir::triton{


struct Div2Mul : public OpRewritePattern<arith::DivFOp>{
    using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::DivFOp op, PatternRewriter &rewriter) const override{
        Value result = op.getResult();
        Value l = op.getLhs();
        Value r = op.getRhs();
        auto loc = op.getLoc();

        if (!result.hasOneUse())
            return failure();
        for (auto &use : result.getUses()){
            if(!dyn_cast<arith::DivFOp>(use.getOwner()))
                return failure();
            auto DivUser = dyn_cast<arith::DivFOp>(use.getOwner());
            if(DivUser.getLhs()!= op.getResult())
                return failure();
            auto originalInsertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(DivUser);
            auto loc_div = DivUser.getLoc();
            auto product = rewriter.create<arith::MulFOp>(loc_div, r, DivUser.getRhs());
            rewriter.setInsertionPointAfter(product);
            auto ResultEnd = rewriter.create<arith::DivFOp>(loc_div, l, product.getResult());
            rewriter.restoreInsertionPoint(originalInsertionPoint);
            rewriter.replaceOp(op, product.getResult());
            DivUser.replaceAllUsesWith(ResultEnd.getResult());
            rewriter.eraseOp(DivUser);
        }
        return success();
    }
};

struct Mul2Mul : public OpRewritePattern<arith::MulFOp>{
    using OpRewritePattern<arith::MulFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op, PatternRewriter &rewriter) const override{
        Value result = op.getResult();
        Value l = op.getLhs();
        Value r = op.getRhs();
        auto loc = op.getLoc();

        if (!result.hasOneUse())
            return failure();
        for (auto &use : result.getUses()){
            if(!dyn_cast<arith::MulFOp>(use.getOwner()))
                return failure();
            auto MulUser = dyn_cast<arith::MulFOp>(use.getOwner());
            if(!(MulUser.getLhs() == op.getResult() && ((MulUser.getRhs().getDefiningOp<arith::ConstantOp>()&& r.getDefiningOp<arith::ConstantOp>())||(r == MulUser.getRhs()))))
                return failure();
            auto originalInsertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(MulUser);
            auto loc_mul = MulUser.getLoc();
            auto product = rewriter.create<arith::MulFOp>(loc_mul, r, MulUser.getRhs());
            rewriter.setInsertionPointAfter(product);
            auto ResultEnd = rewriter.create<arith::MulFOp>(loc_mul, l, product.getResult());
            rewriter.restoreInsertionPoint(originalInsertionPoint);
            rewriter.replaceOp(op, product.getResult());
            MulUser.replaceAllUsesWith(ResultEnd.getResult());
            rewriter.eraseOp(MulUser);
        }
        return success();
    }
};

struct Add2Add : public OpRewritePattern<arith::AddFOp>{
    using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op, PatternRewriter &rewriter) const override{
        Value result = op.getResult();
        Value l = op.getLhs();
        Value r = op.getRhs();
        auto loc = op.getLoc();

        if (!result.hasOneUse())
            return failure();
        for (auto &use : result.getUses()){
            if(!dyn_cast<arith::AddFOp>(use.getOwner()))
                return failure();
            auto AddUser = dyn_cast<arith::AddFOp>(use.getOwner());
            if(!(AddUser.getLhs() == op.getResult() && ((AddUser.getRhs().getDefiningOp<arith::ConstantOp>()&& r.getDefiningOp<arith::ConstantOp>())||(r == AddUser.getRhs()))))
                return failure();
            auto originalInsertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(AddUser);
            auto loc_add = AddUser.getLoc();
            auto sum = rewriter.create<arith::AddFOp>(loc_add, r, AddUser.getRhs());
            rewriter.setInsertionPointAfter(sum);
            auto ResultEnd = rewriter.create<arith::AddFOp>(loc_add, l, sum.getResult());
            rewriter.restoreInsertionPoint(originalInsertionPoint);
            rewriter.replaceOp(op, sum.getResult());
            AddUser.replaceAllUsesWith(ResultEnd.getResult());
            rewriter.eraseOp(AddUser);
        }
        return success();
    }
};

struct Sub2Add : public OpRewritePattern<arith::SubFOp>{
    using OpRewritePattern<arith::SubFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::SubFOp op, PatternRewriter &rewriter) const override{
        Value result = op.getResult();
        Value l = op.getLhs();
        Value r = op.getRhs();
        auto loc = op.getLoc();

        if (!result.hasOneUse())
            return failure();
        for (auto &use : result.getUses()){
            if(!dyn_cast<arith::SubFOp>(use.getOwner()))
                return failure();
            auto SubUser = dyn_cast<arith::SubFOp>(use.getOwner());
            if(!(SubUser.getLhs() == op.getResult() && ((SubUser.getRhs().getDefiningOp<arith::ConstantOp>()&& r.getDefiningOp<arith::ConstantOp>())||(r == SubUser.getRhs()))))
                return failure();
            auto originalInsertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(SubUser);
            auto loc_sub = SubUser.getLoc();
            auto sum = rewriter.create<arith::AddFOp>(loc_sub, r, SubUser.getRhs());
            rewriter.setInsertionPointAfter(sum);
            auto ResultEnd = rewriter.create<arith::SubFOp>(loc_sub, l, sum.getResult());
            rewriter.restoreInsertionPoint(originalInsertionPoint);
            rewriter.replaceOp(op, sum.getResult());
            SubUser.replaceAllUsesWith(ResultEnd.getResult());
            rewriter.eraseOp(SubUser);
        }
        return success();
    }
};

class ExpressionRestructingPass : public TritonExpressionRestructingBase<ExpressionRestructingPass>{
public:
    void runOnOperation() override{
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        ModuleOp m = getOperation();
        patterns.add<Div2Mul>(context);
        patterns.add<Mul2Mul>(context);
        patterns.add<Add2Add>(context);
        patterns.add<Sub2Add>(context);

        if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
            signalPassFailure();
    }
};


std::unique_ptr<mlir::Pass> createExpressionRestructingPass(){
    return std::make_unique<ExpressionRestructingPass>();
}

}



