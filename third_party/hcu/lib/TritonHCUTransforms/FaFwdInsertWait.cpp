#include <iterator>
#include <utility>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h"
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static bool preCondition(scf::ForOp forOp) {
  SmallVector<Operation *> dots;
  // 不是外部forOp
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            if (isa<tt::DotOp>(op))
              dots.push_back(op);
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  if (!dots.size())
    return false;
  return true;
}

static void addParamForBlockRet(RewriterBase &rewriter, scf::ForOp forOp,
                                SmallVector<Operation *> &loadIner,
                                SmallVector<Operation *> &outerMemorySub) {
  SmallVector<Operation *> localAlloc;
  for (auto memOp : outerMemorySub) {
    auto memAlloc = memOp->getOperand(0).getDefiningOp();
    localAlloc.push_back(memAlloc);
  }
  SmallVector<Value> newArgs;
  for (auto alloc : localAlloc) {
    for (auto loadOp : loadIner) {
      auto memTarget = loadOp->getOperand(1);
      auto memDef = memTarget.getDefiningOp();
      auto memAlloc = memDef->getOperand(0).getDefiningOp();
      if (memAlloc == alloc) {
        for (auto commOp : loadOp->getUsers()) {
          if (isa<ttg::AsyncCommitGroupOp>(commOp)) {
            newArgs.push_back(commOp->getResult(0));
            break;
          }
        }
      }
    }
  }
  appendToForOpYield(forOp, newArgs);
}

static void
insertWaitNum(RewriterBase &rewriter, scf::ForOp forOp,
              llvm::MapVector<Operation *, Operation *> &localLoadToMemSub,
              llvm::MapVector<Operation *, Value> &memSubToToken,
              llvm::MapVector<Value, int> &tokenToArgNum) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(forOp.getBody());
  SmallVector<std::pair<Operation *, int>> localToNum;
  int loadNum = tokenToArgNum.size();
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp>(op))
      loadNum++;
    if (isa<ttg::LocalLoadOp>(op)) {
      --loadNum;
      localToNum.push_back(std::make_pair(&op, loadNum));
    }
  }
  for (auto localNum : localToNum) {
    auto mem = localLoadToMemSub[localNum.first];
    if (memSubToToken.find(mem) != memSubToToken.end()) {
      auto token = memSubToToken[mem];
      auto argNum = tokenToArgNum[token];
      auto argToken = forOp.getRegionIterArgs()[argNum - 1];
      auto wait = rewriter.create<ttg::AsyncWaitOp>(forOp.getLoc(), argToken,
                                                    localNum.second);
      wait->moveBefore(localNum.first);
    }
  }
}

static Operation *isLoadLocal(Operation *dotOperands) {
  SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.omitBlockArguments = false;
  options.filter = nullptr;
  getBackwardSlice(dotOperands->getResult(0), &slice, options);
  for (Operation *op : slice) {
    if (isa<ttg::LocalLoadOp>(op) && op->getBlock() == dotOperands->getBlock())
      return op;
  }
  return nullptr;
}
class TritonHCUFaFwdWaitPass
    : public TritonHCUFaFwdWaitBase<TritonHCUFaFwdWaitPass> {
public:
  TritonHCUFaFwdWaitPass() = default;
  TritonHCUFaFwdWaitPass(int cnt_control) { this->cnt_control = cnt_control; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp forOp) {
      if (preCondition(forOp))
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;
    for (scf::ForOp forOp : loops) {
      SmallVector<Operation *> dots;
      SmallVector<Operation *> loadIner;
      for (Operation &op : forOp.getBody()->without_terminator()) {
        if (isa<tt::DotOp>(op))
          dots.push_back(&op);
        if (isa<ttg::AsyncCopyGlobalToLocalOp>(op))
          loadIner.push_back(&op);
      }
      SmallVector<Operation *> dotOperands;
      for (Operation *op : dots) {
        auto dot = dyn_cast<tt::DotOp>(op);
        Value dotA = dot.getA();
        Value dotB = dot.getB();
        // 向上找，只要跟dot相关就行
        Operation *dotADef = dotA.getDefiningOp();
        Operation *dotBDef = dotB.getDefiningOp();
        if (Operation *loadOp = isLoadLocal(dotADef))
          dotOperands.push_back(loadOp);
        if (Operation *loadOp = isLoadLocal(dotBDef))
          dotOperands.push_back(loadOp);
      }
      SmallVector<Operation *> outerMemorySub;
      llvm::MapVector<Operation *, Operation *> localLoadToMemSub;
      for (Operation *op : dotOperands) {
        auto localLoad = dyn_cast<ttg::LocalLoadOp>(op);
        auto loadOperand = localLoad.getOperand(0);
        if (isa<BlockArgument>(loadOperand)) {
          auto blockArg = dyn_cast<BlockArgument>(loadOperand);
          auto argIdx = blockArg.getArgNumber();
          auto blockArgInit = forOp.getBody()->getArgument(argIdx);
          auto argInitDef = forOp.getInitArgs()[argIdx - 1].getDefiningOp();
          if (isa<ttg::MemDescSubviewOp>(argInitDef)) {
            outerMemorySub.push_back(argInitDef);
            localLoadToMemSub[op] = argInitDef;
          }
        }
      }
      SmallVector<Value> commGroup;
      llvm::MapVector<Operation *, Value> memSubToToken;
      for (auto memOp : outerMemorySub) {
        for (auto op : memOp->getUsers()) {
          if (isa<ttg::AsyncCopyGlobalToLocalOp>(op)) {
            for (auto commOp : op->getUsers()) {
              if (isa<ttg::AsyncCommitGroupOp>(commOp) &&
                  (forOp.getOperation()->getBlock() != commOp->getBlock() ||
                   commOp->isBeforeInBlock(forOp))) {
                memSubToToken[memOp] = commOp->getResult(0);
                commGroup.push_back(commOp->getResult(0));
                break;
              }
            }
          }
        }
      }

      unsigned int argsNum = forOp.getBody()->getNumArguments();
      llvm::MapVector<Value, int> tokenToArgNum;
      for (auto group : commGroup)
        tokenToArgNum[group] = argsNum++;
      IRRewriter builder(forOp.getContext());
      builder.setInsertionPoint(forOp);
      scf::ForOp newForOp =
          replaceForOpWithNewSignature(builder, forOp, commGroup);
      forOp.erase();
      forOp = newForOp;
      addParamForBlockRet(builder, forOp, loadIner, outerMemorySub);
      insertWaitNum(builder, forOp, localLoadToMemSub, memSubToToken,
                    tokenToArgNum);
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonHCUFaFwdWaitPass(int cnt_control) {
  return std::make_unique<TritonHCUFaFwdWaitPass>(cnt_control);
}
