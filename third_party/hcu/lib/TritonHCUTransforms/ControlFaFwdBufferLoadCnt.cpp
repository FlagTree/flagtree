// clang-format off
#include <iterator>
#include <utility>

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
// 获取每个AsyncGlobalToLocal的group ,让他们作为forOp的参数传入 才能在
// forOp里进行asyn_wait修改

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h"
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static bool preCondition(Block *block) {
  SmallVector<Operation *> dots;
  block->walk([&](Operation *op) {
    if (isa<tt::DotOp>(op))
      dots.push_back(op);
  });
  if (dots.begin() != dots.end())
    return true;
  return false;
}

static bool controlBufferVmcnt(Block *block) {
  SmallVector<Operation *> dots;
  SmallVector<Operation *> loadIner;
  SmallVector<Operation *> waitInerOp;
  block->walk([&](Operation *op) {
    if (isa<tt::DotOp>(op))
      dots.push_back(op);
    if (isa<ttg::AsyncCopyGlobalToLocalOp>(op))
      loadIner.push_back(op);
    if (isa<ttg::AsyncWaitOp>(op)) {
      if (op->getOperands().size() > 1)
        op->erase();
      else
        waitInerOp.push_back(op);
    }
  });
  for (auto op : waitInerOp) {
    auto wait = dyn_cast<ttg::AsyncWaitOp>(op);
    int loadNum = wait.getNum();
    for (auto load : loadIner) {
      if (load->isBeforeInBlock(wait))
        ++loadNum;
    }
    wait.setNum(loadNum);
  }
  SmallVector<Operation *> dotOperands;
  for (Operation *op : dots) {
    auto dot = dyn_cast<tt::DotOp>(op);
    Value dotA = dot.getA();
    Value dotB = dot.getB();
    Operation *dotADef = dotA.getDefiningOp();
    Operation *dotBDef = dotB.getDefiningOp();
    if (isa<ttg::LocalLoadOp>(dotADef) && dotADef->getBlock() == block)
      dotOperands.push_back(dotADef);
    if (isa<ttg::LocalLoadOp>(dotBDef) && dotBDef->getBlock() == block)
      dotOperands.push_back(dotBDef);
  }

  SmallVector<Operation *> memorySub;

  llvm::MapVector<Operation *, Operation *> localLoadToMemSub;

  for (Operation *op : dotOperands) {
    auto localLoad = dyn_cast<ttg::LocalLoadOp>(op);
    auto loadOperand = localLoad.getOperand(0);
    if (isa<BlockArgument>(loadOperand)) {
      auto blockArg = dyn_cast<BlockArgument>(loadOperand);
      if (blockArg &&
          blockArg.getOwner() == *(block->getPredecessors().begin())) {
        auto cfBlock = *(blockArg.getOwner()->getPredecessors().begin());
        auto brOp = dyn_cast<cf::BranchOp>(cfBlock->getTerminator());
        auto argIdx = blockArg.getArgNumber();
        auto blockArgInit = brOp.getOperand(argIdx);
        auto ArgInitDef = blockArgInit.getDefiningOp();
        if (isa<ttg::MemDescSubviewOp>(ArgInitDef)) {
          memorySub.push_back(ArgInitDef);
          localLoadToMemSub[op] = ArgInitDef;
        }
      }
    }
  }

  SmallVector<Value> commGroup;
  llvm::MapVector<Operation *, Value> memSubToToken;
  for (auto memOp : memorySub) {
    for (auto op : memOp->getUsers()) {
      if (isa<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        for (auto commOp : op->getUsers()) {
          if (isa<ttg::AsyncCommitGroupOp>(commOp)) {
            memSubToToken[memOp] = commOp->getResult(0);
            commGroup.push_back(commOp->getResult(0));
            break;
          }
        }
      }
    }
  }
  SetVector<Operation *> waitOp;
  for (auto group : commGroup) {
    for (auto user : group.getUsers()) {
      if (isa<ttg::AsyncWaitOp>(user))
        waitOp.insert(user);
    }
  }
  for (auto wait : waitOp)
    wait->erase();
  return true;
}

class TritonHCUFaFwdControlCntPass
    : public TritonHCUFaFwdControlCntBase<TritonHCUFaFwdControlCntPass> {
public:
  TritonHCUFaFwdControlCntPass() = default;
  TritonHCUFaFwdControlCntPass(int cnt_control) {
    this->cnt_control = cnt_control;
  }

  void runOnOperation() override {
    SmallVector<Block *> brTrue;
    getOperation()->walk([&](Operation *op) {
      if (isa<cf::CondBranchOp>(op)) {
        auto cop = dyn_cast<cf::CondBranchOp>(op);
        auto block = cop.getTrueDest();
        brTrue.push_back(block);
      }
    });
    if (brTrue.empty())
      return;
    for (auto block : brTrue) {
      if (preCondition(block)) {
        controlBufferVmcnt(block);
      }
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonHCUFaFwdControlCntPass(int cnt_control) {
  return std::make_unique<TritonHCUFaFwdControlCntPass>(cnt_control);
}
