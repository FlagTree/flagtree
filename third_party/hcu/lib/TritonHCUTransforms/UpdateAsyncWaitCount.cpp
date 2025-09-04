#include <iterator>
#include <limits>
#include <utility>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/hcu/lib/TritonHCUTransforms/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

int getNumberOfLoadInstructions(RankedTensorType srcTy, tt::MemDescType dstTy) {
  auto shape = srcTy.getShape();
  //   LinearLayout srcLayout = tt::gpu::toLinearLayout(shape,
  //   srcTy.getEncoding()); LinearLayout sharedLayout =
  //       tt::gpu::toLinearLayout(shape, dstTy.getEncoding());
  std::optional<LinearLayout> srcLayout =
      tt::gpu::toLinearLayout(shape, srcTy.getEncoding());
  std::optional<LinearLayout> sharedLayout =
      tt::gpu::toLinearLayout(shape, dstTy.getEncoding());
  LinearLayout srcToSharedLayout =
      srcLayout.value().invertAndCompose(sharedLayout.value());

  // On GFX9 we cannot split direct to lds loads into multiple ones because we
  // need coalesced writes. So we can divide the number of registers by the
  // contiguity to get the number of load instructions.
  int contig = srcToSharedLayout.getNumConsecutiveInOut();
  int numberOfRegisters = srcToSharedLayout.getInDimSize(
      StringAttr::get(srcTy.getContext(), "register"));
  int loadInstructionCount = std::max(1, numberOfRegisters / contig);
  return loadInstructionCount;
}

// The pipeliner always insert ops following an order of ttg.async_load ->
// [token] -> ttg.async_commit_group -> [token] -> ttg.async_wait. So here we
// scan the operands of ttg.async_commit_group to count the number of issued
// async load intrinsics.
int getNumberOfLoadInstructions(Operation *op) {
  if (isa<ttg::AsyncCommitGroupOp>(op)) {
    int count = 0;
    for (auto token : op->getOperands()) {
      auto defOp = token.getDefiningOp();
      if (!defOp)
        continue;
      if (auto copyOp = llvm::dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(defOp)) {
        count += getNumberOfLoadInstructions(copyOp.getSrc().getType(),
                                             copyOp.getResult().getType());
      }
    }
    return count;
  }
  if (isa<tt::LoadOp, tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp>(op)) {
    op->emitRemark("Global memory operation between async wait and "
                   "async_loads. This will hinder the interleaving of memory "
                   "operations and might impact performance.");
  }
  return 0;
}

void updateWaitCount(ttg::AsyncWaitOp waitOp) {
  int waitCnt = std::numeric_limits<int>::max();
  for (auto token : waitOp.getOperands()) {
    // Traverse def chain from waitOp to the producer of the token and count
    // the minumum number of vmcnt instructions
    auto tokenWaitCnt =
        deduceMinCountOnDefChain(token, waitOp, [](Operation *op) {
          return getNumberOfLoadInstructions(op);
        });
    waitCnt = std::min(waitCnt, tokenWaitCnt);
  }
  if (waitCnt == std::numeric_limits<int>::max() || waitOp.getNum() == waitCnt)
    return;
  waitOp.setNum(waitCnt);
}

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h"

struct TritonHCUUpdateAsyncWaitCountPass
    : public TritonHCUUpdateAsyncWaitCountBase<
          TritonHCUUpdateAsyncWaitCountPass> {

  TritonHCUUpdateAsyncWaitCountPass() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<ttg::AsyncWaitOp> waitOps;
    getOperation()->walk(
        [&](ttg::AsyncWaitOp waitOp) { waitOps.push_back(waitOp); });

    for (auto waitOp : waitOps)
      updateWaitCount(waitOp);
  }
};

std::unique_ptr<Pass> mlir::createTritonHCUUpdateAsyncWaitCountPass() {
  return std::make_unique<TritonHCUUpdateAsyncWaitCountPass>();
}
