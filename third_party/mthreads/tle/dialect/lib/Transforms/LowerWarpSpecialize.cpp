#ifdef __TLE__

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEPREPAREWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kDefaultProducerAttr =
    "musa_tle.default_producer";
static constexpr StringLiteral kConsumerEpochSyncAttr =
    "musa_tle.consumer_epoch_sync";
static constexpr StringLiteral kConsumerEpochRequiredAttr =
    "musa_tle.consumer_epoch_rendezvous_required";
static constexpr StringLiteral kExplicitCtaSyncAttr =
    "musa_tle.explicit_cta_sync";

static FailureOr<unsigned>
countExplicitCtaSyncs(Region &partition,
                      std::optional<unsigned> workerIndex = std::nullopt) {
  unsigned count = 0;
  bool invalid = false;
  partition.walk([&](ttg::BarrierOp barrier) {
    if (!barrier->hasAttr(kExplicitCtaSyncAttr))
      return WalkResult::advance();
    if (barrier.getAddrSpace() != ttg::AddrSpace::Local) {
      barrier.emitOpError(
          "MUSA TLE explicit CTA sync must target local memory");
      invalid = true;
      return WalkResult::interrupt();
    }
    if (barrier->getBlock() != &partition.front()) {
      auto diagnostic = barrier.emitOpError("MUSA TLE explicit CTA sync in ");
      if (workerIndex)
        diagnostic << "worker partition #" << *workerIndex;
      else
        diagnostic << "the default partition";
      diagnostic << " must be directly in the partition top-level block";
      invalid = true;
      return WalkResult::interrupt();
    }
    ++count;
    return WalkResult::advance();
  });
  if (invalid)
    return failure();
  return count;
}

static LogicalResult validateExplicitCtaSyncs(ttg::WarpSpecializeOp ws) {
  FailureOr<unsigned> defaultCount =
      countExplicitCtaSyncs(ws.getDefaultRegion());
  if (failed(defaultCount))
    return failure();

  for (auto [workerIndex, workerRegion] :
       llvm::enumerate(ws.getPartitionRegions())) {
    FailureOr<unsigned> workerCount =
        countExplicitCtaSyncs(*workerRegion, workerIndex);
    if (failed(workerCount))
      return failure();
    if (*workerCount != *defaultCount)
      return ws.emitOpError()
             << "MUSA TLE explicit CTA sync count mismatch: default "
                "partition has "
             << *defaultCount << " but worker partition #" << workerIndex
             << " has " << *workerCount;
  }
  return success();
}

static bool containsAsyncSqmmaAndGlobalStore(scf::ForOp loop) {
  bool hasAsyncSqmma = false;
  bool hasGlobalStore = false;
  loop.walk([&](Operation *op) {
    if (auto squadDot = dyn_cast<triton::musa::SquadDotOp>(op))
      hasAsyncSqmma |= squadDot.getIsAsync();
    else if (auto sqmma = dyn_cast<triton::mtgpu::SqmmaOp>(op))
      hasAsyncSqmma |= sqmma.getIsAsync();
    hasGlobalStore |= isa<triton::StoreOp>(op);
  });
  return hasAsyncSqmma && hasGlobalStore;
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getConstantLoopBounds(scf::ForOp loop) {
  APInt lower;
  APInt upper;
  APInt step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)))
    return failure();
  return std::make_tuple(lower.getSExtValue(), upper.getSExtValue(),
                         step.getSExtValue());
}

// SQMMA wait-all retires one partition's work, but it is not a completion
// edge between independent consumer partitions.  Persistent kernels that
// reuse a warp-specialized CTA across output tiles therefore need one shared
// rendezvous at each consumer outer-loop latch.  Mark only the canonical,
// constant-bound shape used by the persistent GEMM schedule; late lowering
// merges equal markers into one hardware barrier.
static LogicalResult markConsumerEpochRendezvous(
    ttg::WarpSpecializeOp ws, IRRewriter &rewriter) {
  if (ws->hasAttr(kDefaultProducerAttr) ||
      !ws->hasAttr(kConsumerEpochRequiredAttr))
    return success();
  auto skipInference = [&]() {
    ws->removeAttr(kConsumerEpochRequiredAttr);
    return success();
  };

  SmallVector<Region *> consumers{&ws.getDefaultRegion()};
  SmallVector<Region *> partitions(ws.getPartitionRegions().begin(),
                                   ws.getPartitionRegions().end());
  // The regular static layout places the producer in the final worker
  // partition.  The default region and preceding workers are consumers.
  for (unsigned index = 0; index + 1 < partitions.size(); ++index)
    consumers.push_back(partitions[index]);
  if (consumers.size() < 2)
    return skipInference();

  SmallVector<SmallVector<scf::ForOp>> candidatesByConsumer;
  candidatesByConsumer.reserve(consumers.size());
  bool hasPersistentCandidate = false;
  for (Region *consumer : consumers) {
    auto &candidates = candidatesByConsumer.emplace_back();
    consumer->walk([&](scf::ForOp loop) {
      if (containsAsyncSqmmaAndGlobalStore(loop))
        candidates.push_back(loop);
    });
    hasPersistentCandidate |= !candidates.empty();
  }
  if (!hasPersistentCandidate)
    return skipInference();

  SmallVector<scf::ForOp> epochLoops;
  std::optional<std::tuple<int64_t, int64_t, int64_t>> commonBounds;
  for (const SmallVector<scf::ForOp> &candidates : candidatesByConsumer) {
    // Nested K loops contain SQMMA but no output store.  Requiring exactly
    // one SQMMA+store loop keeps this inference deliberately conservative.
    if (candidates.size() != 1) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires exactly one "
          "SQMMA/store epoch loop in every consumer partition");
      return failure();
    }
    FailureOr<std::tuple<int64_t, int64_t, int64_t>> bounds =
        getConstantLoopBounds(candidates.front());
    if (failed(bounds)) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires constant "
          "consumer epoch loop bounds");
      return failure();
    }
    auto [lower, upper, step] = *bounds;
    if (lower != 0 || step != 1 || upper <= 1) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires a zero-based "
          "unit-step consumer epoch loop with at least two iterations");
      return failure();
    }
    if (commonBounds && *commonBounds != *bounds) {
      ws.emitOpError(
          "MUSA TLE persistent named-reader rendezvous requires identical "
          "consumer epoch loop bounds");
      return failure();
    }
    commonBounds = *bounds;
    epochLoops.push_back(candidates.front());
  }

  for (scf::ForOp loop : epochLoops) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    rewriter.setInsertionPoint(yield);
    auto barrier = ttg::BarrierOp::create(rewriter, yield.getLoc(),
                                          ttg::AddrSpace::Local);
    barrier->setAttr(kConsumerEpochSyncAttr,
                     rewriter.getI32IntegerAttr(0));
  }
  ws->removeAttr(kConsumerEpochRequiredAttr);
  return success();
}

class PrepareWarpSpecializePass
    : public impl::TritonMUSAGPUTLEPrepareWarpSpecializeBase<
          PrepareWarpSpecializePass> {
  LogicalResult prepareWarpSpecialize(ModuleOp mod, ttg::WarpSpecializeOp ws,
                                      IRRewriter &rewriter) {
    auto func = ws->getParentOfType<triton::FuncOp>();
    if (!func)
      return ws.emitOpError("requires a parent Triton function");

    Block &entry = func.getBody().front();
    if (ws->getBlock() != &entry)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize must be in the function entry "
          "block");
    if (!ws->getNextNode() || !isa<triton::ReturnOp>(ws->getNextNode()) ||
        ws->getNextNode()->getNextNode())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize must be the final operation "
          "before tt.return");
    if (ws.getNumResults() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize does not support results");

    auto partitions = ws.getPartitionOp();
    auto workerRegions = partitions.getPartitionRegions();
    ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
    if (workerRegions.empty())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires at least one worker "
          "partition");
    if (workerRegions.size() != workerWarps.size())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize worker region and warp-count "
          "sizes must match");
    if (ws->hasAttr(kDefaultProducerAttr) && workerRegions.size() != 1)
      return ws.emitOpError(
          "MUSA TLE default producer requires exactly one consumer worker "
          "partition");
    if (auto requestedRegisters = ws.getRequestedRegisters()) {
      if (requestedRegisters->size() != workerRegions.size())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize requested-register count must "
            "match the worker partition count");
    }

    Region &defaultRegion = ws.getDefaultRegion();
    if (!defaultRegion.hasOneBlock())
      return ws.emitOpError(
          "MUSA TLE static warp_specialize requires a single-block default "
          "region");

    Block &defaultBlock = defaultRegion.front();
    auto defaultYield =
        dyn_cast<ttg::WarpYieldOp>(defaultBlock.getTerminator());
    if (!defaultYield || defaultYield.getNumOperands() != 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default region must yield no "
          "values");

    ValueRange captures = partitions.getExplicitCaptures();
    for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
      if (!workerRegion.hasOneBlock())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must be single-block";
      Block &workerBlock = workerRegion.front();
      if (!isa<ttg::WarpReturnOp>(workerBlock.getTerminator()))
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " must end with ttg.warp_return";
      if (workerBlock.getNumArguments() != captures.size())
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " capture count mismatch";
      for (auto [captureIndex, argumentAndCapture] :
           llvm::enumerate(llvm::zip(workerBlock.getArguments(), captures))) {
        auto [argument, capture] = argumentAndCapture;
        if (argument.getType() != capture.getType())
          return ws.emitOpError()
                 << "MUSA TLE static warp_specialize worker partition #"
                 << workerIndex << " capture #" << captureIndex
                 << " type mismatch";
      }
    }
    DominanceInfo dominance(func);
    for (Value capture : captures) {
      if (!dominance.dominates(capture, ws.getOperation()))
        return ws.emitOpError(
            "MUSA TLE static warp_specialize capture must dominate the "
            "partition split");
    }

    if (failed(validateExplicitCtaSyncs(ws)))
      return failure();

    int32_t baseNumWarps = ttg::lookupNumWarps(ws);
    if (baseNumWarps <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize default warp count must be "
          "positive");
    int32_t warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    if (warpSize <= 0)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize warp size must be positive");

    int64_t nextWarp = baseNumWarps;
    SmallVector<int32_t> workerStartIds;
    workerStartIds.reserve(workerWarps.size());
    for (auto [workerIndex, numWarps] : llvm::enumerate(workerWarps)) {
      if (numWarps <= 0)
        return ws.emitOpError()
               << "MUSA TLE static warp_specialize worker partition #"
               << workerIndex << " requires a positive static warp count";
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
      workerStartIds.push_back(static_cast<int32_t>(nextWarp));
      nextWarp += numWarps;
      if (nextWarp > std::numeric_limits<int32_t>::max())
        return ws.emitOpError(
            "MUSA TLE static warp_specialize warp range exceeds int32");
    }
    if (nextWarp > std::numeric_limits<int32_t>::max() / warpSize)
      return ws.emitOpError(
          "MUSA TLE static warp_specialize thread count exceeds int32");

    int32_t totalNumWarps = static_cast<int32_t>(nextWarp);
    if (auto existingStartIds = ws.getWarpGroupStartIds()) {
      if (existingStartIds->size() != workerStartIds.size() ||
          !llvm::equal(*existingStartIds, workerStartIds))
        return ws.emitOpError(
            "MUSA TLE static worker warp ranges must follow declaration "
            "order after the default partition");
    } else {
      ws.setWarpGroupStartIds(workerStartIds);
    }
    if (auto existing =
            mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps")) {
      if (existing.getInt() != totalNumWarps)
        return ws.emitOpError(
            "MUSA TLE static warp_specialize conflicts with existing "
            "ttg.total-num-warps");
    } else {
      mod->setAttr("ttg.total-num-warps",
                   rewriter.getI32IntegerAttr(totalNumWarps));
    }

    if (failed(markConsumerEpochRendezvous(ws, rewriter)))
      return failure();

    return success();
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty())
      return;
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "MUSA TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(prepareWarpSpecialize(mod, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
