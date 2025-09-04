#include "TritonHCUTransforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "third_party/hcu/include/triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                                    Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp->getResult(0).getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp->getOperand(2));
    return ifOp;
  }
  return tt::predicateOp(rewriter, op, pred);
}

namespace {
class StreamPipeliner {
  enum SchedType {
    SCHED_GLOBAL_LOAD,
    SCHED_LOCAL_STORE,
    SCHED_LOCAL_LOAD,
    SCHED_COMPUTE,
    SCHED_ASYNC_WAIT,
    SCHED_SIZE
  };

public:
  StreamPipeliner(scf::ForOp _forOp, int _numStages, int _globalPrefetch,
                  int _localPrefetch, bool _useAsyncCopy)
      : forOp(_forOp), numStages(_numStages), numBuffers(1),
        useAsyncCopy(_useAsyncCopy), schedule(numStages),
        axisInfoAnalysis(forOp->getParentOfType<ModuleOp>()) {
    int lastStage = numStages - 1;
    stages[SCHED_GLOBAL_LOAD] = 0;
    stages[SCHED_LOCAL_STORE] = _globalPrefetch;
    stages[SCHED_LOCAL_LOAD] = lastStage - _localPrefetch;
    stages[SCHED_COMPUTE] = lastStage;
    stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
  }

  LogicalResult pipelineLoop();

private:
  LogicalResult initSchedule(int maxIndirectionLevel);

  void computeLoadOpsToIndirectionLevelAndUse();
  void assignMemoryLayouts();
  LogicalResult scheduleLoads(DenseSet<Operation *> &rootUsers);
  void scheduleDependencies();
  void scheduleDistanceOneDependencies();
  void scheduleRemainingToLastStage();

  LogicalResult preprocessLoopAndBuildSchedule();
  Value createAlloc(Operation *loadOp, ttg::SharedEncodingAttr sharedEnc);
  bool createAsyncCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx);
  void createStreamCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx);
  void createStreamOps();

  void scheduleOp(Operation *op, SchedType type, int stage = -1) {
    if (stage < 0)
      stage = stages[type];
    schedule.insert(op, stage, clusters[type]);
  }

private:
  scf::ForOp forOp;

  int numStages;
  int numBuffers;
  bool useAsyncCopy;

  int stages[SCHED_SIZE];
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusters;
  tt::CoarseSchedule schedule;

  SmallVector<std::tuple<Operation *, int, Operation *>> loadOpToIndLevelAndUse;

  struct LoadInfo {
    ttg::SharedEncodingAttr sharedEncoding = nullptr;
    int distToUse = 0;
    bool usedByDot = false;
    bool isAsync = false;
  };

  llvm::MapVector<Operation *, LoadInfo> loadToInfo;
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis;
  SmallVector<Value> sharedMemAllocs;
  tt::PipeliningOption options;
};
} // namespace
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.back());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}
LogicalResult StreamPipeliner::initSchedule(int maxIndirectionLevel) {
  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxIndirectionLevel;
  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    return failure();
  }
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  if (useAsyncCopy) {
    numBuffers += 1;
  }

  int asyncWaitCluster = 0;
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    localLoadCluster = localStoreCluster - 1;
  }

  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }
  std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE> clusterVec;
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  return success();
}

bool StreamPipeliner::createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                      Value extractIdx) {
  assert(useAsyncCopy);
  if (numBuffers == 1)
    return false;

  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  Value src = loadOp.getPtr();
  auto srcTy = cast<TensorOrMemDesc>(src.getType());

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  auto sharedEncodingAttr =
      cast<ttg::SharedEncodingAttr>(allocTy.getEncoding());

  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);

  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, alloc, viewLoad);
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loadOp.getLoc(), src, viewLoad, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));

  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad, waitOp);

  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copyOp, loadStage, loadCluster);
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage != stages[SCHED_LOCAL_LOAD])
    scheduleOp(waitOp, SCHED_ASYNC_WAIT);

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
    scheduleOp(sharedLoad, SCHED_LOCAL_LOAD);

  loadOp->replaceAllUsesWith(ValueRange{sharedLoad});
  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] &&
      sharedLoad->hasOneUse()) {
    if (auto cvt =
            dyn_cast<ttg::ConvertLayoutOp>(*sharedLoad->getUsers().begin()))
      scheduleOp(cvt, SCHED_LOCAL_LOAD);
  }

  loadOp.erase();
  return true;
}

void StreamPipeliner::createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                       Value extractIdx) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  Operation *copy = builder.clone(*loadOp);

  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  // Clean up old local caches.
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  // Prefetch load ahead of the dot stage if is used by the dot.
  auto storeOp =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);
  scheduleOp(viewLoad, SCHED_LOCAL_STORE);
  scheduleOp(storeOp, SCHED_LOCAL_STORE);

  // Create local load
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
  Value result = sharedLoad.getResult();
  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
    scheduleOp(sharedLoad, SCHED_LOCAL_LOAD);

  loadOp->replaceAllUsesWith(ValueRange{result});

  if (stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] && result.hasOneUse()) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*result.getUsers().begin()))
      scheduleOp(cvt, SCHED_LOCAL_LOAD);
  }

  loadOp.erase();
}

void StreamPipeliner::computeLoadOpsToIndirectionLevelAndUse() {
  DenseSet<Operation *> seen;

  // Recursively visit the given op and its operands to discover all load ops
  // and collect their indirection levels and uses.
  std::function<void(Operation *, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        // Skip previously visited load ops.
        if (!seen.insert(op).second)
          return;

        if (isa<tt::LoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.emplace_back(op, distance, use);
          use = op;
          ++distance;
        }
        for (Value operand : op->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasTrait<OpTrait::DotLike>())
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }
}

static ttg::HCUMfmaEncodingAttr getDotEncoding(Value inputValue,
                                               unsigned *opIdx) {
  if (!llvm::hasSingleElement(inputValue.getUses()))
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  if (auto dotOp = dyn_cast<tt::DotOp>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::HCUMfmaEncodingAttr>(dotType.getEncoding());
  }
  return getDotEncoding(user->getResult(0), opIdx);
}

static std::optional<ttg::SharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value loadedValue) {
  ttg::SharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<tt::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(userResult).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;

      auto srcTy = cast<TensorOrMemDesc>(loadedValue.getType());
      auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      auto userResEnc = cast<TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        tempAttr = ttg::SharedEncodingAttr::get(
            loadedValue.getContext(), dotOpEnc, srcTy.getShape(), order,
            ctaLayout, bitWidth, /*needTrans=*/false);
      }
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

void StreamPipeliner::assignMemoryLayouts() {
  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(op))
      // TODO: We'd need to verify that the distance is the same.
      continue;

    auto loadOp = cast<tt::LoadOp>(op);
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy) {
      continue;
    }

    auto pointeeTy =
        cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
    unsigned width = vec * pointeeTy.getIntOrFloatBitWidth();

    LoadInfo loadInfo;
    if (isa<tt::DotOp>(use)) {
      // Only use shared memory when feeding into a dot op.
      loadInfo.usedByDot = true;
      // If the max continugous bits we can read is < 32, buffer in registers.
      if (width >= 32) {
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0)).value_or(nullptr);
      }
    } else if (auto useOp = dyn_cast<tt::LoadOp>(use)) {
      if (loadToInfo.count(useOp) == 0) {
        continue;
      }
    }

    loadToInfo[op] = loadInfo;
  }
}

LogicalResult StreamPipeliner::scheduleLoads(DenseSet<Operation *> &rootUsers) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  computeLoadOpsToIndirectionLevelAndUse();

  if (loadOpToIndLevelAndUse.empty())
    return failure();

  // Check which loads are good for pipelining, and assign them memory layouts.
  assignMemoryLayouts();
  if (loadToInfo.empty())
    return failure();

  // Filter out load ops that cannot be pipelined.
  int resize = 0;
  // 可以认为只有dot的直接load才进行流水
  for (int i = 0, e = loadOpToIndLevelAndUse.size(); i < e; ++i) {
    auto [loadOp, distance, use] = loadOpToIndLevelAndUse[i];
    if (loadToInfo.count(loadOp) != 0)
      loadOpToIndLevelAndUse[resize++] = loadOpToIndLevelAndUse[i];
  }
  loadOpToIndLevelAndUse.resize(resize);

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse)
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);

  if (maxIndirectionLevel >= numStages)
    return failure();

  if (failed(initSchedule(maxIndirectionLevel)))
    return failure();

  assert(numStages >= 2 && "requires num_stages=2 at least");
  // 1
  unsigned stagesBetweenLoads =
      llvm::divideCeil(numStages - 2, maxIndirectionLevel + 1);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(use)) {
      scheduleOp(use, SCHED_COMPUTE);
      rootUsers.insert(use);
    }
  }

  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    scheduleOp(loadOp, SCHED_GLOBAL_LOAD, stage);
  }

  // Calculate distance from the load to the use.
  // 2
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }

  return success();
}

void StreamPipeliner::scheduleDependencies() {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; ++stage) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void StreamPipeliner::scheduleDistanceOneDependencies() {
  auto getNestedOperands = [](Operation *op) {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  DenseMap<tt::CoarseSchedule::Cluster *, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getArgNumber() == 0 || arg.getOwner() != op.getBlock())
        continue;
      auto yieldOp = op.getBlock()->getTerminator();
      Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
      Operation *defOp = v.getDefiningOp();
      if (!defOp || schedule.count(defOp) != 0)
        continue;
      if (isa<tt::LoadOp>(defOp)) {
        // Exception: schedule loads with a distance of 1 together with the
        // current op.
        schedule.insertIfAbsent(defOp, stage, cluster);
        schedule.insertDepsOfOp(defOp, stage, cluster, true);
      } else {
        if (dist1Cluster.count(&cluster) == 0) {
          dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
        }
        schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
        schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster], true);
      }
    }
  }
}

void StreamPipeliner::scheduleRemainingToLastStage() {
  int lastStage = numStages - 1;
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  auto cluster = clusters[SCHED_COMPUTE];
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      opToCluster[&op] = cluster;
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == lastStage) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        tt::CoarseSchedule::Cluster userCluster = opToCluster[user];
        tt::CoarseSchedule::Cluster opCluster = schedule[op].second;
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, lastStage, cluster);
  }
}

Value StreamPipeliner::createAlloc(Operation *loadOp,
                                   ttg::SharedEncodingAttr sharedEnc) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), numBuffers);
  Type memdescType = tt::MemDescType::get(bufferShape, ty.getElementType(),
                                          sharedEnc, sharedMemorySpace,
                                          /*mutableMemory=*/true);
  auto alloc = builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
  sharedMemAllocs.push_back(alloc);
  return alloc;
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
void StreamPipeliner::createStreamOps() {
  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding || info.isAsync)
      continue;

    Value alloc = createAlloc(loadOp, info.sharedEncoding);
    assert(alloc && "Failed to create alloc for the async load.");
    loadToAllocs.emplace_back(loadOp, alloc);
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  unsigned curArgIdx = forOp.getNumRegionIterArgs();
  scf::ForOp newLoop =
      replaceForOpWithNewSignature(builder, forOp, {extractIdx});
  if (builder.getInsertionPoint() == forOp->getIterator())
    builder.setInsertionPoint(newLoop);
  forOp.erase();
  forOp = newLoop;

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Replace tt.loads with async copies or stream copies
  for (auto &[op, alloc] : loadToAllocs) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (useAsyncCopy && createAsyncCopy(loadOp, alloc, extractIdx))
        continue;
      createStreamCopy(loadOp, alloc, extractIdx);
    }
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});
}

LogicalResult StreamPipeliner::preprocessLoopAndBuildSchedule() {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  if (failed(scheduleLoads(rootUsers)))
    return failure();
  if (loadToInfo.empty())
    return failure();

  // Convert the loads into shared memory allocations and loads from them.
  createStreamOps();

  scheduleDependencies();

  scheduleDistanceOneDependencies();

  scheduleRemainingToLastStage();

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  // Explicitly deallocate created allocations.
  for (auto alloc : sharedMemAllocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);

  return success();
}
LogicalResult StreamPipeliner::pipelineLoop() {
  if (failed(preprocessLoopAndBuildSchedule()))
    return failure();

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}

static bool checkPrecondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) { return !operand.getDefiningOp(); }))
    return false;

  auto hasInvalidOp = [forOp](Operation *op) {
    // Don't pipeline outer loops.
    if (op != forOp && isa<scf::ForOp, scf::WhileOp>(op))
      return WalkResult::interrupt();
    // Don't pipeline loops with barriers or asserts/prints.
    if (isa<gpu::BarrierOp, tt::AssertOp, tt::PrintOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  return !forOp->walk(hasInvalidOp).wasInterrupted();
}

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h.inc"

namespace {
struct PipelinePass : public TritonHCUStreamPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int32_t _numStages, int32_t _globalPrefetch,
               int32_t _localPrefetch, bool _useAsyncCopy) {
    this->numStages = _numStages;

    this->globalPrefetch = _globalPrefetch;
    this->localPrefetch = _localPrefetch;

    this->useAsyncCopy = _useAsyncCopy;
  }
  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (globalPrefetch < 0 || globalPrefetch >= numStages) {
      moduleOp.emitError("global prefetch control must be in [0, ")
          << numStages << "); " << globalPrefetch << " is out of range";
      return signalPassFailure();
    }
    if (localPrefetch < 0 || localPrefetch >= numStages) {
      moduleOp.emitError("local prefetch control must be in [0, ")
          << numStages << "); " << localPrefetch << " is out of range";
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!checkPrecondition(forOp))
        continue;
      StreamPipeliner sp(forOp, getNumStagesOrDefault(forOp), globalPrefetch,
                         localPrefetch, useAsyncCopy);
      (void)sp.pipelineLoop();
    }
    if (useAsyncCopy) {
      llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
      moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
      combineRedundantWaitOps(waitOps);
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonHCUStreamPipelinePass(int numStages, int globalPrefetch,
                                        int localPrefetch, bool useAsyncCopy) {
  return std::make_unique<PipelinePass>(numStages, globalPrefetch,
                                        localPrefetch, useAsyncCopy);
}
