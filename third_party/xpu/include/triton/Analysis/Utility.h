#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-analysis"

namespace mlir {

inline bool isZeroConst(Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  return false;
}

struct redSMOffsetInfo {
  int64_t startOffset; // bytes
  int64_t endOffset;   // bytes
  llvm::SmallVector<int64_t> offsets;

  redSMOffsetInfo() : startOffset(0), endOffset(0) {}

  redSMOffsetInfo(int64_t _startOffset, llvm::SmallVector<int64_t> &_offsets)
      : startOffset(_startOffset), endOffset(_startOffset), offsets(_offsets) {
    for (auto offset : offsets)
      endOffset += offset;
  }
};

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp op)
      : op(op.getOperation()), axis(op.getAxis()) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  ArrayRef<int64_t> getSrcShape() { return srcShape; }

  Attribute getSrcLayout() { return srcEncoding; }

  triton::ReduceOp getOperation() { return op; }

  bool isReductionOnLayoutFastAxis();

  unsigned getThreadOffsetOnReductionAxis();

  bool isWarpSynchronous();

  unsigned getInterWarpSize();

  unsigned getIntraWarpSize();

  unsigned getInterWarpSizeWithUniqueData();

  unsigned getIntraWarpSizeWithUniqueData();

  unsigned getThreadsReductionAxis();

  SmallVector<unsigned> getScratchConfig();

  SmallVector<unsigned> getOrderWithAxisAtBeginning();

  unsigned getScratchSizeInBytes();

  bool isSupportedLayout();

  bool isReduceWithinCTA();

  unsigned getAxis() { return axis; }

  //===-------------------- For Triton XPU -----------------------===//
  explicit ReduceOpHelper(triton::xpu::ReduceOp op)
      : xpu_op(op.getOperation()), axis(op.getAxis()) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &[i, t] : llvm::enumerate(op.getInputTypes())) {
      if (i == (op.getInputTypes().size() - 1))
        continue; // skip loopIndex
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  triton::xpu::ReduceOp getXPUOperation() { return xpu_op; }

  bool isCoreSynchronous();

  unsigned getIntraGroupSizeWithUniqueData();

  SmallVector<unsigned> getXPUScratchConfig();

  unsigned getXPUScratchSizeInBytes();

  void setReduceId(unsigned _reduceId) { reduceIdMap[xpu_op] = _reduceId; }

  unsigned getReduceId() { return reduceIdMap[xpu_op]; }

  void setReduceNum(unsigned _reduceNum) { reduceNum = _reduceNum; }

  unsigned getReduceNum() { return reduceNum; }

  void setSMOffsets(unsigned _reduceId, SmallVector<int64_t> &_offsets) {
    int64_t _startOffset;
    if (_reduceId == 0) {
      _startOffset = 0;
    } else {
      _startOffset = getSMOffsets(getReduceId() - 1)->endOffset;
    }
    reduceSMOffsetMap[_reduceId] =
        std::make_unique<redSMOffsetInfo>(_startOffset, _offsets);
  }

  redSMOffsetInfo *getSMOffsets(unsigned _reduceId) {
    auto it = reduceSMOffsetMap.find(_reduceId);
    if (it != reduceSMOffsetMap.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  void dumpSMOffsets() {
    LLVM_DEBUG({
      for (auto i = 0; i < reduceNum; ++i) {
        auto info = getSMOffsets(i);
        if (info == nullptr)
          continue;
        llvm::dbgs() << "\nreduceOp" << i << " [start, end] = ["
                     << info->startOffset << ", " << info->endOffset << "]\n";
        llvm::dbgs() << "detail offsets: [";
        for (auto offset : info->offsets) {
          llvm::dbgs() << offset << ",";
        }
        llvm::dbgs() << "]\n";
      }
    });
  }

  SmallVector<Operation *> getReturnDefOps() {
    SmallVector<Operation *> returnDefOps;
    for (Block &block : xpu_op.getCombineOp().getBlocks()) {
      triton::xpu::ReduceReturnOp returnOp =
          cast<triton::xpu::ReduceReturnOp>(block.getTerminator());
      for (auto operand : returnOp.getOperands()) {
        returnDefOps.emplace_back(operand.getDefiningOp());
      }
    }
    return returnDefOps;
  }

  bool isVectorized() {
    for (auto type : xpu_op.getInputTypes()) {
      if (!isa<VectorType>(getElementTypeOrSelf(type))) {
        return false;
      }
    }
    return true;
  }
  //===-----------------------------------------------------------===//

private:
  triton::ReduceOp op;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;

  //===-------------------- For Triton XPU -----------------------===//
  triton::xpu::ReduceOp xpu_op;
  static std::map<Operation *, unsigned> reduceIdMap;
  static unsigned reduceNum;
  static std::map<unsigned, std::unique_ptr<redSMOffsetInfo>> reduceSMOffsetMap;
  SmallVector<Operation *> returnDefOps;
  //===-----------------------------------------------------------===//
};

class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op) : scanOp(op) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }
  // Return true if the lowering of the scan op is supported.
  bool isSupported();
  // Return the number of elements per thread along axis dim.
  unsigned getAxisNumElementsPerThread();
  // Return the number of elements per thread along non-axis dims.
  unsigned getNonAxisNumElementsPerThread();
  // Return the number of threads per warp along non-axis dims.
  unsigned getNonAxisNumThreadsPerWarp();
  // Return the flat numbers of threads computing independent scan results.
  unsigned getNonAxisNumThreadsPerCTA();
  // Return the number of warps per CTA along axis dim.
  unsigned getAxisNumWarps();
  // Return the number of warps per CTA along axis dim with unique data.
  unsigned getAxisNumWarpsWithUniqueData();
  // Return the number of threads per warp along axis dim.
  unsigned getAxisNumThreadsPerWarp();
  // Return the number of threads per warp along axis dim with unique data.
  unsigned getAxisNumThreadsPerWarpWithUniqueData();
  // Return the number of blocks along axis dim.
  unsigned getAxisNumBlocks();
  // Return the number of blocks along non axis dim.
  unsigned getNonAxisNumBlocks();
  // Return the size of the scratch space needed for scan lowering.
  unsigned getScratchSizeInBytes();
  // Return the number of elements of the scratch space needed for scan
  // lowering.
  unsigned getScratchSizeInElems();

  // Stride between contiguous element along axis dim.
  unsigned getAxisElementStride();
  // Stride between contiguous threads along axis dim.
  unsigned getAxisThreadStride();
  // Stride between contiguous blocks along axis dim.
  unsigned getAxisBlockStride();

  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  bool getReverse() { return scanOp.getReverse(); }
  triton::gpu::BlockedEncodingAttr getEncoding();
  llvm::ArrayRef<int64_t> getShape() { return srcShape; }
  unsigned getNumOperands() { return scanOp.getNumOperands(); }
  SmallVector<Type> getElementTypes() { return srcElementTypes; }
  Attribute getSrcLayout() { return srcEncoding; }
  Region &getCombineOp();

  //===-------------------- For Triton XPU -----------------------===//
  explicit ScanLoweringHelper(triton::xpu::ScanOp op) : xpu_op(op) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  bool isCoreSynchronous();

  unsigned getIntraGroupSizeWithUniqueData();

  // Return the number of elements of the scratch space needed for scan
  // lowering.
  unsigned getScratchSizeInElemsXPU();

  void setScanId(unsigned _scanId) { scanIdMap[xpu_op] = _scanId; }

  unsigned getScanId() { return scanIdMap[xpu_op]; }

  void setScanNum(unsigned _scanNum) { scanNum = _scanNum; }

  unsigned getScanNum() { return scanNum; }

  Location getXPULoc() { return xpu_op.getLoc(); }

  unsigned getXPUAxis() { return xpu_op.getAxis(); }

  Region &getXPUCombineOp() { return xpu_op.getCombineOp(); };

  unsigned getXPUNumOperands() { return xpu_op.getNumOperands(); }

  void setSMOffsets(unsigned _scanId, SmallVector<int64_t> &_offsets) {
    int64_t _startOffset;
    if (_scanId == 0) {
      _startOffset = 0; // [TODO]: find reduceOp and replace the last endOffset
    } else {
      _startOffset = getSMOffsets(getScanId() - 1)->endOffset;
    }
    scanSMOffsetMap[_scanId] =
        std::make_unique<redSMOffsetInfo>(_startOffset, _offsets);
  }

  redSMOffsetInfo *getSMOffsets(unsigned _reduceId) {
    auto it = scanSMOffsetMap.find(_reduceId);
    if (it != scanSMOffsetMap.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  void dumpSMOffsets() {
    LLVM_DEBUG({
      for (auto i = 0; i < scanNum; ++i) {
        auto info = getSMOffsets(i);
        if (info == nullptr)
          continue;
        llvm::dbgs() << "\nreduceOp" << i << " [start, end] = ["
                     << info->startOffset << ", " << info->endOffset << "]\n";
        llvm::dbgs() << "detail offsets: [";
        for (auto offset : info->offsets) {
          llvm::dbgs() << offset << ",";
        }
        llvm::dbgs() << "]\n";
      }
    });
  }
  // Return the number of elements per thread along axis dim.
  unsigned getAxisNumElementsPerThreadXPU();

  triton::xpu::ClusterLayoutAttr getXPUEncoding();
  //===-----------------------------------------------------------===//

private:
  triton::ScanOp scanOp;
  Attribute srcEncoding;
  llvm::ArrayRef<int64_t> srcShape;
  SmallVector<Type> srcElementTypes;

  //===-------------------- For Triton XPU -----------------------===//
  triton::xpu::ScanOp xpu_op;
  static std::map<Operation *, unsigned> scanIdMap;
  static unsigned scanNum;
  static std::map<unsigned, std::unique_ptr<redSMOffsetInfo>> scanSMOffsetMap;
  //===-----------------------------------------------------------===//
};

// Decomposes a reshape into simpler pieces.
//
// As an example, suppose we have a reshape from [4,4,4] to [2,2,8,2].
// You might explain what this does as follows.
//
//  - Split the first input dimension into [2,2].
//  - Take the remaining two input dimensions, merge them into a single [16]
//    dim, and then split that into [8,2].
//
// In general, a reshape can be described a sequence of smushing one or more
// input dimensions together and then breaking them apart into one or more
// output dimensions.  So we could represent the example above as follows.
//
//   [
//     ([0], [0, 1]),  # input dim [0] -> output dims [0, 1]
//     ([1, 2], [2, 3]),  # input dims [1, 2] -> output dims [2, 3]
//   ]
//
// Notice that the input dims (first tuple elems) appear in sequential order if
// you read left-to-right-top-to-bottom, and so do the output dims.
//
// This function returns the above decomposition.
SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape);

bool maybeSharedAllocationOp(Operation *op);

bool supportMFMA(triton::DotOp op);

bool supportWMMA(triton::DotOp op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

bool isSingleValue(Value value);

bool isMfmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy);

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

// Return true if the src and dst layout match.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Operation in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort);

/// This uses the toplogicalSort above
SetVector<Operation *>
multiRootGetSlice(Operation *op, TransitiveFilter backwardFilter = nullptr,
                  TransitiveFilter forwardFilter = nullptr);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

/// This class represents a call graph for a given ModuleOp and holds
/// data of type T associated with each FunctionOpInterface.
template <typename T> class CallGraph {
public:
  using FuncDataMapT = DenseMap<FunctionOpInterface, T>;

  /// Constructor that builds the call graph for the given moduleOp.
  explicit CallGraph(ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  /// Walks the call graph and applies the provided update functions
  /// to the edges and nodes.
  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void walk(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    DenseSet<FunctionOpInterface> visited;
    for (auto root : roots) {
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
                                               updateNodeFn);
    }
  }

  /// Retrieves the data associated with a function
  T *getFuncData(FunctionOpInterface funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  /// Getters
  ModuleOp getModuleOp() const { return moduleOp; }
  SmallVector<FunctionOpInterface> getRoots() const { return roots; }
  size_t getNumFunctions() const { return funcMap.size(); }

  /// Returns true if the given function is a root.
  bool isRoot(FunctionOpInterface funcOp) const {
    return llvm::is_contained(roots, funcOp);
  }

  /// Maps the data and the graph nodes associated with a funcOp to a
  /// targetFuncOp.
  template <typename FROM, typename TO>
  void mapFuncOp(FROM funcOp, TO targetFuncOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.second == funcOp) {
          edge.second = targetFuncOp;
        }
      }
    }
    graph[targetFuncOp] = graph[funcOp];
    // Replace in roots
    for (auto it = roots.begin(); it != roots.end(); ++it) {
      if (*it == funcOp) {
        *it = targetFuncOp;
        break;
      }
    }
    // Replace in funcMap
    funcMap[targetFuncOp] = funcMap[funcOp];
  }

  /// Maps the graph edges associated with a callOp to a targetCallOp.
  template <typename FROM, typename TO>
  void mapCallOp(FROM callOp, TO targetCallOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.first == callOp) {
          edge.first = targetCallOp;
        }
      }
    }
  }

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseSet<FunctionOpInterface> visited;
    // Build graph
    moduleOp.walk([&](Operation *op) {
      auto caller = op->getParentOfType<FunctionOpInterface>();
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto *callee = callOp.resolveCallable(&symbolTable);
        auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callee);
        if (funcOp) {
          graph[caller].emplace_back(
              std::pair<CallOpInterface, FunctionOpInterface>(callOp, funcOp));
          visited.insert(funcOp);
        }
      }
    });
    // Find roots
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      if (!visited.count(funcOp)) {
        roots.push_back(funcOp);
      }
    });
  }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doWalk(FunctionOpInterface funcOp,
              DenseSet<FunctionOpInterface> &visited, UpdateEdgeFn updateEdgeFn,
              UpdateNodeFn updateNodeFn) {
    if (visited.count(funcOp)) {
      llvm::report_fatal_error("Cycle detected in call graph");
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PreOrder) {
      updateNodeFn(funcOp);
    }
    for (auto [callOp, callee] : graph[funcOp]) {
      if constexpr (UpdateEdgeOrder == WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(callee, visited, updateEdgeFn,
                                               updateNodeFn);
      if constexpr (UpdateEdgeOrder == WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PostOrder) {
      updateNodeFn(funcOp);
    }
    visited.erase(funcOp);
  }

protected:
  ModuleOp moduleOp;
  DenseMap<FunctionOpInterface,
           SmallVector<std::pair<CallOpInterface, FunctionOpInterface>>>
      graph;
  FuncDataMapT funcMap;
  SmallVector<FunctionOpInterface> roots;
};
// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

triton::MakeTensorPtrOp getMakeTensorPtrOp(Value v);

} // namespace mlir

#undef DEBUG_TYPE

#endif // TRITON_ANALYSIS_UTILITY_H
