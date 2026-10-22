#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace musa = mlir::triton::musa;
namespace musa_tle = mlir::triton::musa_tle;

namespace {

constexpr llvm::StringLiteral
    kAutoSharedLayoutAttr("musa_tle.auto_shared_layout");
constexpr llvm::StringLiteral kExplicitSqmmaAttr("musa_tle.explicit_sqmma");
constexpr llvm::StringLiteral kEnableEncodingRematerializationAttr(
    "tle.enable_encoding_rematerialization");

struct SelectedSqmmaConfig {
  SmallVector<unsigned, 3> instrShape;
  SmallVector<unsigned, 2> warpsPerCTA;
};

static std::optional<SelectedSqmmaConfig>
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  musa::SQMMAEltType operandType, bool preferRowReduction) {
  if (numWarps < 4 || numWarps % 4 != 0)
    return std::nullopt;
  const auto *sqTraits = musa::getMusaSqmmaArchTraits(musa::MusaArch::PH1);
  static constexpr unsigned kMN[] = {128, 64, 32, 16};
  static constexpr unsigned kK[] = {128, 64, 32, 16};

  std::optional<SelectedSqmmaConfig> best;
  unsigned bestCount = std::numeric_limits<unsigned>::max();
  unsigned bestVolume = 0;
  for (unsigned instM : kMN) {
    if (m % instM != 0)
      continue;
    for (unsigned instN : kMN) {
      if (n % instN != 0 ||
          !musa::isSupportedSqmmaInstrMN(operandType, instM, instN, *sqTraits))
        continue;
      for (unsigned instK : kK) {
        if (k % instK != 0 ||
            !musa::isSupportedSqmma(operandType, operandType,
                                    musa::SQMMAEltType::f32, instM, instN,
                                    instK, *sqTraits))
          continue;
        for (unsigned warpsM = 4; warpsM <= numWarps; warpsM *= 2) {
          if (numWarps % warpsM != 0)
            continue;
          unsigned warpsN = numWarps / warpsM;
          unsigned tileM = instM * (warpsM / 4);
          unsigned tileN = instN * warpsN;
          if (m % tileM != 0 || n % tileN != 0)
            continue;
          unsigned count = (m / tileM) * (n / tileN) * (k / instK);
          unsigned volume = instM * instN * instK;
          // On a full tie keep each row inside one squad for row reductions.
          bool preferMSplit = warpsN == 1 &&
                              (instM >= 32 || preferRowReduction) && best &&
                              best->warpsPerCTA[1] != 1;
          if (!best || count < bestCount ||
              (count == bestCount &&
               (volume > bestVolume ||
                (volume == bestVolume && preferMSplit)))) {
            best = SelectedSqmmaConfig{{instM, instN, instK}, {warpsM, warpsN}};
            bestCount = count;
            bestVolume = volume;
          }
        }
      }
    }
  }
  return best;
}

static ttg::CGAEncodingAttr prependBufferDim(ttg::CGAEncodingAttr cgaLayout) {
  auto prependOne = [](SmallVector<unsigned> values) {
    values.insert(values.begin(), 1);
    return values;
  };
  SmallVector<unsigned> order{0};
  for (unsigned dim : cgaLayout.getCTAOrder())
    order.push_back(dim + 1);
  return ttg::CGAEncodingAttr::fromSplitParams(
      cgaLayout.getContext(), prependOne(cgaLayout.getCTAsPerCGA()),
      prependOne(cgaLayout.getCTASplitNum()), order);
}

static ttg::SwizzledSharedEncodingAttr
prependBufferDim(ttg::SwizzledSharedEncodingAttr encoding) {
  SmallVector<unsigned> order;
  for (unsigned dim : encoding.getOrder())
    order.push_back(dim + 1);
  order.push_back(0);
  return ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), encoding.getVec(), encoding.getPerPhase(),
      encoding.getMaxPhase(), order, prependBufferDim(encoding.getCGALayout()));
}

static ttg::LocalAllocOp findRootAlloc(Value value) {
  llvm::SmallPtrSet<void *, 16> visited;
  while (value && visited.insert(value.getAsOpaquePointer()).second) {
    if (auto alloc = value.getDefiningOp<ttg::LocalAllocOp>())
      return alloc;
    // A worker's shared descriptor is an explicit capture, not a new
    // allocation. Follow it back across the isolated partition boundary.
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
          argument.getOwner()->getParentOp());
      if (!partitions ||
          argument.getArgNumber() >= partitions.getExplicitCaptures().size())
        break;
      value = partitions.getExplicitCaptures()[argument.getArgNumber()];
      continue;
    }
    Operation *def = value.getDefiningOp();
    if (auto index = dyn_cast_or_null<ttg::MemDescIndexOp>(def))
      value = index.getSrc();
    else if (auto subslice = dyn_cast_or_null<ttg::MemDescSubsliceOp>(def))
      value = subslice.getSrc();
    else if (auto reinterpret =
                 dyn_cast_or_null<ttg::MemDescReinterpretOp>(def))
      value = reinterpret.getSrc();
    else if (auto reshape = dyn_cast_or_null<ttg::MemDescReshapeOp>(def))
      value = reshape.getSrc();
    else if (auto trans = dyn_cast_or_null<ttg::MemDescTransOp>(def))
      value = trans.getSrc();
    else
      break;
  }
  return {};
}

// Analyze the unchanged graph before selecting or rewriting any SQMMA.
class RowReductionAnalysis {
  struct SharedFlow {
    unsigned writers = 0;
    Value value;
    SmallVector<Value> loads;
    SmallVector<musa_tle::SqmmaOp> lhsDots;
    bool ambiguous = false;
  };
  DenseMap<Operation *, SharedFlow> shared;

  // Only a complete, untransposed single-tile allocation proves this memory
  // edge. Multi-stage aliases and partial accesses stay on the default policy.
  static bool isWholeView(Value value) {
    auto root = findRootAlloc(value);
    if (!root)
      return false;
    auto nonUnitShape = [](ttg::MemDescType type) {
      SmallVector<int64_t> shape;
      for (int64_t dim : type.getShape())
        if (dim != 1)
          shape.push_back(dim);
      return shape;
    };
    if (nonUnitShape(cast<ttg::MemDescType>(root.getType())) !=
        nonUnitShape(cast<ttg::MemDescType>(value.getType())))
      return false;
    while (value != root.getResult()) {
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        auto part = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
            arg.getOwner()->getParentOp());
        if (!part || arg.getArgNumber() >= part.getExplicitCaptures().size())
          return false;
        value = part.getExplicitCaptures()[arg.getArgNumber()];
      } else if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
        auto src = cast<ttg::MemDescType>(index.getSrc().getType());
        if (src.getShape().front() != 1 ||
            !matchPattern(index.getIndex(), m_Zero()))
          return false;
        value = index.getSrc();
      } else if (auto slice = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
        if (llvm::any_of(slice.getOffsets(), [](int32_t x) { return x != 0; }))
          return false;
        value = slice.getSrc();
      } else {
        return false;
      }
    }
    return true;
  }

  // Preserve the complete row dimension through supported tensor views.
  static std::optional<unsigned> getRowAxis(Operation *op, Value input,
                                            unsigned rowAxis) {
    auto src = cast<RankedTensorType>(input.getType());
    if (auto trans = dyn_cast<tt::TransOp>(op)) {
      auto order = trans.getOrder();
      auto found = llvm::find(order, rowAxis);
      if (found != order.end())
        return std::distance(order.begin(), found);
    } else if (auto reshape = dyn_cast<tt::ReshapeOp>(op)) {
      auto dst = cast<RankedTensorType>(reshape.getType());
      for (const auto &[srcDims, dstDims] :
           getReshapeDecomposition(src.getShape(), dst.getShape())) {
        if (!llvm::is_contained(srcDims, rowAxis))
          continue;
        if (llvm::any_of(srcDims, [&](int64_t dim) {
              return dim != rowAxis && src.getDimSize(dim) != 1;
            }))
          return std::nullopt;
        std::optional<unsigned> result;
        for (int64_t dim : dstDims) {
          if (dst.getDimSize(dim) == 1)
            continue;
          if (result || dst.getDimSize(dim) != src.getDimSize(rowAxis))
            return std::nullopt;
          result = dim;
        }
        return result;
      }
    } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(op)) {
      return rowAxis + (expand.getAxis() <= rowAxis);
    } else if (isa<tt::SplitOp>(op)) {
      if (rowAxis + 1 < src.getRank())
        return rowAxis;
    } else if (isa<tt::JoinOp, tt::BroadcastOp, ttg::ConvertLayoutOp,
                   musa_tle::SqmmaWaitOp>(op) ||
               (op->hasTrait<OpTrait::Elementwise>() &&
                isMemoryEffectFree(op)) ||
               (isa<tt::ExternElementwiseOp>(op) &&
                cast<tt::ExternElementwiseOp>(op).getPure())) {
      return rowAxis;
    }
    return std::nullopt;
  }

public:
  explicit RowReductionAnalysis(ModuleOp module) {
    auto addWriter = [&](Value dst, Value value) {
      auto root = findRootAlloc(dst);
      if (!root)
        return;
      auto &flow = shared[root];
      ++flow.writers;
      flow.value = value;
      flow.ambiguous |=
          !isWholeView(dst) || bool(getTleExplicitMemoryEncoding(root));
    };
    module.walk([&](ttg::LocalAllocOp alloc) {
      if (alloc.getSrc())
        addWriter(alloc.getResult(), alloc.getSrc());
    });
    module.walk([&](ttg::LocalStoreOp store) {
      addWriter(store.getDst(), store.getSrc());
    });
    module.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (!isa<ttg::MemDescType>(operand.getType()))
          continue;
        auto root = findRootAlloc(operand);
        auto found = shared.find(root);
        if (!root || found == shared.end())
          continue;
        auto &flow = found->second;
        auto valueTy = cast<RankedTensorType>(flow.value.getType());
        auto sameView = [&]() {
          auto type = cast<ttg::MemDescType>(operand.getType());
          return isWholeView(operand) &&
                 type.getShape() == valueTy.getShape() &&
                 type.getElementType() == valueTy.getElementType();
        };
        if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
          flow.ambiguous |= !sameView();
          flow.loads.push_back(load.getResult());
        } else if (auto dot = dyn_cast<musa_tle::SqmmaOp>(op)) {
          flow.ambiguous |= operand != dot.getA() || !sameView();
          if (operand == dot.getA())
            flow.lhsDots.push_back(dot);
        } else if (isa<ttg::LocalStoreOp, ttg::MemDescIndexOp,
                       ttg::MemDescSubsliceOp, ttg::WarpSpecializePartitionsOp,
                       ttg::WarpSpecializeOp>(op)) {
          continue;
        } else if (auto pointers = dyn_cast<musa_tle::LocalPointersOp>(op)) {
          flow.ambiguous |= !pointers.getResult().use_empty();
        } else {
          flow.ambiguous = true;
        }
      }
    });
  }

  bool matches(musa_tle::SqmmaOp producer) {
    if (getTleExplicitValueEncoding(producer.getC()) ||
        getTleExplicitSqmmaEncoding(producer))
      return false;
    for (Value operand : {producer.getA(), producer.getB()}) {
      auto root = findRootAlloc(operand);
      if (root && getTleExplicitMemoryEncoding(root))
        return false;
    }
    struct State {
      Value value;
      unsigned rowAxis;
      bool reduced;
    };
    SmallVector<State> worklist;
    DenseMap<Value, unsigned> rowAxes;
    DenseMap<Value, unsigned> visited;
    bool conflict = false;
    bool matched = false;
    auto enqueue = [&](Value value, unsigned axis, bool reduced) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || axis >= type.getRank() ||
          isa<tt::PointerType>(type.getElementType())) {
        conflict = true;
        return;
      }
      auto [it, inserted] = rowAxes.try_emplace(value, axis);
      if (!inserted && it->second != axis) {
        conflict = true;
        return;
      }
      unsigned bit = reduced ? 2 : 1;
      if (visited[value] & bit)
        return;
      visited[value] |= bit;
      worklist.push_back({value, axis, reduced});
    };
    enqueue(producer.getD(), 0, false);
    while (!worklist.empty() && !conflict) {
      State state = worklist.pop_back_val();
      for (OpOperand &use : state.value.getUses()) {
        Operation *user = use.getOwner();
        for (Value operand : user->getOperands())
          conflict |= bool(getTleExplicitValueEncoding(operand));
        for (unsigned i = 0; i < user->getNumResults(); ++i)
          conflict |= bool(getTleExplicitResultEncoding(user, i));
        if (auto dot = dyn_cast<musa_tle::SqmmaOp>(user)) {
          if (use.getOperandNumber() == 2) {
            enqueue(dot.getD(), state.rowAxis, state.reduced);
          } else if (use.getOperandNumber() == 0 && state.rowAxis == 0) {
            matched |= state.reduced;
          } else {
            conflict = true;
          }
        } else if (auto dot = dyn_cast<tt::DotOpInterface>(user)) {
          if (dot.getA() == state.value && state.rowAxis == 0)
            matched |= state.reduced;
          else
            conflict = true;
        } else if (auto reduce = dyn_cast<tt::ReduceOp>(user)) {
          if (reduce.getAxis() == state.rowAxis) {
            conflict = true;
            break;
          }
          unsigned axis = state.rowAxis - (reduce.getAxis() < state.rowAxis);
          for (Value result : reduce.getResults())
            enqueue(result, axis, true);
        } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          unsigned index = use.getOperandNumber();
          if (auto loop = dyn_cast<scf::ForOp>(yield->getParentOp())) {
            enqueue(loop.getRegionIterArg(index), state.rowAxis, state.reduced);
            enqueue(loop.getResult(index), state.rowAxis, state.reduced);
          } else if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp())) {
            enqueue(branch.getResult(index), state.rowAxis, state.reduced);
          } else {
            conflict = true;
          }
        } else if (auto loop = dyn_cast<scf::ForOp>(user)) {
          unsigned index = use.getOperandNumber();
          if (index < 3) {
            conflict = true;
            break;
          }
          enqueue(loop.getRegionIterArg(index - 3), state.rowAxis,
                  state.reduced);
          enqueue(loop.getResult(index - 3), state.rowAxis, state.reduced);
        } else if (isa<ttg::LocalStoreOp, ttg::LocalAllocOp>(user)) {
          Value dst = isa<ttg::LocalStoreOp>(user)
                          ? Value(cast<ttg::LocalStoreOp>(user).getDst())
                          : Value(user->getResult(0));
          auto root = findRootAlloc(dst);
          auto found = shared.find(root);
          if (!root || found == shared.end() || found->second.writers != 1 ||
              found->second.ambiguous) {
            conflict = true;
            break;
          }
          const auto &flow = found->second;
          for (Value load : flow.loads)
            enqueue(load, state.rowAxis, state.reduced);
          for (musa_tle::SqmmaOp dot : flow.lhsDots) {
            if (state.rowAxis != 0)
              conflict = true;
            else
              matched |= state.reduced;
          }
        } else if (isa<tt::StoreOp>(user) && use.getOperandNumber() == 1) {
          continue;
        } else {
          auto axis = getRowAxis(user, state.value, state.rowAxis);
          if (!axis) {
            conflict = true;
            break;
          }
          for (Value result : user->getResults()) {
            auto type = dyn_cast<RankedTensorType>(result.getType());
            auto src = cast<RankedTensorType>(state.value.getType());
            if (!type || *axis >= type.getRank() ||
                type.getDimSize(*axis) != src.getDimSize(state.rowAxis)) {
              conflict = true;
              break;
            }
            enqueue(result, *axis, state.reduced);
          }
        }
      }
    }
    return matched && !conflict;
  }
};

static DenseSet<musa_tle::SqmmaOp>
findRowReductionDots(ModuleOp module, ArrayRef<musa_tle::SqmmaOp> dots) {
  RowReductionAnalysis analysis(module);
  DenseSet<musa_tle::SqmmaOp> result;
  for (auto dot : dots)
    if (analysis.matches(dot))
      result.insert(dot);
  // All directly accumulated dots must agree on the policy for their chain.
  bool changed;
  do {
    changed = false;
    for (auto dot : dots) {
      Value acc = dot.getC();
      while (true) {
        if (auto wait = acc.getDefiningOp<musa_tle::SqmmaWaitOp>())
          acc = wait.getInput();
        else if (auto convert = acc.getDefiningOp<ttg::ConvertLayoutOp>())
          acc = convert.getSrc();
        else
          break;
      }
      if (auto previous = acc.getDefiningOp<musa_tle::SqmmaOp>()) {
        if (result.contains(dot) != result.contains(previous)) {
          result.erase(dot);
          result.erase(previous);
          changed = true;
        }
      }
    }
  } while (changed);
  return result;
}

static musa::SQMMALayout inferLayout(ttg::MemDescType type) {
  auto order = ttg::getOrder(type);
  bool rowMajor = !order.empty() && order.front() + 1 == type.getRank();
  return rowMajor ? musa::SQMMALayout::row : musa::SQMMALayout::col;
}

static FailureOr<ttg::MemDescType>
inferTransposeSourceType(ttg::MemDescTransOp trans, ttg::MemDescType targetTy) {
  auto sourceTy = dyn_cast<ttg::MemDescType>(trans.getSrc().getType());
  Attribute targetEncoding = targetTy.getEncoding();
  ArrayRef<int32_t> order = trans.getOrder();
  if (!sourceTy || !targetEncoding || order.size() != targetTy.getRank() ||
      order.size() != sourceTy.getRank())
    return failure();

  SmallVector<int32_t> inverseOrder(order.size());
  SmallVector<int64_t> expectedSourceShape(order.size());
  for (auto [targetDim, sourceDim] : llvm::enumerate(order)) {
    if (sourceDim < 0 || static_cast<size_t>(sourceDim) >= order.size())
      return failure();
    inverseOrder[sourceDim] = targetDim;
    expectedSourceShape[sourceDim] = targetTy.getShape()[targetDim];
  }
  if (expectedSourceShape != sourceTy.getShape())
    return failure();

  Dialect &dialect = targetEncoding.getDialect();
  auto inferLayoutInterface =
      dyn_cast<tt::DialectInferLayoutInterface>(&dialect);
  if (!inferLayoutInterface)
    return failure();

  Attribute sourceEncoding;
  if (failed(inferLayoutInterface->inferTransOpEncoding(
          targetEncoding, targetTy.getShape(), inverseOrder, sourceEncoding,
          trans.getLoc())))
    return failure();

  SmallVector<int64_t> sourceAllocShape;
  ArrayRef<int64_t> targetAllocShape = targetTy.getAllocShape();
  if (!targetAllocShape.empty()) {
    if (targetAllocShape.size() < order.size())
      return failure();
    size_t prefixSize = targetAllocShape.size() - order.size();
    sourceAllocShape.append(targetAllocShape.begin(),
                            targetAllocShape.begin() + prefixSize);
    SmallVector<int64_t> sourceTail(order.size());
    ArrayRef<int64_t> targetTail = targetAllocShape.take_back(order.size());
    for (auto [targetDim, sourceDim] : llvm::enumerate(order))
      sourceTail[sourceDim] = targetTail[targetDim];
    sourceAllocShape.append(sourceTail.begin(), sourceTail.end());
  }

  return ttg::MemDescType::get(sourceTy.getShape(), sourceTy.getElementType(),
                               sourceEncoding, sourceTy.getMemorySpace(),
                               sourceTy.getMutableMemory(), sourceAllocShape);
}

static LogicalResult updateOperandLayout(
    musa_tle::SqmmaOp op, unsigned operandIdx,
    ttg::MUSASqmmaEncodingAttr mmaEnc,
    DenseMap<Operation *, ttg::MemDescType> &physicalRootTypes) {
  Value operand = operandIdx == 0 ? op.getA() : op.getB();
  auto operandTy = cast<ttg::MemDescType>(operand.getType());
  int64_t elemBytes =
      std::max<int64_t>(1, (operandTy.getElementTypeBitWidth() + 7) / 8);
  ttg::LocalAllocOp root = findRootAlloc(operand);
  if (!root)
    return op.emitOpError("requires SQMMA operands rooted at ttg.local_alloc");
  if (!root->hasAttr(kAutoSharedLayoutAttr))
    return op.emitOpError(
        "requires layout=None and nv_mma_shared_layout=True for initial "
        "mthreads TLE SQMMA operands");

  auto order = ttg::getOrder(operandTy);
  auto cga = ttg::getCGALayout(operandTy.getEncoding());
  auto shared = mmaEnc.composeSharedLayoutForOperand(
      cga, operandIdx, operandTy.getShape(), order,
      /*kWidth=*/0, operandTy.getElementTypeBitWidth(),
      /*needTrans=*/false);

  auto desiredTy = ttg::MemDescType::get(
      operandTy.getShape(), operandTy.getElementType(), shared,
      operandTy.getMemorySpace(), operandTy.getMutableMemory(),
      operandTy.getAllocShape());

  // trans_a/trans_b are represented by a descriptor-only memdesc_trans view.
  // Select the SQMMA layout for the final logical view, then propagate that
  // layout backwards so TME still writes a single physical landing buffer.
  Value landing = operand;
  ttg::MemDescType landingTy = desiredTy;
  if (auto trans = operand.getDefiningOp<ttg::MemDescTransOp>()) {
    auto sourceTy = inferTransposeSourceType(trans, desiredTy);
    if (failed(sourceTy))
      return op.emitOpError(
                 "failed to infer the physical SQMMA landing type through "
                 "ttg.memdesc_trans for operand ")
             << operandIdx;
    landing = trans.getSrc();
    landingTy = *sourceTy;
  }

  if (failed(musa::resolveTMESwizzleConfigFromEncoding(landingTy)))
    return op.emitOpError(
               "inferred physical SQMMA landing layout is not uniquely "
               "TME-compatible for operand ")
           << operandIdx;

  auto rootTy = cast<ttg::MemDescType>(root.getType());
  auto landingEncoding =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(landingTy.getEncoding());
  if (!landingEncoding)
    return op.emitOpError("requires a swizzled physical SQMMA landing layout");
  Attribute rootEncoding = landingEncoding;
  if (rootTy.getRank() == landingTy.getRank() + 1)
    rootEncoding = prependBufferDim(landingEncoding);
  else if (rootTy.getRank() != landingTy.getRank())
    return op.emitOpError("unsupported staged SQMMA allocation rank");

  auto newRootTy =
      ttg::MemDescType::get(rootTy.getShape(), rootTy.getElementType(),
                            rootEncoding, rootTy.getMemorySpace(),
                            rootTy.getMutableMemory(), rootTy.getAllocShape());
  auto [rootIt, inserted] =
      physicalRootTypes.try_emplace(root.getOperation(), newRootTy);
  if (!inserted && rootIt->second != newRootTy)
    return op.emitOpError(
        "SQMMA consumers require incompatible physical shared layouts");
  root.getResult().setType(newRootTy);
  landing.setType(landingTy);
  operand.setType(desiredTy);

  // Explicit warp-specialize captures are represented by operands on the
  // isolated partitions container and corresponding block arguments in every
  // partition. Changing the captured root value type does not update those
  // block arguments automatically, so keep the isolation boundary consistent
  // with the inferred TME/SQMMA layout.
  SmallVector<Value, 4> rootAliases{root.getResult()};
  root->getParentOfType<tt::FuncOp>().walk(
      [&](ttg::WarpSpecializePartitionsOp partitions) {
        for (auto [index, capture] :
             llvm::enumerate(partitions.getExplicitCaptures())) {
          if (capture != root.getResult())
            continue;
          for (Region &partition : partitions.getPartitionRegions()) {
            partition.getArgument(index).setType(newRootTy);
            rootAliases.push_back(partition.getArgument(index));
          }
        }
      });

  bool rowMajor = inferLayout(desiredTy) == musa::SQMMALayout::row;
  auto checkAndSet = [&](Operation *target) -> LogicalResult {
    if (!target)
      return success();
    if (auto oldIdx = musa::getSqmmaOpIdx(target)) {
      auto oldBytes = musa::getSqmmaElemBytes(target);
      bool oldRow = musa::getSqmmaRowMajor(target, rowMajor);
      if (*oldIdx != static_cast<int64_t>(operandIdx) || !oldBytes ||
          *oldBytes != elemBytes || oldRow != rowMajor)
        return target->emitOpError("conflicting TLE SQMMA consumer contract");
    }
    musa::setSqmmaAttrs(target, operandIdx, elemBytes, rowMajor);
    return success();
  };
  // The final view owns the logical operand contract. A staged allocation can
  // serve both K^T and V when their physical shared layouts are identical.
  // Propagating that view's row/column contract to the root would reject this
  // valid reuse even though no data needs to move.
  if (failed(checkAndSet(operand.getDefiningOp())))
    return failure();

  // Keep stage views and warp-specialize explicit captures type-consistent.
  bool changed = true;
  while (changed) {
    changed = false;
    root->getParentOfType<tt::FuncOp>().walk([&](ttg::MemDescIndexOp index) {
      auto srcTy = cast<ttg::MemDescType>(index.getSrc().getType());
      auto dstTy = cast<ttg::MemDescType>(index.getType());
      if (srcTy.getRank() != dstTy.getRank() + 1 ||
          !llvm::is_contained(rootAliases, index.getSrc()))
        return;
      if (index.getResult().getType() != landingTy) {
        index.getResult().setType(landingTy);
        changed = true;
      }
    });
  }
  return success();
}

static void setLoopAccumulator(scf::ForOp loop, unsigned index, Value init,
                               DenseMap<Value, Value> &nativeAccumulators) {
  loop->setOperand(3 + index, init);
  Value argument = loop.getRegionIterArg(index);
  Value result = loop.getResult(index);
  argument.setType(init.getType());
  result.setType(init.getType());
  nativeAccumulators[argument] = argument;
  nativeAccumulators[result] = result;
}

static LogicalResult verifyAsyncUses(musa_tle::SqmmaOp op) {
  SmallVector<Value> pending{op.getD()};
  DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<musa_tle::SqmmaOp>(user) && use.getOperandNumber() == 2)
        continue;
      if (isa<musa_tle::SqmmaWaitOp>(user))
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        if (use.getOperandNumber() >= 3) {
          unsigned index = use.getOperandNumber() - 3;
          pending.push_back(loop.getRegionIterArg(index));
          pending.push_back(loop.getResult(index));
          continue;
        }
      }
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(yield->getParentOp())) {
          unsigned index = use.getOperandNumber();
          pending.push_back(loop.getRegionIterArg(index));
          pending.push_back(loop.getResult(index));
          continue;
        }
      }
      return op.emitOpError(
          "async result must be consumed by musa_tle.sqmma_wait before "
          "ordinary tensor use");
    }
  }
  return success();
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERSQMMA
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUTLELowerSqmmaPass
    : impl::TritonMUSAGPUTLELowerSqmmaBase<TritonMUSAGPUTLELowerSqmmaPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<musa_tle::SqmmaOp> dots;
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *candidate) {
      if (auto dot = dyn_cast<musa_tle::SqmmaOp>(candidate))
        dots.push_back(dot);
      if (isa<musa_tle::SqmmaOp, musa_tle::SqmmaWaitOp>(candidate))
        worklist.push_back(candidate);
    });
    if (!dots.empty())
      module->setAttr(kEnableEncodingRematerializationAttr,
                      UnitAttr::get(module.getContext()));

    auto rowReductionDots = findRowReductionDots(module, dots);
    DenseMap<musa_tle::SqmmaOp, ttg::MUSASqmmaEncodingAttr> encodings;
    for (musa_tle::SqmmaOp dot : dots) {
      if (failed(verifyAsyncUses(dot)))
        return signalPassFailure();
      auto aTy = cast<ttg::MemDescType>(dot.getA().getType());
      auto bTy = cast<ttg::MemDescType>(dot.getB().getType());
      auto operandType = musa::getWmmaEltType(aTy.getElementType());
      if (!operandType) {
        dot.emitOpError("cannot map operand dtype to an SQMMA element type");
        return signalPassFailure();
      }
      unsigned numWarps = ttg::maybeLookupNumWarps(dot).value_or(0);
      bool preferRowReduction = rowReductionDots.contains(dot);
      auto config = selectSqmmaConfig(aTy.getShape()[0], bTy.getShape()[1],
                                      aTy.getShape()[1], numWarps, *operandType,
                                      preferRowReduction);
      if (!config) {
        dot.emitOpError("cannot select a supported SQMMA configuration");
        return signalPassFailure();
      }
      auto accTy = cast<RankedTensorType>(dot.getC().getType());
      auto cga = ttg::getCGALayout(accTy.getEncoding());
      auto mmaEnc = ttg::MUSASqmmaEncodingAttr::get(
          dot.getContext(), 3, 1, config->warpsPerCTA, cga, config->instrShape);
      encodings[dot] = mmaEnc;
    }

    // Infer all physical layouts before replacing any TLE SQMMA op.
    DenseMap<Operation *, ttg::MemDescType> physicalRootTypes;
    for (musa_tle::SqmmaOp dot : dots) {
      if (failed(
              updateOperandLayout(dot, 0, encodings[dot], physicalRootTypes)) ||
          failed(
              updateOperandLayout(dot, 1, encodings[dot], physicalRootTypes)))
        return signalPassFailure();
    }

    DenseMap<Value, Value> nativeAccumulators;
    for (Operation *candidate : worklist) {
      OpBuilder builder(candidate);
      if (auto dot = dyn_cast<musa_tle::SqmmaOp>(candidate)) {
        auto oldAccTy = cast<RankedTensorType>(dot.getC().getType());
        auto nativeTy = RankedTensorType::get(
            oldAccTy.getShape(), oldAccTy.getElementType(), encodings[dot]);
        Value nativeAcc = nativeAccumulators.lookup(dot.getC());
        // An async loop carries the encoded registers through its iter_arg,
        // yield and result; only a final wait releases them for tensor use.
        if (!nativeAcc) {
          if (auto argument = dyn_cast<BlockArgument>(dot.getC())) {
            auto loop =
                dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
            if (loop && argument.getArgNumber() >= loop.getNumInductionVars()) {
              unsigned index =
                  argument.getArgNumber() - loop.getNumInductionVars();
              auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
              if (yield.getOperand(index).getDefiningOp<musa_tle::SqmmaOp>()) {
                OpBuilder initBuilder(loop);
                Value init =
                    nativeAccumulators.lookup(loop.getInitArgs()[index]);
                if (!init)
                  init = ttg::ConvertLayoutOp::create(
                      initBuilder, loop.getLoc(), nativeTy,
                      loop.getInitArgs()[index]);
                setLoopAccumulator(loop, index, init, nativeAccumulators);
                nativeAcc = argument;
              }
            }
          }
        }
        if (nativeAcc) {
          auto carriedTy = cast<RankedTensorType>(nativeAcc.getType());
          auto carriedEncoding =
              dyn_cast<ttg::MUSASqmmaEncodingAttr>(carriedTy.getEncoding());
          if (!carriedEncoding ||
              carriedEncoding.toLinearLayout(carriedTy.getShape()) !=
                  encodings[dot].toLinearLayout(nativeTy.getShape())) {
            dot.emitOpError("requires compatible C register layouts for async "
                            "accumulator reuse");
            return signalPassFailure();
          }
          // C ownership is independent of K. Keep the producer's encoding;
          // the current operands selected this instruction's K above.
          nativeTy = carriedTy;
        }
        if (!nativeAcc)
          nativeAcc = dot.getC();
        // An explicit SQMMA layout can describe another instruction shape.
        // Being MMA-encoded alone does not make its register distribution
        // compatible with this dot's selected accumulator type.
        if (nativeAcc.getType() != nativeTy)
          nativeAcc = ttg::ConvertLayoutOp::create(builder, dot.getLoc(),
                                                   nativeTy, nativeAcc);
        Value useC = arith::ConstantIntOp::create(builder, dot.getLoc(), 1, 1);
        auto carriedEncoding =
            cast<ttg::MUSASqmmaEncodingAttr>(nativeTy.getEncoding());
        auto instr = carriedEncoding.getInstrShape();
        unsigned instructionK = encodings[dot].getInstrShape()[2];
        auto operandType = musa::getWmmaEltType(
            cast<ttg::MemDescType>(dot.getA().getType()).getElementType());
        assert(operandType && "SQMMA operand type must be verified");
        auto nativeDot = musa::SquadDotOp::create(
            builder, dot.getLoc(), nativeTy, dot.getA(), dot.getB(), nativeAcc,
            useC, instr[0], instr[1], instructionK, musa::SQMMAEltType::f32,
            *operandType, *operandType,
            inferLayout(cast<ttg::MemDescType>(dot.getA().getType())),
            inferLayout(cast<ttg::MemDescType>(dot.getB().getType())), true,
            musa::SQMMAAccumulationMode::hardware,
            static_cast<int32_t>(dot.getInputPrecision()), 0);
        nativeDot->setAttr(kExplicitSqmmaAttr, builder.getUnitAttr());
        nativeAccumulators[dot.getD()] = nativeDot.getD();
        for (OpOperand &use :
             llvm::make_early_inc_range(dot.getD().getUses())) {
          if (auto loop = dyn_cast<scf::ForOp>(use.getOwner())) {
            if (use.getOperandNumber() >= 3)
              setLoopAccumulator(loop, use.getOperandNumber() - 3,
                                 nativeDot.getD(), nativeAccumulators);
          } else if (isa<scf::YieldOp>(use.getOwner())) {
            use.set(nativeDot.getD());
          }
        }
        continue;
      }

      auto wait = cast<musa_tle::SqmmaWaitOp>(candidate);
      Value nativeInput = nativeAccumulators.lookup(wait.getInput());
      if (!nativeInput) {
        wait.emitOpError("input must be the async result of musa_tle.sqmma");
        return signalPassFailure();
      }
      auto nativeWait =
          musa::SquadDotWaitOp::create(builder, wait.getLoc(), nativeInput);
      nativeWait->setAttr(kExplicitSqmmaAttr, builder.getUnitAttr());
      Value released = ttg::ConvertLayoutOp::create(builder, wait.getLoc(),
                                                    wait.getOutput().getType(),
                                                    nativeWait.getResult(0));
      if (Attribute explicitEncoding =
              getTleExplicitValueEncoding(wait.getOutput()))
        setTleExplicitResultEncoding(released.getDefiningOp(), 0,
                                     explicitEncoding);
      wait.getOutput().replaceAllUsesWith(released);
      wait.erase();
    }

    for (musa_tle::SqmmaOp dot : llvm::reverse(dots))
      dot.erase();
  }
};

} // namespace mlir

#endif // __TLE__
