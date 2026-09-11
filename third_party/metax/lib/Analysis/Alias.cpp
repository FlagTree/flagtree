#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/LLVM.h"
#ifdef __MCTLE__
#include "triton/Dialect/Triton/IR/Types.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#ifdef __MCTLE__
// mctle.local_pointers turns a shared buffer into plain !tt.ptr values, and
// every later tt.load/tt.store/tt.atomic_rmw reaches the buffer through
// those pointers. Without the two cases below the pointers carry no alias,
// so Allocation sees the buffer's last use at local_pointers itself and
// hands its bytes to reduction/scan scratch and to other local_alloc
// buffers -- measured on a C550: scan scratch at byte offsets 1..7 on top
// of a 2048xi32 histogram, and a whole 512xf32 buffer with no offset of its
// own (metadata.shared 2012 B short). Ported from lib/Analysis/Alias.cpp
// (#ifdef __TLE__, tle.local_pointers).
static bool isTritonPtrLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType>(tensorTy.getElementType());
  return false;
}
#endif

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

LogicalResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  auto result = op->getResult(0);
  // skip ops that return memdesc in a different memory space.
  if (auto memdescTy = dyn_cast<triton::gpu::MemDescType>(result.getType())) {
    if (!isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
            memdescTy.getMemorySpace()))
      return success();
  }

  // Only LocalAllocOp creates a new buffer.
  if (isa<triton::gpu::LocalAllocOp>(op)) {
    aliasInfo.insert(result);
    pessimistic = false;
  } else if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else if (isa<ub::PoisonOp>(op)) {
    aliasInfo = AliasInfo();
    pessimistic = false;
#ifdef __MCTLE__
  } else if (op->getName().getStringRef() == "mctle.local_pointers" &&
             !operands.empty()) {
    // Local pointer views alias their source memdesc.
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else if (isTritonPtrLikeType(result.getType())) {
    // Carry the alias through tt.splat / tt.broadcast / tt.addptr chains so
    // the buffer stays live across every pointer-arithmetic user.
    for (auto *operand : operands)
      aliasInfo = AliasInfo::join(aliasInfo, operand->getValue());
    pessimistic = false;
#endif
  } else {
    assert(!isa<triton::gpu::MemDescType>(result.getType()) &&
           "unknown operation creating memory descriptor");
  }

  if (pessimistic) {
    setAllToEntryStates(results);
    return success();
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));

  return success();
}

void SharedMemoryAliasAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<dataflow::Lattice<AliasInfo> *> argLattices, unsigned firstIndex) {
  auto wsOp = dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(op);
  if (!wsOp) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
    return;
  }

  // Propagate aliases from the parent operation's operands to the block
  // arguments.
  assert(!successor.isParent());
  ProgramPoint *point = getProgramPointAfter(wsOp);

  for (auto [capture, argLattice] :
       llvm::zip(wsOp.getParentOp().getExplicitCaptures(), argLattices)) {
    propagateIfChanged(
        argLattice,
        argLattice->join(getLatticeElementFor(point, capture)->getValue()));
  }
}

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}

} // namespace mlir
