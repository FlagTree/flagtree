#include "BufferOpsEmitter.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;
using namespace triton::HCU;

namespace {

// Utility function to determine if a scalar/tensor value is zero
bool isZero(Value v) {
  if (auto constantOp = v.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue()))
      return attr.getValue().isZero();
    if (auto attr = dyn_cast<FloatAttr>(constantOp.getValue()))
      return attr.getValue().isZero();
    if (auto denseAttr =
            dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
      return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
    if (auto denseAttr =
            dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
      return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  }
  return false;
}
} // namespace

namespace mlir::LLVM::HCU {
BufferEmitter::BufferEmitter(RewriterBase &rw, Location loc, TargetInfo ti)
    : rewriter(rw), loc(loc), targetInfo(ti) {}

Value BufferEmitter::createResourceDescriptor(Value basePtr) {
  // 1. Create the resource descriptor
  // bits 0-11: dst sel, ignored by these intrinsics
  // bits 12-14: data format (ignored, must be nonzero, 7=float)
  // bits 15-18: data format (ignored, must be nonzero, 4=32bit)
  // bit 19: In nested heap (0 here)
  // bit 20: Behavior on unmap (0 means  "return 0 / ignore")
  // bits 21-22: Index stride for swizzles (N/A)
  // bit 23: Add thread ID (0)
  // bit 24: Reserved to 1 (RDNA) or 0 (CDNA)
  // bits 25-26: Reserved (0)
  // bit 27: Buffer is non-volatile (CDNA only)
  // bits 28-29: Out of bounds select (RDNA only)
  //             (0 = structured,
  //              1 = check index,
  //              2 = none,
  //              3 = either swizzles or testing against offset field)
  // bits 30-31: Type (must be 0)
  uint32_t flags = (7 << 12) | (4 << 15);
  if (targetInfo.getISAFamily() == ISAFamily::RDNA2 ||
      targetInfo.getISAFamily() == ISAFamily::RDNA3) {
    flags |= (1 << 24);
    uint32_t oob = 3;
    flags |= (oob << 28);
  }
  Value stride = int_val(16, 0);
  Value flagsConst = int_val(32, flags);
  Type rsrcType = LLVM::LLVMPointerType::get(rewriter.getContext(), 8);
  Value numRecordsByte = int_val(32, std::numeric_limits<int>::max() - 1);

  Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
      loc, rsrcType, basePtr, stride, numRecordsByte, flagsConst);
  return resource;
}

Value BufferEmitter::emitLoad(Type type, Value rsrcDesc, Value offset,
                              Value pred, Value falseVal) {
  SmallVector<Value, 6> args;
  fillCommonArgs(type, rsrcDesc, offset, pred, args);
  Type bufferType = getBufferOpType(type, false);
  Value data = rewriter.create<ROCDL::RawPtrBufferLoadOp>(
      loc, bufferType, args, ArrayRef<NamedAttribute>());
  data = bitcast(data, type);
  if (!isZero(falseVal))
    data = select(pred, data, falseVal);
  return data;
}

// Utility function to traverse a struct, get to the GEP contained
// in the struct at position `pos` and extract its base pointer and offset
mlir::FailureOr<std::pair<Value, Value>> getBaseAndOffset(Value ptr,
                                                          int64_t pos) {
  // Typedef the return type here, to have more coincise code
  using ReturnType = mlir::FailureOr<std::pair<Value, Value>>;

  Operation *currentOp = ptr.getDefiningOp();
  auto res =
      llvm::TypeSwitch<Operation *, ReturnType>(currentOp)
          .Case<LLVM::GEPOp>([&](auto gepOp) -> ReturnType {
            SmallVector<Value> indices =
                llvm::to_vector(gepOp.getDynamicIndices());
            if (indices.size() == 1)
              return std::make_pair(gepOp.getBase(), indices[0]);
            return failure();
          })
          .Case<LLVM::AddrSpaceCastOp>([&](auto addrspaceCastOp) -> ReturnType {
            return getBaseAndOffset(addrspaceCastOp.getArg(), pos);
          })
          .Case<LLVM::ExtractValueOp>([&](auto extractValOp) -> ReturnType {
            ArrayRef<int64_t> position = extractValOp.getPosition();
            if (position.size() > 1)
              return failure();
            return getBaseAndOffset(extractValOp.getContainer(), position[0]);
          })
          .Case<LLVM::InsertValueOp>([&](auto insertValOp) -> ReturnType {
            ArrayRef<int64_t> position = insertValOp.getPosition();
            if (position.size() > 1)
              return failure();
            if (position[0] == pos)
              return getBaseAndOffset(insertValOp.getValue(), 0);
            return getBaseAndOffset(insertValOp.getContainer(), pos);
          })
          .Default([&](Operation *op) -> ReturnType { return failure(); });
  return res;
}

void BufferEmitter::emitAtomic(RewriterBase &rewriter, Location loc,
                               triton::HCU::TargetInfo targetInfo, Value ptr,
                               Value val, Value pred, bool useBufferOps) {
  // Use a predicated buffer store intrinsic if we can. This should be optimal,
  // since we don't have to emit any branch, ever.
  auto maybeBaseAndOffset = getBaseAndOffset(ptr);
  if (!failed(maybeBaseAndOffset)) {
    Type elemTy = val.getType();
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    Value basePtr = maybeBaseAndOffset->first;
    Value offset = maybeBaseAndOffset->second;
    val = bitcast(val, vecType);
    Type bufferType = getBufferOpType(vecType, true);
    SmallVector<Value, 6> args{val};
    Value rsrcDesc = createResourceDescriptor(basePtr);
    fillCommonArgs(vecType, rsrcDesc, offset, pred, args);
    rewriter.create<ROCDL::RawPtrBufferAtomicFaddOp>(
        loc, TypeRange{}, args, ArrayRef<NamedAttribute>());
    return;
  }
}

void BufferEmitter::emitStore(Value rsrcDesc, Value offset, Value data,
                              Value pred) {
  VectorType vecTy = cast<VectorType>(data.getType());
  Type bufferType = getBufferOpType(vecTy, false);
  if (vecTy != bufferType)
    data = bitcast(data, bufferType);
  SmallVector<Value, 6> args{data};
  fillCommonArgs(vecTy, rsrcDesc, offset, pred, args);
  rewriter.create<ROCDL::RawPtrBufferStoreOp>(loc, TypeRange{}, args,
                                              ArrayRef<NamedAttribute>());
}

Type BufferEmitter::getBufferOpType(Type type, bool atomicsOp) {
  int64_t vecSize = 1;
  Type elementType = type;
  if (auto vecType = dyn_cast<VectorType>(type)) {
    vecSize = vecType.getNumElements();
    elementType = vecType.getElementType();
  }

  const int valueElemNBits = std::max(8u, elementType.getIntOrFloatBitWidth());
  const size_t totalWidthBits = valueElemNBits * vecSize;

  // For bf16, always convert to i16
  Type bufferElementType = elementType;
  if (elementType.isBF16() && !atomicsOp)
    bufferElementType = rewriter.getI16Type();

  // If we are dealing with a subword type (e.g., i8 or f16) but we
  // still need multiple words, then pack the subwords into 32bit integers
  // and update the vector length and the type
  int64_t bufferVecSize = vecSize;
  if (valueElemNBits < 32 && !atomicsOp) {
    if (totalWidthBits > 32) {
      bufferElementType = rewriter.getI32Type();
      bufferVecSize = totalWidthBits / 32;
    } else {
      bufferElementType = rewriter.getIntegerType(totalWidthBits);
      bufferVecSize = 1;
    }
  }

  // This is the buffer type that the buffer operation will use. It
  // will be bitcast-able to the original type. So if the types
  // ended up different, we simply have to emit a `bitcastOp` to convert
  Type bufferType = type;
  if (bufferVecSize != vecSize || bufferElementType != elementType)
    bufferType = VectorType::get(bufferVecSize, bufferElementType);
  if (bufferVecSize == 1)
    bufferType = getElementTypeOrSelf(bufferType);

  return bufferType;
}

void BufferEmitter::fillCommonArgs(Type type, Value rsrcDesc,
                                   Value vOffsetElems, Value pred,
                                   SmallVector<Value> &args) {

  // 1. Create the (masked) offset
  Type elementType = getElementTypeOrSelf(type);
  const int valueElemNBits = std::max(8u, elementType.getIntOrFloatBitWidth());
  const int elementByteWidth = valueElemNBits / 8;
  // Please note: the index passed is not in bytes, but in number of elements
  // In order to pass the index to the buffer operation, we need to convert in
  // bytes (i.e., we need to multiply by `elementByteWidth`)
  Value vOffsetOutOfBunds = int_val(
      32, static_cast<int>(std::numeric_limits<int>::max() + int64_t(1)));
  Value vOffsetBytes = mul(int_val(32, elementByteWidth), vOffsetElems);
  Value maskedOffsetBytes = select(pred, vOffsetBytes, vOffsetOutOfBunds);

  // 2. Set the sgprOffset to 0
  Value sgprOffset = int_val(32, 0);

  // 3. Create the cache modifiers word
  // bit 0: GLC = 0 (atomics drop value, less coherency)
  // bits 1-2: SLC, DLC = 0 (similarly)
  // bit 3: swizzled (0 for raw)
  Value cacheModifiers = int_val(32, 0);

  // 5. Add the arguments
  args.push_back(rsrcDesc);
  args.push_back(maskedOffsetBytes);
  args.push_back(sgprOffset);
  args.push_back(cacheModifiers);
}

} // namespace mlir::LLVM::HCU
