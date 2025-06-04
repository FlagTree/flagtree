//===------------------- Tx81MemrefToLLVM.cpp------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Tx81MemrefToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include <cstdint>
#include <vector>

#define DEBUG_TYPE "tx81-memref-to-llvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"

// Used for allocate spm memory
uint64_t spmPointer = 0x10000;

namespace {
// Used for kcore load/store data from/to spm
const int64_t spmMappingOffset = 0x30400000;

//===----------------------------------------------------------------------===//
// Tx81 Custom MemRef Op Conversion Patterns
//===----------------------------------------------------------------------===//

struct TsmMemRefAllocOpLowering : public AllocLikeOpLLVMLowering {
  TsmMemRefAllocOpLowering(const LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<Value, Value>
  allocateBufferFromSPM(ConversionPatternRewriter &rewriter, Location loc,
                        Operation *op) const {
    // create GEPOp for spm address.
    MemRefType memRefType = getMemRefResultType(op);
    Value spmOffsetOp = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(), rewriter.getI32IntegerAttr(spmPointer));
    Type elementType = typeConverter->convertType(memRefType.getElementType());
    auto elementPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value spmAddr = rewriter.create<LLVM::ZeroOp>(loc, elementPtrType);

    spmAddr = rewriter.create<LLVM::PtrToIntOp>(op->getLoc(),
                                                rewriter.getI64Type(), spmAddr);
    spmAddr = rewriter.create<LLVM::AddOp>(op->getLoc(), rewriter.getI64Type(),
                                           spmAddr, spmOffsetOp);

    spmAddr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), elementPtrType,
                                                spmAddr);
    Value allocatedPtr = spmAddr;
    if (!allocatedPtr)
      return std::make_tuple(Value(), Value());
    Value alignedPtr = allocatedPtr;

    // update spm pointer
    auto elemCount = memRefType.getNumElements();
    auto bitWidth = memRefType.getElementTypeBitWidth();
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (allocOp.getAlignment().has_value())
      bitWidth = allocOp.getAlignment().value();
    uint64_t totalByte = (elemCount * bitWidth + 7) / 8;
    spmPointer += totalByte;

    return std::make_tuple(allocatedPtr, alignedPtr);
  }

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    return allocateBufferFromSPM(rewriter, loc, op);
  }
};

template <typename MemrefOp>
struct MemrefLoadOrStoreOpLowering : public ConvertOpToLLVMPattern<MemrefOp> {

  using ConvertOpToLLVMPattern<MemrefOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename MemrefOp::Adaptor;

  LogicalResult
  matchAndRewrite(MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();

    Value dataPtr = ConvertToLLVMPattern::getStridedElementPtr(
        op.getLoc(), type, adaptor.getMemref(), adaptor.getIndices(), rewriter);

    // TODO: Add spm offset according the memory space
    MemRefDescriptor memRefDescriptor(adaptor.getMemref());
    auto intPtrType = ConvertToLLVMPattern::getIntPtrType(
        memRefDescriptor.getElementPtrType().getAddressSpace());
    Value ptrValue =
        rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), intPtrType, dataPtr);

    // Workaround: Should add memory space analysis pass.
    Operation *opBase = op;
    if (!opBase->hasAttr("isSpm")) {
      return rewriter.notifyMatchFailure(
          op, "Load/Store should have isSpm attribute.");
    }
    int isSpm =
        cast<IntegerAttr>(opBase->getAttr("isSpm")).getValue().getSExtValue();

    Value adjustedPtr = dataPtr;
    if (isSpm) {
      auto spmMemoryOffset = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(),
          rewriter.getI64IntegerAttr(spmMappingOffset));
      auto spmMemoryAddr = rewriter.create<LLVM::AddOp>(
          op.getLoc(), rewriter.getI64Type(),
          SmallVector<Value>({ptrValue, spmMemoryOffset}));

      auto ptrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(),
          *ConvertToLLVMPattern::getTypeConverter()->getMemRefAddressSpace(
              type));
      auto spmMemoryAddrPtr =
          rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, spmMemoryAddr);

      adjustedPtr = spmMemoryAddrPtr;
    }

    // Wether need memoryspace cast
    if constexpr (std::is_same<MemrefOp, memref::LoadOp>()) {

      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), adjustedPtr,
                                                0, false, op.getNontemporal());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
          op, adaptor.getValue(), adjustedPtr, 0, false, op.getNontemporal());
    }

    return success();
  }
};

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = castOp.getSource().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  /// Extracts allocated, aligned pointers and offset from a ranked or unranked
  /// memref type. In unranked case, the fields are extracted from the
  /// underlying ranked descriptor.
  void extractPointersAndOffset(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &typeConverter,
                                Value originalOperand, Value convertedOperand,
                                Value *allocatedPtr, Value *alignedPtr,
                                Value *offset = nullptr) const {
    Type operandType = originalOperand.getType();
    if (isa<MemRefType>(operandType)) {
      MemRefDescriptor desc(convertedOperand);
      *allocatedPtr = desc.allocatedPtr(rewriter, loc);
      *alignedPtr = desc.alignedPtr(rewriter, loc);
      if (offset != nullptr)
        *offset = desc.offset(rewriter, loc);
      return;
    }

    // These will all cause assert()s on unconvertible types.
    unsigned memorySpace = *typeConverter.getMemRefAddressSpace(
        cast<UnrankedMemRefType>(operandType));
    auto elementPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), memorySpace);

    // Extract pointer to the underlying ranked memref descriptor and cast it to
    // ElemType**.
    UnrankedMemRefDescriptor unrankedDesc(convertedOperand);

    // FIXME: workaround, take memRefDescPtr as naked ptr.
    Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);
    *allocatedPtr = underlyingDescPtr;
    *alignedPtr = underlyingDescPtr;

    if (offset != nullptr) {
      *offset = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getI32IntegerAttr(0));
    }
  }

  LogicalResult convertSourceMemRefToDescriptor(
      ConversionPatternRewriter &rewriter, Type srcType,
      memref::ReinterpretCastOp castOp,
      memref::ReinterpretCastOp::Adaptor adaptor, Value *descriptor) const {
    MemRefType targetMemRefType =
        cast<MemRefType>(castOp.getResult().getType());
    auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
        typeConverter->convertType(targetMemRefType));
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);

    // Set allocated and aligned pointers.
    Value allocatedPtr, alignedPtr;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             castOp.getSource(), adaptor.getSource(),
                             &allocatedPtr, &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.getOffsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.getSizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.getStrides()[dynStrideId++]);
      else
        desc.setConstantStride(rewriter, loc, i, castOp.getStaticStride(i));
    }
    *descriptor = desc;
    return success();
  }
};

/// Materialize the MemRef descriptor represented by the results of
/// ExtractStridedMetadataOp.
class ExtractStridedMetadataOpLowering
    : public ConvertOpToLLVMPattern<memref::ExtractStridedMetadataOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractStridedMetadataOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!LLVM::isCompatibleType(adaptor.getOperands().front().getType()))
      return failure();

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.getSource());
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    auto sourceMemRefType = cast<MemRefType>(source.getType());
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    // Base buffer.
    Value baseBuffer = sourceMemRef.allocatedPtr(rewriter, loc);
    Value alignedBuffer = sourceMemRef.alignedPtr(rewriter, loc);
    MemRefDescriptor dstMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(),
        cast<MemRefType>(extractStridedMetadataOp.getBaseBuffer().getType()),
        baseBuffer, alignedBuffer);
    results.push_back((Value)dstMemRef);

    // Offset.
    results.push_back(sourceMemRef.offset(rewriter, loc));

    // Sizes.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.size(rewriter, loc, i));
    // Strides.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.stride(rewriter, loc, i));

    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};

/// Unpack the pointer returned by a memref.extract_aligned_pointer_as_index.
class ConvertExtractAlignedPointerAsIndex
    : public ConvertOpToLLVMPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractAlignedPointerAsIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp extractOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BaseMemRefType sourceTy = extractOp.getSource().getType();

    // FIXME: We want allocated ptr instead of aligned ptr.
    Value alignedPtr;
    if (sourceTy.hasRank()) {
      MemRefDescriptor desc(adaptor.getSource());
      alignedPtr = desc.allocatedPtr(rewriter, extractOp->getLoc());
    } else {
      auto elementPtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(), sourceTy.getMemorySpaceAsInt());

      UnrankedMemRefDescriptor desc(adaptor.getSource());
      Value descPtr = desc.memRefDescPtr(rewriter, extractOp->getLoc());

      alignedPtr = UnrankedMemRefDescriptor::allocatedPtr(
          rewriter, extractOp->getLoc(), descPtr, elementPtrTy);
    }

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        extractOp, getTypeConverter()->getIndexType(), alignedPtr);
    return success();
  }
};

} // namespace

void mlir::triton::populateTx81MemrefToLLVMConversionPatterns(
    RewritePatternSet &patterns, LLVMTypeConverter &converter) {
  // clang-format off
  patterns.add<TsmMemRefAllocOpLowering,
                MemRefReinterpretCastOpLowering,
                ExtractStridedMetadataOpLowering,
                ConvertExtractAlignedPointerAsIndex,
                MemrefLoadOrStoreOpLowering<memref::LoadOp>,
                MemrefLoadOrStoreOpLowering<memref::StoreOp>>(
                  converter);
  // clang-format on
}
