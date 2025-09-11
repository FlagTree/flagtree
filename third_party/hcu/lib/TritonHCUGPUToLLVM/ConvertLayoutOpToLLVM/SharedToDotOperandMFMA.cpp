/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "../PatternTritonGPUOpToLLVM.h"
#include "SharedToDotOperandHelper.h"
#include "Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::HCUMfmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandMFMA {

/**
 * @brief This function maps particular load of mfma dot operand to element
 * indexes(row, col)
 *
 * Whole tensor is broken into "blocks" of warps along "non-K" axis.
 * One block could be processed by multiple warps.
 * One warp works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of warp-block in tensor
 * 2. Offset of warp inside one warp-block
 * 3. Offset of tile in one warp
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axies independently
 * Note that this function returns the offsets of elements in the first
 * warp-block. The offsets of elements in later warp-blocks can be computed
 * by adding a constant stride to the xor-ed offsets of elements in the
 * first warp-block.
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one MFMA instruction
 * @param warpId id component of 2d warp grid along non-K axis
 * @param laneId lane id in warp [0..63]
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repetition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension size of one MFMA instruction
 * @param iKDim K dimension size of one MFMA instruction
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>> computeTensorElemMappingInBlock(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value warpId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
    int loadVecSize, unsigned iNonKDim, unsigned iKDim, bool interleave) {
  auto numM = reps[1];
  auto numK = reps[2];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value nonKDim = i32_val(iNonKDim);
  Value warpVOffset = mul(warpId, i32_val(elemsPerInstr[0]));

  auto rank = smemOffsets.size();

  for (int tile = 0; tile < numK; ++tile) {
    Value tileVOffset = _0;
    Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

    Value laneVOffset = urem(laneId, nonKDim);
    Value laneHOffset;
    if (!interleave) {
      if (iNonKDim == 32)
        laneHOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);
      else {
        // In this configuration warp contains 16 copies of same data
        if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4) {
          laneHOffset = i32_val(0);
        } else {
          assert(iKDim * iNonKDim / numOfElems == 64 &&
                 "seems no all threads in warp contain unique elements");
          laneHOffset = mul(udiv(laneId, nonKDim), i32_val(numOfElems));
        }
      }
    } else
      laneHOffset = udiv(laneId, nonKDim);

    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemVOffset = _0;
      Value elemHOffset =
          interleave ? i32_val(loadId * loadVecSize * iNonKDim / numOfElems)
                     : i32_val(loadId * loadVecSize);

      Value sliceVOffset =
          add(add(add(tileVOffset, laneVOffset), elemVOffset), warpVOffset);
      Value sliceHOffset = add(add(tileHOffset, laneHOffset), elemHOffset);

      Value row = add(sliceVOffset, smemOffsets[rank - 2]);
      Value col = add(sliceHOffset, smemOffsets[rank - 1]);

      mapping[loadsPerThread * tile + loadId] = {row, col};
    }
  }
  return mapping;
}

bool hasSwizzleEnabled(const SharedEncodingAttr &srcEncoding) {
  return srcEncoding.getMaxPhase() > 1;
}

bool useDsReadMat(const HCUMfmaEncodingAttr &mfmaLayout, int bitWidth,
                  int opIdx) {
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  if ((mDim == 32 && nDim == 32) && (bitWidth == 16 || bitWidth == 8))
    return true;
  if ((mDim == 16 && nDim == 32) && (bitWidth == 16 || bitWidth == 8) &&
      (opIdx == 1))
    return true;
  return false;
}

bool useMmacFuse(const HCUMfmaEncodingAttr &mfmaLayout, int bitWidth,
                 int opIdx) {
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  auto kDim = mfmaLayout.getKDim();
  if ((((mDim == 16 && nDim == 64) ||
        (mDim == 16 && nDim == 16 && kDim == 32)) &&
       (bitWidth == 16)) ||
      (((mDim == 16 && nDim == 16 && kDim == 64) ||
        (mDim == 16 && nDim == 64 && kDim == 64)) &&
       bitWidth == 8))
    return true;
  return false;
}

// Computes offsets for operand B or transposed operand A
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape [K, nonK] consumed by one MFMA
// instruction
// @param warpId warp id for the "non K" axis
// @param laneId lane id in warp [0..63]
// @param warpsPerBlock number of warps per horizontal axis
// @param numOfElems number of elements accessed by threads per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsets(ConversionPatternRewriter &rewriter, Location loc,
                       const ArrayRef<int64_t> &elemsPerInstr, Value warpId,
                       Value laneId, int warpsPerBlock, int numOfElems,
                       ArrayRef<int64_t> reps, Value cSwizzleOffset,
                       bool interleave) {
  auto numK = reps[1];
  auto numN = reps[2];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  auto iKDim = elemsPerInstr[0];
  auto iNonKDim = elemsPerInstr[1];
  int lineSize = warpsPerBlock * iNonKDim * numN;
  Value _nonKDim = i32_val(iNonKDim);
  Value warpOffset = mul(warpId, i32_val(iNonKDim));
  Value colOffset = urem(laneId, _nonKDim);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * iNonKDim * warpsPerBlock);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * iKDim * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        // halfOffset is an offset related to wrapping of warp in the tile.
        // for example, mfma 32 case (mapping of tensor elements to lane ids in
        // warp):
        //
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        // 32 33 34 35 ... 63  <- at this point warp is wrapping
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        Value rowOffset;
        if (!interleave) {
          Value halfOffset;
          if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4)
            halfOffset = i32_val(0);
          else
            halfOffset =
                mul(udiv(laneId, _nonKDim), i32_val(numOfElems * lineSize));
          rowOffset = add(i32_val(elem * lineSize), halfOffset);
        } else {
          Value halfOffset = mul(udiv(laneId, _nonKDim), i32_val(lineSize));
          rowOffset =
              add(i32_val(elem * lineSize * iNonKDim / numOfElems), halfOffset);
        }
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(warpOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

llvm::SmallVector<Value> dsReadMatComputeOffsets(
    ConversionPatternRewriter &rewriter, Location loc,
    SharedEncodingAttr sharedLayout, const ArrayRef<int64_t> &elemsPerInstr,
    Value waveId, Value laneId, int warpsPerBlock, int totolElemsPerThread,
    ArrayRef<int64_t> reps, Value nonKStrides, int interleave) {
  auto numK = reps[1];
  auto numN = reps[2];
  SmallVector<Value> offsets(numK * numN);
  auto iKDim = elemsPerInstr[0];
  auto iNonKDim = elemsPerInstr[1];
  Value lineSize = nonKStrides;
  Value _nonKDim = i32_val(iNonKDim);
  Value waveOffset = mul(waveId, i32_val(iNonKDim));
  int numOfThreadsNonK =
      32 /*magic number*/ /
      totolElemsPerThread; // threadLayout in warps: 32x2 for ds_read_m32x32b8;
                           // 16x4 for ds_read_m32x16b16
  int vecSize = sharedLayout.getVec();
  int swizzleWidth = numOfThreadsNonK * totolElemsPerThread;
  auto vec = i32_val(sharedLayout.getVec());
  auto perPhase = i32_val(sharedLayout.getPerPhase());
  auto maxPhase = i32_val(sharedLayout.getMaxPhase());

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * iNonKDim * warpsPerBlock);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = mul(i32_val(tile * iKDim), lineSize);
      {
        // ds_read_m32x32b8: (v0&2)*16+(v0/2)*MT or ds_read_m32x16b16:
        // (v0&3)*8+(v0/4)*MT
        Value colOrdered;
        Value row = udiv(
            laneId, i32_val(numOfThreadsNonK)); // (rowid / 4) + (rowid % 4) * 4
        if (interleave) {                       // FA secondDot OPT PASS
          row = add(udiv(row, i32_val(4)),
                    mul(urem(row, i32_val(4)), i32_val(4)));
        }
        Value colStart = add(waveOffset, blockOffset);
        Value col = add(colStart, mul(urem(laneId, i32_val(numOfThreadsNonK)),
                                      i32_val(totolElemsPerThread)));
        if (vecSize < swizzleWidth)
          colOrdered = urem(col, vec);
        else
          colOrdered = mul(urem(laneId, i32_val(numOfThreadsNonK)),
                           i32_val(totolElemsPerThread));
        auto phase = urem(udiv(row, perPhase), maxPhase);
        auto colOffSwizzled = mul(xor_(udiv(col, vec), phase), vec);
        auto colOffset = add(colOffSwizzled, colOrdered);
        Value rowOffset = mul(row, lineSize);
        Value elemOffset = add(rowOffset, colOffset);
        offsets[block * numK + tile] = elemOffset;
      }
    }
  }
  return offsets;
}

llvm::SmallVector<Value> mmacFuseComputeOffsets(
    ConversionPatternRewriter &rewriter, Location loc,
    SharedEncodingAttr sharedLayout, const ArrayRef<int64_t> &elemsPerInstr,
    Value waveId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, Value nonKStrides, int interleave, int kWidth) {

  auto iNonKDim = elemsPerInstr[0];
  auto iKDim = elemsPerInstr[1];
  auto iterNum = elemsPerInstr[0] / 16;
  auto numN = reps[1] / iterNum;
  auto numK = reps[2] / 2;
  SmallVector<Value> offsets(numK * numN * iterNum);
  Value lineSize = nonKStrides;
  Value _nonKDim = i32_val(iNonKDim);
  Value waveOffset = mul(waveId, i32_val(iNonKDim));
  int vecSize = sharedLayout.getVec();

  auto vec = i32_val(sharedLayout.getVec());
  auto perPhase = i32_val(sharedLayout.getPerPhase());
  auto maxPhase = i32_val(sharedLayout.getMaxPhase());

  for (int block = 0; block < numN; ++block) {
    Value blockOffset_row = i32_val(block * iNonKDim * warpsPerBlock);
    for (int n = 0; n < iterNum; n++) {
      for (int tile = 0; tile < numK; ++tile) {
        Value tileOffset_col = i32_val(tile * iKDim);
        Value row_1 = urem(laneId, i32_val(16));
        Value row =
            add(add(add(mul(row_1, i32_val(iterNum)), i32_val(n)), waveOffset),
                blockOffset_row);
        auto colOffset =
            add(mul(udiv(laneId, i32_val(16)), i32_val(kWidth * 2)),
                tileOffset_col);
        auto phase = urem(udiv(row, perPhase), maxPhase);
        auto colOffSwizzled =
            mul(xor_(udiv(colOffset, vec), phase), vec); // zhenggf TODO
        auto rowOffset = mul(row, lineSize);
        Value elemOffset = add(rowOffset, colOffSwizzled);
        offsets[block * iterNum * numK + n * numK + tile] = elemOffset;
      }
    }
  }
  return offsets;
}

bool isColMajor(::llvm::ArrayRef<unsigned> order) {
  auto rank = order.size();
  return order[0] == (rank - 2);
}

Value generateDsReadM32x16B16Op(ConversionPatternRewriter &rewriter,
                                Location loc, Value addr, Value offset,
                                Value valVec) {
  StringRef dsReadName = "rocdl.ds.read.m32x16b16";
  auto resType = valVec.getType();
  OperationState loweredOp(loc, dsReadName);
  loweredOp.addTypes(resType);
  loweredOp.addOperands({addr, offset});
  return rewriter.create(loweredOp)->getResult(0);
}

Value generateDsReadM32x32B8Op(ConversionPatternRewriter &rewriter,
                               Location loc, Value addr, Value offset,
                               Value valVec) {
  StringRef dsReadName = "rocdl.ds.read.m32x32b8";
  auto resType = valVec.getType();
  OperationState loweredOp(loc, dsReadName);
  loweredOp.addTypes(resType);
  loweredOp.addOperands({addr, offset});
  return rewriter.create(loweredOp)->getResult(0);
}

SmallVector<Value> loadX4(ConversionPatternRewriter &rewriter, Location loc,
                          Value ptrs, Value offset, Type elemType) {
  // The struct should have exactly the same element types.

  auto vecTy = vec_ty(i32_ty, 4);
  Value valVec = undef(vecTy);

  Value resV4i32;
  if (elemType.isBF16() || elemType.isF16()) {
    resV4i32 = generateDsReadM32x16B16Op(rewriter, loc, ptrs, offset, valVec);
  } else if (elemType.isInteger(8)) {
    resV4i32 = generateDsReadM32x32B8Op(rewriter, loc, ptrs, offset, valVec);
  } else
    llvm::report_fatal_error("ds_read_m* data type not supported");

  return {extract_element(i32_ty, resV4i32, i32_val(0)),
          extract_element(i32_ty, resV4i32, i32_val(1)),
          extract_element(i32_ty, resV4i32, i32_val(2)),
          extract_element(i32_ty, resV4i32, i32_val(3))};
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  auto aTensorTy = cast<MemDescType>(tensor.getType());
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto rank = shape.size();
  int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto mfmaLayout = cast<HCUMfmaEncodingAttr>(encoding.getParent());
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64) ||
         (mDim == 16 && nDim == 32) || (mDim == 16 && nDim == 64));
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto sharedLayout = cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto order = sharedLayout.getOrder();
  assert((rank == 2 || order[2] == 0) &&
         "expect batch to be the slowest dimension");

  auto elemTy = aTensorTy.getElementType();
  int bitWidth = elemTy.getIntOrFloatBitWidth();
  assert(bitWidth >= 1);
  int totolElemsPerThread = 128 / bitWidth;
  auto kWidth = encoding.getKWidth();
  auto elemsPerInstr = mfmaLayout.getMFMAInstrShapeForOperands(kWidth, opIdx);
  bool isMmacFuse = useMmacFuse(mfmaLayout, bitWidth, opIdx);
  if (isMmacFuse)
    elemsPerInstr = mfmaLayout.getMFMAInstrShapeForOperands(kWidth * 2, opIdx);
  bool isDsReadMat = useDsReadMat(mfmaLayout, bitWidth, opIdx);

  int64_t mfmaInstrNonK;
  int64_t mfmaInstrK;
  // TODO(Lixun): make it simpler
  // getMFMAInstrShapeForOperands always returns a 2D vector
  if (rank == 3) {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
    mfmaInstrK = elemsPerInstr[kDimIdx - 1];
  } else {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
    mfmaInstrK = elemsPerInstr[kDimIdx];
  }

  auto numReps = mfmaLayout.getMFMARepForOperands(shape, kWidth, opIdx);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];
  auto repB = numReps[0];
  // TODO(Lixun): make it simpler
  // getMFMARepForOperands always returns a 3D vector
  if (rank == 2) {
    if (isDsReadMat) {
      numReps[nonKDimIdx + 1] =
          std::max(numReps[nonKDimIdx + 1] / 2, (int64_t)1);
    }
    numRepNonK = numReps[nonKDimIdx + 1];
    numRepK = numReps[kDimIdx + 1];
  }

  unsigned iWarpSize = triton::gpu::getWarpSize(mfmaLayout);
  assert(iWarpSize == 64);
  Value warpSize = i32_val(iWarpSize);
  Value linearWarpId = udiv(thread, warpSize);
  Value lane = urem(thread, warpSize);

  Value spatialWarpId = HCU::getWarpIdInBlock(
      rewriter, loc, linearWarpId, warpsPerCTA, mfmaInstrNonK,
      shape[nonKDimIdx], nonKDimIdx, triton::gpu::getOrder(mfmaLayout));

  // number of duplicates of elements in warp
  // In case of 64x4 x 4x4 multiplication, 4x4 B operand is duplicated 16 times
  int numSubBlocks = 1;
  if ((mfmaInstrK == 4 || mfmaInstrK == 1) && mfmaInstrNonK == 4)
    numSubBlocks = 16;
  // numOfElemsPerThreadPerMfmaInstr
  int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
  int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  int warpsPerBatch =
      rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
  Value warpIdInBatch = urem(linearWarpId, i32_val(warpsPerBatch));
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  bool isFastPath =
      !HCU::isKMajor(order, opIdx) && !hasSwizzleEnabled(sharedLayout);
  bool interleave = mfmaLayout.getInterleave();
  Value nonKStrides = smemObj.strides[order[1]];
  if (isFastPath || isDsReadMat || isMmacFuse) {
    // fast path handles tensors that are not k-major and have swizzling
    // disabled, in which case offsets computation can be simplified
    // TODO (zhanglx): later when we enable vector access to LDS for non k-major
    // tensors, we'll refactor the scope of fast and normal path
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    if (opIdx == 0) {          // 矩阵A
      if (isColMajor(order)) { // 转置
        SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> reps{numReps[0], numReps[2], numReps[1]};

        if (isDsReadMat) {
          offsets = dsReadMatComputeOffsets(
              rewriter, loc, sharedLayout, elemsPerInstr, spatialWarpId, lane,
              warpsPerBlockNonK, totolElemsPerThread, reps, nonKStrides,
              interleave);
        } else if (isMmacFuse) {
          offsets = HCU::computeOffsetsAType(
              rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
              spatialWarpId, lane, warpsPerBlockNonK, numOfElems, numReps,
              smemObj, sharedLayout, mDim, mfmaInstrK, interleave);
        } else {
          offsets = fastPathComputeOffsets(
              rewriter, loc, elemsPerInstr, spatialWarpId, lane,
              warpsPerBlockNonK, numOfElems, reps, cSwizzleOffset, interleave);
        }
      } else { // 不转置
        if (isMmacFuse) {
          offsets = mmacFuseComputeOffsets(
              rewriter, loc, sharedLayout, elemsPerInstr, spatialWarpId, lane,
              warpsPerBlockNonK, numOfElems, numReps, nonKStrides, interleave,
              kWidth);
        } else
          llvm_unreachable(
              "row major operand A should be handled in the normal path");
      }
    } else { // 矩阵B
      if (isColMajor(order)) {
        // SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> elemsPerInstr{mfmaInstrNonK, mfmaInstrK};
        SmallVector<int64_t> reps{numReps[0], numReps[2], numReps[1]};
        if (isMmacFuse) {
          offsets = mmacFuseComputeOffsets(rewriter, loc, sharedLayout,
                                           elemsPerInstr, spatialWarpId, lane,
                                           warpsPerBlockNonK, numOfElems, reps,
                                           nonKStrides, interleave, kWidth);
        } else
          llvm_unreachable(
              "col major operand B should be handled in the normal path");
      } else {
        if (isMmacFuse) {
          llvm_unreachable(
              "row major operand B should be handled in the normal path");
        } else if (isDsReadMat) {
          offsets = dsReadMatComputeOffsets(
              rewriter, loc, sharedLayout, elemsPerInstr, spatialWarpId, lane,
              warpsPerBlockNonK, totolElemsPerThread, numReps, nonKStrides,
              interleave);
        } else {
          offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                           spatialWarpId, lane,
                                           warpsPerBlockNonK, numOfElems,
                                           numReps, cSwizzleOffset, interleave);
        }
      }
    }
    smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  } else { // normal path
    // Normal path handles tensors that fall into either of the following three
    // cases:
    //   1. k-major + swizzling is enabled <-- this should be the most
    //   performant case
    //   2. k-major + swizzling is disabled <-- for testing purpose only
    //   3. non k-major + swizzling is enabled <-- for testing purpose only
    if (opIdx == 0) {
      offsets = HCU::computeOffsetsAType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWarpId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          sharedLayout, mDim, mfmaInstrK, interleave);
    } else {
      assert(opIdx == 1);
      offsets = HCU::computeOffsetsBType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWarpId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          sharedLayout, nDim, mfmaInstrK, interleave);
    }
    smemBase = HCU::computeBasePtr(rewriter, loc, smemObj);
  }

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  if (isDsReadMat) {

    int mmacNumElemsPerThread = numOfElems / 2;
    int numUnPackRegToElems = 32 /*register bit width*/ / bitWidth;
    loadedValues.resize(numRepNonK * numRepK * totolElemsPerThread);

    for (int nonK = 0; nonK < numRepNonK; ++nonK) {
      for (int k = 0; k < numRepK; ++k) {

        auto loadVecTy = vec_ty(elemTy, 1);
        Value loadOffset;
        loadOffset = offsets[nonK * numRepK + k];
        Value nonKStridesBytes =
            mul(i32_val(bitWidth / 8), i32_val(shape[nonKDimIdx]));
        Value tileOffset = mul(i32_val(k * mfmaInstrK), nonKStridesBytes);
        Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
        SmallVector<Value> result;
        result = loadX4(rewriter, loc, loadAddress, tileOffset, resElemTy);
        {
          auto vecTy0 = vec_ty(resElemTy, mmacNumElemsPerThread);
          Value valVec0 = undef(vecTy0);
          auto vecTy1 = vec_ty(resElemTy, mmacNumElemsPerThread);
          Value valVec1 = undef(vecTy1);
          for (int n = 0; n < 2; ++n) {
            Value regValue = result[n];
            auto vecTy = vec_ty(resElemTy, numUnPackRegToElems);
            auto valTy = bitcast(regValue, vecTy);
            for (int i = 0; i < numUnPackRegToElems; ++i) {
              auto val0 = extract_element(resElemTy, valTy, i32_val(i));
              valVec0 = insert_element(vecTy0, valVec0, val0,
                                       i32_val(n * numUnPackRegToElems + i));
            }
          }
          for (int n = 0; n < 2; ++n) {
            Value regValue = result[n + 2];
            auto vecTy = vec_ty(resElemTy, numUnPackRegToElems);
            auto valTy = bitcast(regValue, vecTy);
            for (int i = 0; i < numUnPackRegToElems; ++i) {
              auto val1 = extract_element(resElemTy, valTy, i32_val(i));
              valVec1 = insert_element(vecTy1, valVec1, val1,
                                       i32_val(n * numUnPackRegToElems + i));
            }
          }

          for (int i = 0; i < mmacNumElemsPerThread; i++) {
            loadedValues[(2 * nonK) * numRepK * mmacNumElemsPerThread +
                         k * mmacNumElemsPerThread + i] =
                extract_element(resElemTy, valVec0, i32_val(i));
            loadedValues[(2 * nonK + 1) * numRepK * mmacNumElemsPerThread +
                         k * mmacNumElemsPerThread + i] =
                extract_element(resElemTy, valVec1, i32_val(i));
          }
        }
      }
    }
  } else if (isMmacFuse) {
    SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
    if (opIdx == 1)
      numRepNonK = numRepNonK / 4 > 0 ? numRepNonK / 4 : 1;
    numRepK = numRepK / 2;
    int loadsPerThread = offsets.size() / numRepK / numRepNonK; // 4
    int elemsPerLoad = kWidth * 2;                              // 4
    numOfElems = loadsPerThread * elemsPerLoad;                 // 16
    assert(numOfElems % loadsPerThread == 0);

    for (int b = 0; b < repB; ++b) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset = mul(i32_val(operandSize),
                              add(warpIdInBatch, i32_val(b * warpsPerBatch)));
      for (int nonK = 0; nonK < numRepNonK; ++nonK) {
        int blockNonKOffset = nonK * mfmaInstrNonK * warpsPerBlockNonK;
        Value warpBlockOffAdjust = i32_val(blockNonKOffset * shape[order[0]]);
        for (int k = 0; k < numRepK; ++k) {
          auto vecTy = vec_ty(resElemTy, numOfElems);
          for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
            auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
            Value loadOffset;
            loadOffset = offsets[nonK * loadsPerThread * numRepK +
                                 k * loadsPerThread + loadId];
            loadOffset = add(loadOffset, batchOffset);
            Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
            Value loadedValue = load(loadVecTy, loadAddress);
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(elemTy, loadedValue, i32_val(elemId));
              loadedValues.push_back(elemVal);
            }
          }
        }
      }
    }
  } else {

    int loadsPerThread = offsets.size() / numRepK / numRepNonK;
    int elemsPerLoad = numOfElems / loadsPerThread;
    assert(numOfElems % loadsPerThread == 0);

    for (int b = 0; b < repB; ++b) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset = mul(i32_val(operandSize),
                              add(warpIdInBatch, i32_val(b * warpsPerBatch)));
      for (int nonK = 0; nonK < numRepNonK; ++nonK) {
        int blockNonKOffset = nonK * mfmaInstrNonK * warpsPerBlockNonK;
        Value warpBlockOffAdjust = i32_val(blockNonKOffset * shape[order[0]]);
        for (int k = 0; k < numRepK; ++k) {
          auto vecTy = vec_ty(resElemTy, numOfElems);
          for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
            auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
            Value loadOffset;
            loadOffset = offsets[nonK * loadsPerThread * numRepK +
                                 k * loadsPerThread + loadId];
            loadOffset = add(loadOffset, batchOffset);
            Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
            Value loadedValue = load(loadVecTy, loadAddress);
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(elemTy, loadedValue, i32_val(elemId));
              loadedValues.push_back(elemVal);
            }
          }
        }
      }
    }
  }
  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandMFMA
