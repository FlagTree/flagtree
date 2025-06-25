//===------------------------ wdma.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Wdma, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"
#include <stdio.h>

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __Wdma(uint64_t *src, uint64_t *dst, int *src_shape, int *src_stride,
            int *dst_shape, int *dst_stride, uint32_t elem_bytes,
            uint32_t fmt) {
  // Dynamic shape, kernel implementation will cause shape equal to 0
  if (src_shape[0] == 0 || src_shape[1] == 0 || src_shape[2] == 0 ||
      src_shape[3] == 0)
    return;

  // Inner dim must be contiguous,last stride is always 1.
  assert(src_stride[3] == 1);
  assert(dst_stride[3] == 1);

  // Create gemm command buffer.
  TsmWdma *wdma = TsmNewWdma();
  TsmWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  if (is_contiguous(src_shape, src_stride, elem_bytes)) {
    wdma->AddSrcDst(&inst, (uint64_t)src, (uint64_t)dst, (Data_Format)fmt);

    wdma->ConfigStrideIteration(&inst, dst_shape[3], dst_stride[2],
                                dst_shape[2], dst_stride[1], dst_shape[1],
                                dst_stride[0], dst_shape[0]);
    TsmExecute(&inst);
    TsmDeleteWdma(wdma);
    return;
  }

  for (int64_t i = 0; i < src_shape[0]; ++i) {
    uint64_t src_ptr0 = (uint64_t)src + i * src_stride[0] * elem_bytes;
    uint64_t dst_ptr0 = (uint64_t)dst + i * dst_stride[0] * elem_bytes;

    for (int64_t j = 0; j < src_shape[1]; ++j) {
      uint64_t src_ptr1 = src_ptr0 + j * src_stride[1] * elem_bytes;
      uint64_t dst_ptr1 = dst_ptr0 + j * dst_stride[1] * elem_bytes;

      for (int64_t k = 0; k < src_shape[2]; ++k) {
        uint64_t src_ptr2 = src_ptr1 + k * src_stride[2] * elem_bytes;
        uint64_t dst_ptr2 = dst_ptr1 + k * dst_stride[2] * elem_bytes;
        wdma->Wdma1d(&inst, (uint64_t)src_ptr2, (uint64_t)dst_ptr2,
                     src_shape[3], (Data_Format)fmt);
        TsmExecute(&inst);
        TsmWaitfinish();
      }
    }
  }

  // Destroy the command buffer.
  TsmDeleteWdma(wdma);
}
