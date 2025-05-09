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

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __Wdma(uint64_t *src, uint64_t *dst, int shape_n, int shape_h, int shape_w,
            int shape_c, int stride_n, int stride_h, int stride_w,
            uint32_t fmt) {
  // Create gemm command buffer.
  TsmWdma *wdma = TsmNewWdma();
  TsmWdmaInstr inst = {I_WDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  wdma->AddSrcDst(&inst, (uint64_t)src, (uint64_t)dst, (Data_Format)fmt);

  wdma->ConfigStrideIteration(&inst, shape_c, stride_w, shape_w, stride_h,
                              shape_h, stride_n, shape_n);

  //   wdma->Wdma1d(&inst, (uint64_t)src, (uint64_t)dst, shape_c,
  //   (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteWdma(wdma);
}
