//===------------------------ rdma.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Rdma, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __Rdma(uint64_t *src, uint64_t *dst, int shape_n, int shape_h, int shape_w,
            int shape_c, int stride_n, int stride_h, int stride_w,
            uint32_t fmt) {
  // Create gemm command buffer.
  TsmRdma *rdma = TsmNewRdma();
  TsmRdmaInstr inst = {I_RDMA,
                       {
                           0,
                       },
                       {
                           0,
                       }};

  rdma->AddSrcDst(&inst, (uint64_t)src, (uint64_t)dst, (Data_Format)fmt);

  rdma->ConfigStrideIteration(&inst, shape_c, stride_w, shape_w, stride_h,
                              shape_h, stride_n, shape_n);

  //   rdma->Rdma1d(&inst, (uint64_t)src, (uint64_t)dst, shape_c,
  //   (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRdma(rdma);
}
