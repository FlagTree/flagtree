//===------------------------ gatherscatter.c -----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::GatherScatter see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t size, uint32_t src_s0,
                   uint32_t src_i0, uint32_t src_s1, uint32_t src_i1,
                   uint32_t src_s2, uint32_t src_i2, uint32_t dst_s0,
                   uint32_t dst_i0, uint32_t dst_s1, uint32_t dst_i1,
                   uint32_t dst_s2, uint32_t dst_i2) {
  // Create command buffer.
  TsmDataMove *cmd = TsmNewDataMove();
  TsmDataMoveInstr inst = {I_CGRA, {0,}, {0,}};

  St_StrideIteration src_si = {src_s0, src_i0, src_s1, src_i1, src_s2, src_i2};
  St_StrideIteration dst_si = {dst_s0, dst_i0, dst_s1, dst_i1, dst_s2, dst_i2};

  cmd->GatherScatter(&inst, (uint64_t)src, (uint64_t)dst, size, &src_si, &dst_si);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteDataMove(cmd);
}
