//===------------------------ nchw2nhwc.c ---------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Nchw2nhwc see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Nchw2nhwc(uint64_t *src, uint16_t src_n, uint16_t src_h, uint16_t src_w,
               uint16_t src_c, uint64_t *dst, uint16_t dst_n, uint16_t dst_h,
               uint16_t dst_w, uint16_t dst_c, uint16_t fmt) {
  // Create command buffer.
  TsmDataMove *cmd = TsmNewDataMove();
  TsmDataMoveInstr inst = {I_CGRA, {0,}, {0,}};

  Data_Shape shape1 = { src_n, src_h, src_w, src_c };
  Data_Shape shape2 = { dst_n, dst_h, dst_w, dst_c };
  cmd->Nchw2nhwc(&inst, (uint64_t) src, shape1, (uint64_t) dst, shape2,
  (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteDataMove(cmd);
}
