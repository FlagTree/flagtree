//===------------------------ memset.c ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Memset see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Memset(uint64_t *dst, uint32_t value, uint32_t elem_count, uint32_t s0,
              uint32_t i0, uint32_t s1, uint32_t i1, uint32_t s2, uint32_t i2,
              uint16_t fmt) {
  // Create command buffer.
  TsmPeripheral *cmd = TsmNewPeripheral();
  TsmDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  St_StrideIteration si = {s0, i0, s1, i1, s2, i2};
  cmd->Memset(&inst, (uint64_t)dst, value, elem_count, &si, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeletePeripheral(cmd);
}
