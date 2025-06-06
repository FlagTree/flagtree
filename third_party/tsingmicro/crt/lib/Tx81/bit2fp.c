//===------------------------ bit2fp.c ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Bit2Fp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Bit2Fp(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmPeripheral *cmd = TsmNewPeripheral();
  TsmPeripheralInstr inst = {I_CGRA,
                             {
                                 0,
                             },
                             {
                                 0,
                             }};
  ;

  cmd->Bit2Fp(&inst, (uint64_t)src, (uint64_t)dst, elem_count,
              (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeletePeripheral(cmd);
}
