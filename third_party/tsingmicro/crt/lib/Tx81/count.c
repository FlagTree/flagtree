//===------------------------ count.c -------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Count see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Count(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt,
             uint64_t *p_wb_data0, uint64_t *p_wb_data1) {
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

  cmd->Count(&inst, (uint64_t)src, (uint64_t)dst, elem_count, (Data_Format)fmt,
             p_wb_data0, p_wb_data1);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeletePeripheral(cmd);
}
