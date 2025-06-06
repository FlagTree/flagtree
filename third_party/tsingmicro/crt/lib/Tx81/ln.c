//===------------------------ ln.c ----------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Ln see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __Ln(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmTranscendental *cmd = TsmNewTranscendental();
  TsmTranscendentalInstr inst = {I_CGRA,
                                 {
                                     0,
                                 },
                                 {
                                     0,
                                 }};

  cmd->Ln(&inst, (uint64_t)src, (uint64_t)dst, elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteTranscendental(cmd);
}
