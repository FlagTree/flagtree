//===------------------------ int8_tf32.c ---------------------------------===//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::INT8_TF32 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __INT8_TF32(uint64_t *src, uint32_t zp, uint64_t *dst, uint32_t elem_count) {
  // Create command buffer.
  TsmConvert *cmd = TsmNewConvert();
  TsmConvertInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->INT8_TF32(&inst, (uint64_t)src, zp, (uint64_t)dst, elem_count);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteConvert(cmd);
}
