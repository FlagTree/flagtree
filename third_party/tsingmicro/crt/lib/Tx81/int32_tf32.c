//===------------------------ int32_tf32.c --------------------------------===//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::INT32_TF32 see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __INT32_TF32(uint64_t *src, uint64_t *dst, uint32_t elem_count,
                RND_MODE round) {
  // Create command buffer.
  TsmConvert *cmd = TsmNewConvert();
  TsmConvertInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->INT32_TF32(&inst, (uint64_t)src, (uint64_t)dst, elem_count, round);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteConvert(cmd);
}
