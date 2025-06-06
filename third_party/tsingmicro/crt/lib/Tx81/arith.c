//===------------------------ arith.c ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::ArithOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __AddVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
    uint32_t elem_count, RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->AddVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             round, (Data_Format) fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __SubVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
    uint32_t elem_count, RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->SubVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             round, (Data_Format) fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __MulVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
    uint32_t elem_count, RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->MulVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             round, (Data_Format) fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}


void __DivVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
    uint32_t elem_count, RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA, {0,}, {0,}};

  cmd->DivVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
             round, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __AddVS(uint64_t *src0, uint32_t src1, uint64_t *dst, uint32_t elem_count,
             RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->AddVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count, round,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __SubVS(uint64_t *src0, uint32_t src1, uint64_t *dst, uint32_t elem_count,
             RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->SubVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count, round,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __MulVS(uint64_t *src0, uint32_t src1, uint64_t *dst, uint32_t elem_count,
             RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->MulVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count, round,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}

void __DivVS(uint64_t *src0, uint32_t src1, uint64_t *dst, uint32_t elem_count,
             RND_MODE round, uint16_t fmt) {
  // Create command buffer.
  TsmArith *cmd = TsmNewArith();
  TsmArithInstr inst = {I_CGRA,
                        {
                            0,
                        },
                        {
                            0,
                        }};

  cmd->DivVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count, round,
             (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteArith(cmd);
}
