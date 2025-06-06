//===---------------------- reduce.c --------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::TsmReduce, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __ReduceSum(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  // Create reduce command buffer.
  TsmReduce *cmd = TsmNewReduce();
  TsmReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};
  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceSum(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteReduce(cmd);
}

void __ReduceAvg(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  // Create reduce command buffer.
  TsmReduce *cmd = TsmNewReduce();
  TsmReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};
  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceAvg(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteReduce(cmd);
}

void __ReduceMax(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  // Create reduce command buffer.
  TsmReduce *cmd = TsmNewReduce();
  TsmReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};

  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceMax(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteReduce(cmd);
}

void __ReduceMin(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
                 uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt) {
  // Create reduce command buffer.
  TsmReduce *cmd = TsmNewReduce();
  TsmReduceInstr inst = {I_CGRA,
                         {
                             0,
                         },
                         {
                             0,
                         }};

  // TODO
  Data_Shape shape1 = {src_n, src_h, src_w, src_c};
  cmd->ReduceMin(&inst, (uint64_t)src, (uint64_t)dst, dim, shape1,
                 (Data_Format)fmt);
  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteReduce(cmd);
}
