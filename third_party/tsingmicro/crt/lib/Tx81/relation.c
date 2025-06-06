//===------------------------ relation.c-----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::RelationOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __BoolEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                   uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                   elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolUnEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                     uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolUnEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                     elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolGreaterEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                          uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolGreaterEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                          elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolGreaterVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                     uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolGreaterVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                     elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolLessEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                       uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolLessEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                       elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolLessThenVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
                      uint32_t elem_count, uint16_t fmt) {
  // Create command buffer.
  TsmRelation *cmd = TsmNewRelation();
  TsmRelationInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  cmd->BoolLessThenVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                      elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}
