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

void __EqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->EqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst, elem_count,
               (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __UnEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->UnEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                 elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __GreaterEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->GreaterEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                      elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __GreaterVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->GreaterVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                 elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __LessEqualVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->LessEqualVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                   elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __LessThenVV(uint64_t *src0, uint64_t *src1, uint64_t *dst,
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

  cmd->LessThenVV(&inst, (uint64_t)src0, (uint64_t)src1, (uint64_t)dst,
                  elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                   (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolUnEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolUnEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                     (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolGreaterEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolGreaterEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst,
                          elem_count, (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolGreaterVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolGreaterVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                     (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolLessEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolLessEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                       (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __BoolLessThenVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->BoolLessThenVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                      (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __EqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->EqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
               (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __UnEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->UnEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                 (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __GreaterEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->GreaterEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                      (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __GreaterVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->GreaterVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                 (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __LessEqualVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->LessEqualVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                   (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}

void __LessThenVS(uint64_t *src0, uint32_t src1, uint64_t *dst,
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

  cmd->LessThenVS(&inst, (uint64_t)src0, src1, (uint64_t)dst, elem_count,
                  (Data_Format)fmt);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
  TsmDeleteRelation(cmd);
}
