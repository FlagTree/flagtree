//===------------------------ mask_move.c ---------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::MaskMoveOp see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

void __MaskMove(uint64_t *src, uint64_t *target, uint32_t elem_count,
                uint64_t *mask, int32_t fmt) {
  TsmMaskDataMove *move = TsmNewMaskDataMove();
  TsmMaskDataMoveInstr inst = {I_CGRA,
                               {
                                   0,
                               },
                               {
                                   0,
                               }};

  move->MaskMove(&inst, (uint64_t)src, (uint64_t)mask, (uint64_t)target,
                 elem_count, (Data_Format)fmt);

  TsmExecute(&inst);

  TsmDeleteMaskDataMove(move);
}
