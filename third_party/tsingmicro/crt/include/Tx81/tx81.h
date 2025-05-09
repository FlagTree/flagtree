//===----------------------- tx81.h ---------------------------*- C -*-----===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef CRT_TARGET_TX81_H
#define CRT_TARGET_TX81_H

#include "instr_adapter.h"
#include "instr_def.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

enum MemorySpace : int32_t {
  UNKNOWN = 0,
  SPM = 1,
  DDR = 2,
};

#endif // CRT_TARGET_TX81_H
