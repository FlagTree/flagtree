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
#include "lib_log.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

enum MemorySpace : int32_t {
  UNKNOWN = 0,
  SPM = 1,
  DDR = 2,
};

// Neural engine activate mode
enum ActFuncMode : int32_t {
  None = 0,
  ENRelu = 1,
  ENLeakRelu = 2,
};

inline uint64_t spm_print_offset(uint64_t addr) {
  return (uint64_t)addr + 0x030400000;
}
#endif // CRT_TARGET_TX81_H
