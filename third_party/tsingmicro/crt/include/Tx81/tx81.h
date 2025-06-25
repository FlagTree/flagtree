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

// Neural engine activate mode
enum ActFuncMode : int32_t {
  None = 0,
  ENRelu = 1,
  ENLeakRelu = 2,
};

#ifdef __cplusplus
extern "C" {
#endif

float set_value2float32(Data_Format fmt, int8_t *value);

bool is_contiguous(int *shape, int *strides, int elem_bytes);

// Use in simulation mode, return the spm address mapping
int8_t *get_spm_memory_mapping(uint64_t offset);
// Hardware mode will use add the spmMappingOffset to get the real spm address
// Simulation mode will call get_spm_memory_mapping
int8_t *get_spm_memory_mapping_wrapper(uint64_t offset);

#ifdef __cplusplus
}
#endif

#endif // CRT_TARGET_TX81_H
