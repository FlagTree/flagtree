//===------------------------- tx81.c--------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

#ifdef __cplusplus
extern "C" {
#endif

bool is_contiguous(int *shape, int *strides, int elem_bytes) {
  int expected_stride = elem_bytes;
  for (int i = 0; i < 4; i++) {
    if (strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= shape[i];
  }
  return true;
}

// Used for kcore load/store data from/to spm
const int64_t spmMappingOffset = 0x30400000;

int8_t *get_spm_memory_mapping_wrapper(uint64_t ptr) {
#ifdef USE_SIM_MODE
  return get_spm_memory_mapping(ptr);
#else
  return (int8_t *)(ptr + spmMappingOffset);
#endif
}

#ifdef __cplusplus
}
#endif
