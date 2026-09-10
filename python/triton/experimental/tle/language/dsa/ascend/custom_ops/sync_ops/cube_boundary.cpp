// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 0ULL
#endif
#include "kernel_operator.h"

// Local CUBE pipeline barriers; these do not synchronize cores or allocate
// state.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_cube_begin(int32_t token) {
  AscendC::PipeBarrier<PIPE_ALL>();
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_cube_end(int32_t token) {
  AscendC::PipeBarrier<PIPE_ALL>();
}
