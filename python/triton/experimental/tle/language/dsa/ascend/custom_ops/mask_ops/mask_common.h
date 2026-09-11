// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// These are library fragments, not kernel entry points. Suppress the strong
// per-translation-unit tiling-key definition emitted by AscendC headers.
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 0ULL
#endif
#include "Utils.h"
#include "kernel_operator.h"

// TLE passes contiguous UB memref descriptors, including a possible offset.
template <typename T>
__aiv__ __attribute__((always_inline)) AscendC::LocalTensor<T>
mask_local_tensor(memref_t<__ubuf__ T, 1> *buffer) {
  AscendC::TBuffAddr address{};
  address.dataLen = buffer->sizes[0] * sizeof(T);
  address.bufferAddr = static_cast<uint32_t>(
      reinterpret_cast<uint64_t>(buffer->aligned + buffer->offset));
  address.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
  AscendC::LocalTensor<T> tensor;
  tensor.SetAddr(address);
  return tensor;
}
