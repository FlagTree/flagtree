// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

// This is a library fragment, not a kernel entry point.
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 0ULL
#endif
#include "Utils.h"
#include "kernel_operator.h"

template <typename T>
__aiv__ __attribute__((always_inline)) AscendC::LocalTensor<T>
cast_local_tensor(memref_t<__ubuf__ T, 1> *buffer) {
  AscendC::TBuffAddr address{};
  address.dataLen = buffer->sizes[0] * sizeof(T);
  address.bufferAddr = static_cast<uint32_t>(
      reinterpret_cast<uint64_t>(buffer->aligned + buffer->offset));
  address.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
  AscendC::LocalTensor<T> tensor;
  tensor.SetAddr(address);
  return tensor;
}

// Each byte contains two signed INT4 values, low nibble first.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_cast_int4_to_fp16(memref_t<__ubuf__ uint8_t, 1> *src,
                                      memref_t<__ubuf__ half, 1> *dst) {
  AscendC::PipeBarrier<PIPE_ALL>();
  auto packed = cast_local_tensor(src);
  auto values = cast_local_tensor(dst);
  AscendC::Cast(values, packed.ReinterpretCast<AscendC::int4b_t>(),
                AscendC::RoundMode::CAST_NONE, src->sizes[0] * 2);
  AscendC::PipeBarrier<PIPE_ALL>();
}
