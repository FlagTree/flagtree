// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "mask_common.h"

// Bit i (least-significant bit first) records src[i] == scalar.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_compare_scalar_float(memref_t<__ubuf__ float, 1> *src,
                                         float scalar,
                                         memref_t<__ubuf__ uint16_t, 1> *dst) {
  AscendC::PipeBarrier<PIPE_ALL>();
  auto input = mask_local_tensor(src);
  auto mask = mask_local_tensor(dst);
  AscendC::CompareScalar(mask.ReinterpretCast<uint8_t>(), input, scalar,
                         AscendC::CMPMODE::EQ, src->sizes[0]);
  AscendC::PipeBarrier<PIPE_ALL>();
}
