// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "mask_common.h"

// Stable compaction. Only dst[0:count[0]] is defined; dst's tail is
// unspecified.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_mask_float(memref_t<__ubuf__ float, 1> *src,
                                      memref_t<__ubuf__ uint16_t, 1> *mask,
                                      memref_t<__ubuf__ float, 1> *dst,
                                      memref_t<__ubuf__ int32_t, 1> *count) {
  AscendC::PipeBarrier<PIPE_ALL>();
  auto input = mask_local_tensor(src);
  auto bits = mask_local_tensor(mask);
  auto output = mask_local_tensor(dst);
  auto number = mask_local_tensor(count);
  uint64_t found = 0;
  AscendC::GatherMaskParams params{1, 1, 8, 1};
  AscendC::GatherMask(output, input, bits.ReinterpretCast<uint32_t>(), true,
                      src->sizes[0], params, found);
  AscendC::PipeBarrier<PIPE_ALL>();
  // Read inside the primitive: no intervening op may overwrite this register.
  found = get_rsvd_cnt();
  AscendC::Duplicate(number, static_cast<int32_t>(found), 8);
  AscendC::PipeBarrier<PIPE_ALL>();
}
