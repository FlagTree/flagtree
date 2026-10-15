// Copyright 2026- FlagOS Contributors
// Copyright 2026- Xcoresigma Technology Co., Ltd

// Elementwise add example for the @dialect(name="cann") source-JIT channel:
// user-supplied AscendC source shipped next to test_dialect_custom_ops.py,
// compiled standalone (AIV/Vector) by ccec on first use. It is not part of
// the prebuilt custom_ops.bc used by the registry-based ops.
#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
#define INTRINSIC_NO_ARGS(NAME) NAME()

// Contiguous 1D elementwise add on UB buffers: dst[i] = src0[i] + src1[i].
// n need not be a multiple of the per-repeat element count; the tail is
// masked off with a counter mask. One vadd covers at most 255 repeats
// (255 * 256B), longer inputs are split into chunks.
template <typename T>
__aiv__ __attribute__((always_inline)) void
add_impl(memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,
         memref_t<__ubuf__ T, 1> *dst) {
  constexpr int64_t ELEMS_PER_REPEAT = 256 / sizeof(T);
  constexpr int64_t MAX_ELEMS_PER_CALL = 255 * ELEMS_PER_REPEAT;

  __ubuf__ T *s0 = src0->aligned + src0->offset;
  __ubuf__ T *s1 = src1->aligned + src1->offset;
  __ubuf__ T *d = dst->aligned + dst->offset;

  const int64_t n = dst->sizes[0];
  for (int64_t off = 0; off < n; off += MAX_ELEMS_PER_CALL) {
    const int64_t len = MIN(n - off, MAX_ELEMS_PER_CALL);
    INTRINSIC_NO_ARGS(set_mask_count);
    INTRINSIC(set_vector_mask, 0, len);
    // Contiguous 1D layout: block strides 1 and repeat strides 8 (32B units),
    // so each repeat continues where the previous one ended.
    INTRINSIC(vadd, d + off, s0 + off, s1 + off,
              CEIL_DIV(len, ELEMS_PER_REPEAT), 1, 1, 1, 8, 8, 8);
    INTRINSIC_NO_ARGS(set_mask_norm);
  }
}

extern "C" {

#define REGISTER_CIFACE_ADD(TYPE_NAME, T)                                      \
  __aiv__                                                                      \
      __attribute__((always_inline)) void _mlir_ciface_custom_add_##TYPE_NAME( \
          memref_t<__ubuf__ T, 1> *src0, memref_t<__ubuf__ T, 1> *src1,        \
          memref_t<__ubuf__ T, 1> *dst) {                                      \
    add_impl<T>(src0, src1, dst);                                              \
  }

REGISTER_CIFACE_ADD(int32, int32_t)
REGISTER_CIFACE_ADD(half, half)
}
