// Copyright 2026-     FlagOS Contributors
// Ring shift via aclshmem_ptr: each PE reads the next PE's symmetric buffer
// into a local UB tile, then barrier_all. Device symbols come from
// library="cannshmem_mlir" late-linked AscendNPU-IR Shmem template bitcode.

#include "Utils.h"

#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

extern "C" {
__aicore__ int _mlir_ciface_aclshmem_my_pe(void);
__aicore__ int _mlir_ciface_aclshmem_n_pes(void);
__aicore__ void _mlir_ciface_aclshmem_barrier_all(void);
__aicore__ void _mlir_ciface_aclshmem_ptr_float(memref_t<__gm__ float, 1> *symmPtr,
                                                memref_t<__gm__ float, 1> *ptr, int pe);

// Operands (tle_raw.call order with output_indices=[1]):
//   inputs:  symm (GM), n
//   outputs: out (UB)
__aiv__ void _mlir_ciface_cannshmem_ring_read(memref_t<__gm__ float, 1> *symm, int64_t n,
                                              memref_t<__ubuf__ float, 1> *out) {
  const int me = _mlir_ciface_aclshmem_my_pe();
  const int npes = _mlir_ciface_aclshmem_n_pes();
  const int peer = (me + 1) % npes;

  memref_t<__gm__ float, 1> remote;
  _mlir_ciface_aclshmem_ptr_float(&remote, symm, peer);

  __gm__ float *src = remote.aligned + remote.offset;
  __ubuf__ float *dst = out->aligned + out->offset;
  // Small N: one aligned GM→UB burst (bytes must be 32B-aligned for MTE).
  const uint32_t nbytes = static_cast<uint32_t>(n * sizeof(float));
  INTRINSIC(copy_gm_to_ubuf_align_b32, dst, src, 0, 1, nbytes, 0, 0, 0, 0);

  _mlir_ciface_aclshmem_barrier_all();
}
}
