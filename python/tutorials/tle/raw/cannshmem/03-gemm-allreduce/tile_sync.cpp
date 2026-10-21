// Copyright 2026-     FlagOS Contributors
// NV-style tile ready mark/wait (see
// nvshmem/03-gemm-allreduce/gemm-ar-device.cu). Symmetric int32
// ready[num_tiles]; producer SETs local slot, consumer waits all PEs.

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

extern "C" {

// inputs: ready (GM int32), tile_id, epoch | output: dummy (UB half)
__aiv__ void
_mlir_ciface_cannshmem_mark_tile_ready(memref_t<__gm__ int32_t, 1> *ready,
                                       int64_t tile_id, int64_t epoch,
                                       memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  __gm__ int32_t *slot = ready->aligned + ready->offset + tile_id;
  *slot = static_cast<int32_t>(epoch);
  PipeBarrier<PIPE_ALL>();
}

// inputs: ready (GM int32), tile_id, epoch, num_tiles | output: dummy
__aiv__ void _mlir_ciface_cannshmem_wait_tile_ready(
    memref_t<__gm__ int32_t, 1> *ready, int64_t tile_id, int64_t epoch,
    int64_t num_tiles, memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  (void)num_tiles;
  __gm__ int32_t *local = ready->aligned + ready->offset;
  const int npes = aclshmem_n_pes();
  const int32_t want = static_cast<int32_t>(epoch);
  for (int pe = 0; pe < npes; ++pe) {
    __gm__ int32_t *peer =
        (__gm__ int32_t *)aclshmem_ptr((__gm__ void *)local, pe);
    aclshmem_signal_wait_until(peer + tile_id, ACLSHMEM_CMP_GE, want);
  }
  // Catcoc AIV: barrier after stage ready so peer Cube payloads are visible.
  aclshmemx_barrier_all_vec();
}

} // extern "C"
