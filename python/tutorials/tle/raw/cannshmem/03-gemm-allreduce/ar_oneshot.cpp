// Copyright 2026-     FlagOS Contributors
// Tile oneshot allreduce (Catcoc AIV epilogue): barrier + peer-get + sum.
// ``aclshmemx_barrier_all_vec`` matches Catcoc MatmulAllReduce AIV path.

#include "Utils.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

namespace {
constexpr uint32_t kChunk = 2048;
}

extern "C" {

// Operands (tle_raw.call order with output_indices=[5]):
//   inputs:  symm (GM), out (GM), symm_off, out_off, n
//   outputs: dummy (UB)
__aiv__ void
_mlir_ciface_cannshmem_ar_oneshot(memref_t<__gm__ half, 1> *symm,
                                  memref_t<__gm__ half, 1> *out,
                                  int64_t symm_off, int64_t out_off, int64_t n,
                                  memref_t<__ubuf__ half, 1> *dummy) {
  (void)dummy;
  __gm__ half *local = symm->aligned + symm->offset;
  __gm__ half *gout = out->aligned + out->offset;
  const int me = aclshmem_my_pe();
  const int npes = aclshmem_n_pes();

  TPipe pipe;
  TBuf<> tbuf;
  pipe.InitBuffer(tbuf, kChunk * sizeof(half) * 2);
  LocalTensor<half> acc = tbuf.Get<half>(kChunk);
  LocalTensor<half> tmp =
      tbuf.GetWithOffset<half>(kChunk, kChunk * sizeof(half));

  aclshmemx_barrier_all_vec();

  for (int64_t off = 0; off < n; off += kChunk) {
    uint32_t len = static_cast<uint32_t>(
        (n - off) < static_cast<int64_t>(kChunk) ? (n - off) : kChunk);
    uint32_t len_align = (len + 15u) & ~15u;

    GlobalTensor<half> g_local;
    g_local.SetGlobalBuffer(local + symm_off + off);
    DataCopy(acc, g_local, len_align);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    for (int pe = 0; pe < npes; ++pe) {
      if (pe == me) {
        continue;
      }
      __gm__ half *rbase =
          (__gm__ half *)aclshmem_ptr((__gm__ void *)local, pe);
      GlobalTensor<half> g_remote;
      g_remote.SetGlobalBuffer(rbase + symm_off + off);
      DataCopy(tmp, g_remote, len_align);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
      Add(acc, acc, tmp, len_align);
      PipeBarrier<PIPE_V>();
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<half> g_out;
    g_out.SetGlobalBuffer(gout + out_off + off);
    DataCopy(g_out, acc, len_align);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }

  aclshmemx_barrier_all_vec();
}

} // extern "C"
