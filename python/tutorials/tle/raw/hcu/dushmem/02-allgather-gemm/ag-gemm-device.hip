#include <hip/hip_runtime.h>
#include <stdint.h>

// Ready flags only. Triton performs the GEMM and calls these between tiles
// so local rows can be consumed while the copy engine is still pulling remote rows.

extern "C" __device__ __attribute__((used, always_inline)) void ag_mark_local_ready(uint64_t *ready, int rank) {
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_store(ready + rank, (uint64_t)1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" __device__ __attribute__((used, always_inline)) void ag_wait_ready(uint64_t *ready, int signal_index) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(ready + signal_index, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < 1) {
    }
  }
  __syncthreads();
}
