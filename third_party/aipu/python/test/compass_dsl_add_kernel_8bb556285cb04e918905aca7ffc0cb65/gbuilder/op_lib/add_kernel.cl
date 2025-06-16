#include <aipu/tvm_aipu.h>

GEN_DMA_DIRECT_EXT2INT(kGlobal, kLsram);
GEN_DMA_DIRECT_INT2EXT(kLsram, kGlobal);

__kernel void add_kernel(__global float* var_11, __global float* var_20, __global float* var_31, int var_14, int var_33, int var_35, int var_37, int var_3, int var_39, int var_41);

__kernel void add_kernel(__global float* var_11, __global float* var_20, __global float* var_31, int var_14, int var_33, int var_35, int var_37, int var_3, int var_39, int var_41) {
  __lsram float buf[1024];
  __lsram float buf_1[1024];
  __lsram float buf_2[1024];
  int var_0 = get_local_size(0);
  int var_1 = get_local_id(0);
  int var_4 = (var_3 * var_0);
  int var_5 = (var_4 + var_1);
  int var_8 = (var_5 * 1024);
  int var_9 = var_8;
  int var_12 = (var_9 + 1024);
  int var_15 = var_14;
  int var_16 = min(var_12, var_15);
  int var_17 = max(var_16, var_9);
  int var_18 = (var_17 - var_9);
  int cse_var_1 = (var_18 * 4);
  DmaDirect_kGlobal_to_kLsram((int)buf, (int)(var_11 + var_9), cse_var_1, cse_var_1, cse_var_1, cse_var_1);
  DmaDirect_kGlobal_to_kLsram((int)buf_1, (int)(var_20 + var_9), cse_var_1, cse_var_1, cse_var_1, cse_var_1);
  for (int var_24 = 0; var_24 < (1024 - 0); var_24 += 8) {
    float8 var_26 = __vload((__lsram float8*)(buf + var_24), ALL_TRUE_w);
    float8 var_28 = __vload((__lsram float8*)(buf_1 + var_24), ALL_TRUE_w);
    float8 var_29 = (var_26 + var_28);
    __vstore(var_29, (__lsram float8*)(buf_2 + var_24), ALL_TRUE_w);
  }
  DmaDirect_kLsram_to_kGlobal((int)(var_31 + var_9), (int)buf_2, cse_var_1, cse_var_1, cse_var_1, cse_var_1);
  barrier(CLK_LOCAL_MEM_FENCE);return;
  barrier(CLK_LOCAL_MEM_FENCE);
}
