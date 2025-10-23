# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
import triton
import triton.language as tl
import test_common
import torch
import torch_npu

def standard_(x0,dtype):
    res, index = torch.min(x0, 0,keepdim=True)
    return res

@triton.jit
def triton_min_vector(in_ptr0, out_ptr0, N : tl.constexpr,NUMEL : tl.constexpr):
    idx_block = tl.arange(0,NUMEL)
    if in_ptr0.dtype == tl.int8:
        padding = 127
    else :
        padding = float('inf')
    x=tl.load(in_ptr0+idx_block, mask = idx_block<N,other = padding)

    ret = tl.min(x,0)
    tl.store(out_ptr0+idx_block, ret, mask = idx_block<1)

types=[
    (torch.float32,'float32'),
    # (torch.float16,'float16'),  TODO : fix reduceConverter bug
    # (torch.bfloat16,'bfloat16'),  waiting for supporting or testing
    # (torch.int8,'int8'),  TODO : fix compiler bug
    # (torch.int16,'int16'),  waiting for supporting or testing
    # (torch.int32,'int32'),  waiting for supporting or testing
    # (torch.int64,'int64'),  waiting for supporting or testing
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes=[
    (3,32),
    (-32,32),
    (37,64),
    (-256,256),
    (781,1024),
]

map_for_64_t = {37:31}

@pytest.mark.skip(reason="randomly failed")
@pytest.mark.parametrize('dtype, sigtype',types)
@pytest.mark.parametrize('N, NUMEL',shapes)
def test_reduce_dim0_common(dtype, sigtype, N, NUMEL):
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N] if N in map_for_64_t else N

    print(f"elementwise : ({N},) {dtype} {sigtype}")

    x0 = test_common.generate_tensor(shape = (N,),dtype = sigtype)

    ans = standard_(x0, dtype)
    x0=x0.npu()

    output = torch.zeros((1,), dtype = dtype).npu()
    triton_min_vector[1,1,1](x0, output, N = N,NUMEL = NUMEL, debug = True)

    test_common.validate_cmp(sigtype,output,ans)
