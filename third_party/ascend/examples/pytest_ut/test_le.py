import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu

def standard_binary(x0, y0):
    res = x0 <= y0
    return res

@triton.jit
def triton_elementwise_binary(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    y = tl.load(in_ptr1 + idx_block, mask=idx_block < N)
    ret = x <= y
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)

types = [
    (torch.float32, 'float32'),
    (torch.float16, 'float16'),
    # (torch.bfloat16, 'bfloat16'),
    (torch.int8, 'int8'),
    (torch.int16, 'int16'),
    (torch.int32, 'int32'),
    (torch.int64, 'int64'),
]

shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]

map_for_64_t = {37: 31}

@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('N,NUMEL', shapes)
def test_elementwsie_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == "int64":
        N = map_for_64_t[N] if N in map_for_64_t else N

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    y0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    ans = standard_binary(x0, y0)
    out = torch.zeros((N,), dtype=torch.bool).npu()
    triton_elementwise_binary[1, 1, 1](x0, y0, out, N, NUMEL)
    test_common.validate_cmp(sigtype, out, ans)
