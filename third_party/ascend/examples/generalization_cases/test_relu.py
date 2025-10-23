import triton
import triton.language as tl
import torch
import pytest
import test_common
import triton.language.extra.ascend.libdevice as libdevice
from test_common import TestUtils
import math

def torch_relu(x0, x1):
    res = x0 + torch.relu(x1)
    return res


@triton.jit
def triton_relu(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tmp0 + libdevice.relu(tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_relu(dtype, shape):
    # 生成数据
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()

    numel = x0.numel()
    ncore = 1 if numel <= 32 else 32
    xblock = math.ceil(numel / ncore)
    xblock_sub = numel if numel <= ncore else math.ceil(numel / ncore)

    # torch结果
    torch_res = torch_relu(x0, x1)
    # triton结果
    triton_res = test_common.generate_tensor(shape, dtype).npu()
    triton_relu[ncore, 1, 1](x0, x1, triton_res, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
