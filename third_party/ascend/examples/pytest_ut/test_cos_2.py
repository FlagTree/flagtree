# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest

import triton
import triton.language as tl
import time

import torch
import torch_npu
import test_common

def torch_cos(x0):
    res = torch.cos(x0)
    return res

@triton.jit
def triton_cos(in_ptr0, out_ptr0, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.cos(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 32, 2048, 64],
                         ])
def test_cos(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype)
    y_ref = torch_cos(x0)
    tyname = test_common.get_triton_sig_typename(dtype)
    y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
    x0 = x0.npu()
    triton_cos[ncore, 1, 1](x0, y_cal, xblock, xblock_sub, debug=True)
    test_common.validate_cmp(dtype, y_cal, y_ref)
