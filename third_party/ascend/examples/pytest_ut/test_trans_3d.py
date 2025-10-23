# -*- coding: utf-8 -*-
# # Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import logging
import math

import pytest
import test_common
import torch

import triton
import triton.language as tl


@triton.jit
def fn_npu_102(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, KB: tl.constexpr):
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    kidx = tl.arange(0, KB)
    idx = yidx[:, None, None] * ZB * KB + zidx[None, :, None] * KB + kidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.trans(X, 1, 0, 2)

    oidx = (
        zidx[:, None, None] * YB * KB + yidx[None, :, None] * KB + kidx[None, None, :]
    )

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_021(output_ptr, x_ptr, YB: tl.constexpr, ZB: tl.constexpr, KB: tl.constexpr):
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    kidx = tl.arange(0, KB)
    idx = yidx[:, None, None] * ZB * KB + zidx[None, :, None] * KB + kidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.trans(X, (0, 2, 1))

    oidx = (
        yidx[:, None, None] * ZB * KB + kidx[None, :, None] * ZB + zidx[None, None, :]
    )

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize("shape", [(23, 5, 31)])
@pytest.mark.parametrize("dtype", ["float32"])
def test_permute_3d(shape, dtype):
    logging.debug(f"dtype:{dtype} shape:{shape}")

    data_type = eval("torch." + dtype)
    x = torch.randint(low=0, high=2, size=shape, dtype=data_type).npu()

    triton_res = torch.empty((shape[1], shape[0], shape[2]), dtype=data_type).npu()
    torch_res = torch.permute(x, (1, 0, 2))
    fn_npu_102[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])
    test_common.validate_cmp(dtype, triton_res, torch_res)

    triton_res = torch.empty((shape[0], shape[2], shape[1]), dtype=data_type).npu()
    torch_res = torch.permute(x, (0, 2, 1))
    fn_npu_021[1, 1, 1](triton_res, x, shape[0], shape[1], shape[2])
    test_common.validate_cmp(dtype, triton_res, torch_res)
