# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pytest
try:
    import torch
    from torch.testing import assert_close as torch_assert_close
    HAS_TORCH = True
    HAS_PADDLE = False
except Exception:
    import paddle
    HAS_TORCH = False
    HAS_PADDLE = True

import triton
import triton.language as tl
from triton.testing import assert_close as tt_assert_close


@triton.jit
def matmul_tma_load_store(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        OUTPUT_F16: tl.constexpr  #
):
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)
    if OUTPUT_F16:
        c = c.to(tl.float16)

    tl.store(c_block_ptr, c)


def _randn(shape, dtype, trans=False):
    if HAS_TORCH:
        t = torch.randn(shape, device='cuda', dtype=dtype)
        return t.T if trans else t
    else:
        # ensure GPU as current device
        try:
            paddle.device.set_device('gpu')
        except Exception:
            pass
        t = paddle.randn(shape, dtype=dtype)
        return paddle.transpose(t, [1, 0]) if trans else t


def _empty(shape, dtype, like=None):
    if HAS_TORCH:
        dev = like.device if like is not None else 'cuda'
        return torch.empty(shape, device=dev, dtype=dtype)
    else:
        return paddle.empty(shape, dtype=dtype)


def _matmul(a, b):
    if HAS_TORCH:
        return torch.matmul(a, b)
    else:
        return paddle.matmul(a, b)


def _get_strides_2d(t):
    if HAS_TORCH:
        s = t.stride()
        return s[0], s[1]
    else:
        s = t.strides
        return s[0], s[1]


def _dtype_f16():
    return torch.float16 if HAS_TORCH else paddle.float16


def _dtype_f32():
    return torch.float32 if HAS_TORCH else paddle.float32


def _assert_close(x, y, rtol, atol, check_dtype=False):
    if HAS_TORCH:
        # keep original torch.testing.assert_close semantics
        torch_assert_close(x, y, rtol=rtol, atol=atol, check_dtype=check_dtype)
    else:
        # triton.testing.assert_close is framework-agnostic (compares in numpy)
        tt_assert_close(x, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize('M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_F16', [
    [64, 64, 16, 1, 4, False, True, False],
    [64, 64, 16, 1, 4, False, True, True],
    [128, 64, 32, 1, 4, False, True, False],
    [128, 64, 32, 1, 4, False, True, True],
    [64, 128, 32, 1, 4, False, True, False],
    [64, 128, 32, 1, 4, False, True, True],
    [128, 128, 64, 1, 4, False, True, False],
    [128, 128, 64, 1, 4, False, True, True],
])
def test_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16):
    # Create inputs (with optional transpose to set non-trivial strides)
    a_shape = (K, M) if TRANS_A else (M, K)
    b_shape = (N, K) if TRANS_B else (K, N)
    a = _randn(a_shape, _dtype_f16(), trans=TRANS_A)
    b = _randn(b_shape, _dtype_f16(), trans=TRANS_B)

    c_dtype = _dtype_f16() if OUTPUT_F16 else _dtype_f32()
    c = _empty((M, N), dtype=c_dtype, like=a)

    a_s0, a_s1 = _get_strides_2d(a)
    b_s0, b_s1 = _get_strides_2d(b)
    c_s0, c_s1 = _get_strides_2d(c)

    matmul_tma_load_store[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a_s0, stride_ak=a_s1,  #
        stride_bk=b_s0, stride_bn=b_s1,  #
        stride_cm=c_s0, stride_cn=c_s1,  #
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,  #
        num_warps=NUM_WARPS, num_ctas=NUM_CTAS,  #
        OUTPUT_F16=OUTPUT_F16)

    golden = _matmul(a, b)

    if HAS_TORCH:
        torch.set_printoptions(profile="full")
    _assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
