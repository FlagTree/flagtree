# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def ptr_gather_kernel(src, dst):
    offsets = tl.arange(0, 8)
    pointers = src + offsets[None, :] + 8 * offsets[:, None]
    value = tl.gather(
        pointers,
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    )
    output_offsets = tl.arange(0, 4)
    tl.store(dst + output_offsets, value)


@triton.jit
def ptr_scatter_kernel(src, dst):
    offsets = tl.arange(0, 8)
    pointers = dst + offsets[None, :] + 8 * offsets[:, None]
    input_offsets = tl.arange(0, 4)
    value = tl.load(src + input_offsets)
    tl.scatter(
        pointers,
        value,
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    )


def test_tensor_view_ptr_gather():
    src = torch.arange(64, dtype=torch.float32, device="npu").reshape(8, 8)
    actual = torch.empty(4, dtype=torch.float32, device="npu")

    ptr_gather_kernel[(1, )](src, actual)

    expected = src[[0, 2, 4, 6], [1, 3, 5, 7]]
    torch.testing.assert_close(actual, expected)


def test_tensor_view_ptr_scatter():
    src = torch.tensor([11, 23, 45, 67], dtype=torch.float32, device="npu")
    actual = torch.zeros((8, 8), dtype=torch.float32, device="npu")

    ptr_scatter_kernel[(1, )](src, actual)

    expected = torch.zeros_like(actual)
    expected[[0, 2, 4, 6], [1, 3, 5, 7]] = src
    torch.testing.assert_close(actual, expected)
