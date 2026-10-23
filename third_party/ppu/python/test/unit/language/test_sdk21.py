"""Execute memory, inline-move and MMA kernels through the SDK 2.1 toolchain."""

import os

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    os.environ.get("FLAGTUNE_RUN_PPU_SDK_TESTS") != "1",
    reason="set FLAGTUNE_RUN_PPU_SDK_TESTS=1 on an idle SDK 2.1 PPU")


@triton.jit
def _memory_kernel(x, out, N: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + offsets, offsets < N, other=0)
    moved = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=r,r", args=[value], dtype=tl.int32, is_pure=True,
                                      pack=1)
    tl.store(out + offsets, moved + 1, offsets < N)


@triton.jit
def _mma_kernel(a, b, out, B: tl.constexpr):
    offsets = tl.arange(0, B)
    lhs = tl.load(a + offsets[:, None] * B + offsets[None, :])
    rhs = tl.load(b + offsets[:, None] * B + offsets[None, :])
    result = tl.dot(lhs, rhs)
    tl.store(out + offsets[:, None] * B + offsets[None, :], result)


def test_sdk21_memory_and_inline_move():
    x = torch.arange(1025, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    _memory_kernel[(triton.cdiv(x.numel(), 256), )](x, out, x.numel(), 256)
    torch.testing.assert_close(out, x + 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdk21_mma(dtype):
    torch.manual_seed(0)
    a = torch.randn((64, 64), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    compiled = _mma_kernel[(1, )](a, b, out, 64, num_warps=4)
    assert "mma" in compiled.asm["llir"].lower()
    torch.testing.assert_close(out, a.float() @ b.float(), rtol=0.03, atol=0.05)
