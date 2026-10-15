# Copyright 2026- FlagOS Contributors
# Copyright 2026- Xcoresigma Technology Co., Ltd

"""
@dialect(name="cann") custom op end-to-end tests
===================================================

Exercises the kernel-level custom op channel (tle_raw.call with a @dialect
function object + output_indices) against a user-supplied AscendC source.
The example op is: add (int32 / fp16) via source JIT (format omitted)

Correctness only, no benchmarking.
"""

from pathlib import Path

import torch
import torch_npu
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect

DEVICE = "npu"

# ══════════════════════════════════════════════════════════════════════════
# Dialect-bound custom op
# ══════════════════════════════════════════════════════════════════════════


# format omitted: the input defaults to source; add.cpp is JIT-compiled to
# bitcode with ccec on first use and cached under ~/.triton/cache (vector
# core -> dav-c220-vec). Each dtype is a separately exported extern symbol.
@dialect(
    name="cann",
    file="./add.cpp",
    extern_func_name="custom_add_int32",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def add_int32(*args, **kwargs):
    ...


@dialect(
    name="cann",
    file="./add.cpp",
    extern_func_name="custom_add_half",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def add_half(*args, **kwargs):
    ...


# ══════════════════════════════════════════════════════════════════════════
# Kernel
# ══════════════════════════════════════════════════════════════════════════


@triton.jit
def custom_add_kernel(a, b, out, N_ELEMENTS: tl.constexpr, DTYPE: tl.constexpr):
    offs = tl.arange(0, N_ELEMENTS)

    # PIPE_V compute ops read/write UB: stage both operands GM -> UB first.
    a_ub = tle.dsa.alloc([N_ELEMENTS], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([N_ELEMENTS], dtype=DTYPE, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(a + offs, a_ub, [N_ELEMENTS])
    tle.dsa.copy(b + offs, b_ub, [N_ELEMENTS])

    # args[2] (`out_t`) is the output/aliased operand; the exported extern
    # function takes (src0, src1, dst).
    out_t = tl.zeros([N_ELEMENTS], dtype=DTYPE)
    if DTYPE == tl.int32:
        out_t = tle_raw.call(add_int32, [tle.dsa.to_tensor(a_ub), tle.dsa.to_tensor(b_ub), out_t], output_indices=[2])
    elif DTYPE == tl.float16:
        out_t = tle_raw.call(add_half, [tle.dsa.to_tensor(a_ub), tle.dsa.to_tensor(b_ub), out_t], output_indices=[2])

    tl.store(out + offs, out_t)


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════


def test_custom_add(torch_dtype, tl_dtype, tol, n_elements):
    if torch_dtype == torch.int32:
        a = torch.randint(-1000, 1000, (n_elements, ), dtype=torch_dtype, device=DEVICE)
        b = torch.randint(-1000, 1000, (n_elements, ), dtype=torch_dtype, device=DEVICE)
    else:
        torch.manual_seed(0)
        a = torch.randn((n_elements, ), dtype=torch_dtype, device=DEVICE)
        b = torch.randn((n_elements, ), dtype=torch_dtype, device=DEVICE)

    ref = a + b

    out = torch.zeros((n_elements, ), dtype=torch_dtype, device=DEVICE)
    custom_add_kernel[(1, 1, 1)](a, b, out, n_elements, tl_dtype)
    torch_npu.npu.synchronize()

    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=tol, atol=tol)
    print(f"[PASS] dialect add correctness ({torch_dtype}, n_elements={n_elements})")


def main():
    for torch_dtype, tl_dtype, tol in ((torch.int32, tl.int32, 0), (torch.float16, tl.float16, 1e-3)):
        for n_elements in [32, 256, 1024]:
            test_custom_add(torch_dtype, tl_dtype, tol, n_elements)
    print("\nAll @dialect(name='cann') custom op tests passed.")


if __name__ == "__main__":
    main()
