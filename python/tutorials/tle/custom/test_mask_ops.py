# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness checks for compare_scalar and gather_mask on Ascend."""

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import compare_scalar, gather_mask


@triton.jit
def compare_kernel(X, Mask, scalar, N: tl.constexpr):
    values = tl.load(X + tl.arange(0, N))
    mask = tl.full((N // 16, ), 0, tl.uint16)
    mask = tle.dsa.ascend.raw("compare_scalar", values, scalar, out=mask)
    tl.store(Mask + tl.arange(0, N // 16), mask)


@triton.jit
def gather_kernel(X, Mask, Out, Count, N: tl.constexpr):
    values = tl.load(X + tl.arange(0, N))
    mask = tl.load(Mask + tl.arange(0, N // 16))
    selected = tl.full((N, ), 0, tl.float32)
    count = tl.full((8, ), 0, tl.int32)
    selected, count = tle.dsa.ascend.raw("gather_mask", values, mask, out=[selected, count])
    found = tl.sum(tl.where(tl.arange(0, 8) == 0, count, 0), 0)
    tl.store(Out + tl.arange(0, N), selected, tl.arange(0, N) < found)
    tl.store(Count + tl.arange(0, 8), count)


@triton.jit
def composed_kernel(IDs, Out, Count, expert, R: tl.constexpr, N: tl.constexpr):
    lane = tl.arange(0, N)
    count_total = 0
    for start in range(0, R, N):
        ids = tl.load(IDs + start + lane, start + lane < R, other=-1)
        mask = tl.full((N // 16, ), 0, tl.uint16)
        mask = tle.dsa.ascend.raw("compare_scalar", ids.to(tl.float32), expert.to(tl.float32), out=mask)
        selected = tl.full((N, ), 0, tl.float32)
        count = tl.full((8, ), 0, tl.int32)
        selected, count = tle.dsa.ascend.raw("gather_mask", lane.to(tl.float32), mask, out=[selected, count])
        found = tl.sum(tl.where(tl.arange(0, 8) == 0, count, 0), 0)
        tl.store(Out + count_total + lane, selected.to(tl.int32) + start, lane < found)
        count_total += found
    tl.store(Count, count_total)


def _pack(bits):
    words = bits.reshape(-1, 16).astype(np.uint32)
    return np.sum(words << np.arange(16, dtype=np.uint32), axis=1).astype(np.uint16)


def test_compare_scalar():
    rng = np.random.default_rng(17)
    cases = 0
    for n in (256, 512, 1024, 2048, 4096):
        values = rng.integers(-4, 5, size=n).astype(np.float32) * 0.5
        values[:6] = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.5]
        x = torch.from_numpy(values).to("npu")
        mask = torch.empty(n // 16, dtype=torch.uint16, device="npu")
        for scalar in (0.0, -0.0, 1.5, np.inf, -np.inf, np.nan, 32.0):
            compare_kernel[(1, )](x, mask, float(scalar), N=n)
            np.testing.assert_array_equal(mask.cpu().numpy(), _pack(values == scalar))
            cases += 1
    print(f"[PASS] compare_scalar: {cases} size/scalar cases")


def test_gather_mask():
    rng = np.random.default_rng(23)
    cases = 0
    for n in (256, 512, 1024, 2048, 4096):
        values = np.arange(n, dtype=np.float32) * 0.125 - 0.5
        values[:4] = [-0.0, np.inf, -np.inf, np.nan]
        x = torch.from_numpy(values).to("npu")
        boundary = np.zeros(n, dtype=bool)
        boundary[[0, 15, 16, 31, 32, 63, 64, 127, 128, 255, n - 1]] = True
        for bits in (np.zeros(n, dtype=bool), np.ones(n, dtype=bool), np.arange(n) % 2 == 1, rng.random(n)
                     < 0.1, boundary):
            mask = torch.from_numpy(_pack(bits)).to("npu")
            out = torch.empty_like(x)
            count = torch.empty(8, dtype=torch.int32, device="npu")
            gather_kernel[(1, )](x, mask, out, count, N=n)
            found = int(bits.sum())
            np.testing.assert_array_equal(count.cpu().numpy(), np.full(8, found))
            # Compare bits, including NaNs, infinities and the sign of zero.
            np.testing.assert_array_equal(out.cpu().numpy()[:found].view(np.uint32), values[bits].view(np.uint32))
            cases += 1
    print(f"[PASS] gather_mask: {cases} size/mask cases")


def test_composed_routing():
    rng = np.random.default_rng(29)
    cases = 0
    for r in (1, 255, 256, 257, 4095, 4096, 4097, 12288):
        for kind in ("random", "none", "all", "boundary"):
            ids = rng.integers(0, 4, size=r).astype(np.int32)
            if kind == "none":
                ids.fill(0)
            elif kind == "all":
                ids.fill(3)
            elif kind == "boundary":
                ids.fill(0)
                ids[np.unique(np.array([0, min(15, r - 1), min(16, r - 1), r - 1]))] = 3
            x = torch.from_numpy(ids).to("npu")
            out = torch.empty(r, dtype=torch.int32, device="npu")
            count = torch.empty(1, dtype=torch.int32, device="npu")
            composed_kernel[(1, )](x, out, count, 3, R=r, N=4096)
            expected = np.flatnonzero(ids == 3).astype(np.int32)
            assert int(count.cpu()[0]) == len(expected)
            np.testing.assert_array_equal(out.cpu().numpy()[:len(expected)], expected)
            cases += 1
    print(f"[PASS] composed routing: {cases} length/distribution cases")


def test_graph_replay():
    n = 4096
    x = torch.zeros(n, dtype=torch.int32, device="npu")
    out = torch.empty_like(x)
    count = torch.empty(1, dtype=torch.int32, device="npu")
    for _ in range(3):
        composed_kernel[(1, )](x, out, count, 3, R=n, N=n)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        composed_kernel[(1, )](x, out, count, 3, R=n, N=n)
    for value in (3, 0, 3):
        x.fill_(value)
        graph.replay()
        found = n if value == 3 else 0
        assert int(count.cpu()[0]) == found
        np.testing.assert_array_equal(out.cpu().numpy()[:found], np.arange(found, dtype=np.int32))
    print("[PASS] graph replay with changed inputs")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _init(cls, *args, **kwargs):
    # Match the custom-op dispatcher initialization without generating IR.
    op = cls.__new__(cls)
    op.arg_type = {}
    cls.__init__(op, *args, **kwargs)
    return op


def test_validation():
    src = _tensor(tl.float32, [256])
    mask = _tensor(tl.uint16, [16])
    count = _tensor(tl.int32, [8])
    assert _init(compare_scalar, src, 0.0, out=mask).arg_type["scalar"] == tl.float32
    _init(gather_mask, src, mask, out=[src, count])
    invalid = [
        (compare_scalar, [_tensor(tl.int32, [256]), 0.0], mask),
        (compare_scalar, [_tensor(tl.float32, [16, 16]), 0.0], mask),
        (compare_scalar, [_tensor(tl.float32, [128]), 0.0], _tensor(tl.uint16, [8])),
        (compare_scalar, [src, 0.0], _tensor(tl.uint32, [16])),
        (compare_scalar, [src, 0.0], _tensor(tl.uint16, [32])),
        (compare_scalar, [src, 0.0], None),
        (gather_mask, [src, mask], None),
        (gather_mask, [src, mask], [src]),
        (gather_mask, [src, _tensor(tl.uint32, [16])], [src, count]),
        (gather_mask, [src, mask], [_tensor(tl.float32, [512]), count]),
        (gather_mask, [src, mask], [src, _tensor(tl.int32, [1])]),
        (gather_mask, [src, mask], [src, _tensor(tl.float32, [8])]),
    ]
    for cls, args, out in invalid:
        try:
            _init(cls, *args, out=out)
        except AssertionError:
            continue
        raise AssertionError(f"{cls.__name__} accepted an invalid signature")
    print(f"[PASS] registration validation: {len(invalid)} rejected signatures")


def main():
    test_validation()
    test_compare_scalar()
    test_gather_mask()
    test_composed_routing()
    test_graph_replay()
    print("All mask custom op correctness tests passed.")


if __name__ == "__main__":
    main()
