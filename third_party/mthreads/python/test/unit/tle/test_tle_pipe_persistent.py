"""Runtime regressions for persistent MThreads TLE pipe reuse."""

import re

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import compile_musa, require_mthreads_libtriton


require_mthreads_libtriton()


@triton.jit
def _persistent_single_writer(
    writer,
    a_desc,
    b_desc,
    outer_tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 256
    block_k: tl.constexpr = 64
    for tile in range(outer_tiles):
        for k_iter in range(k_tiles):
            token = tile * k_tiles + k_iter
            slot = writer.acquire(token)
            tle.gpu.copy(
                a_desc,
                slot.a,
                (block_m, block_k),
                (tile * block_m, k_iter * block_k),
            )
            tle.gpu.copy(
                b_desc,
                slot.b,
                (block_k, block_n),
                (k_iter * block_k, 0),
            )
            writer.commit(token)


@triton.jit
def _persistent_single_reader(
    reader,
    out,
    outer_tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 256
    half_n: tl.constexpr = 128
    for tile in range(outer_tiles):
        acc_lo = tl.zeros((block_m, half_n), tl.float32)
        acc_hi = tl.zeros((block_m, half_n), tl.float32)
        for k_iter in range(k_tiles):
            token = tile * k_tiles + k_iter
            ready = reader.wait(token)
            b_lo = ready.slot.b.slice(0, half_n, dim=1)
            b_hi = ready.slot.b.slice(half_n, half_n, dim=1)
            acc_lo = tle.gpu.wgmma(ready.slot.a, b_lo, acc_lo)
            acc_hi = tle.gpu.wgmma(ready.slot.a, b_hi, acc_hi)
            acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
            acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
            reader.release(token)

        rows = tile * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, half_n)
        tl.store(
            out + rows[:, None] * block_n + cols[None, :],
            acc_lo.to(out.dtype.element_ty),
        )
        tl.store(
            out + rows[:, None] * block_n + half_n + cols[None, :],
            acc_hi.to(out.dtype.element_ty),
        )


@triton.jit
def _persistent_single_reader_kernel(
    a_desc,
    b_desc,
    out,
    OUTER_TILES: tl.constexpr,
    K_TILES: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 256
    block_k: tl.constexpr = 64
    stages: tl.constexpr = 3
    a = tle.gpu.alloc(
        (stages, block_m, block_k),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (stages, block_k, block_n),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="persistent_single_reader",
        a=a,
        b=b,
    )
    tle.gpu.warp_specialize(
        [
            (
                _persistent_single_reader,
                (pipe.reader(), out, OUTER_TILES, K_TILES),
            ),
            (
                _persistent_single_writer,
                (pipe.writer(), a_desc, b_desc, OUTER_TILES, K_TILES),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[192],
    )


@triton.jit
def _persistent_multi_writer(
    writer,
    a_desc,
    b_main_desc,
    b_tail_desc,
    outer_tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 320
    block_k: tl.constexpr = 64
    main_n: tl.constexpr = 256
    tail_n: tl.constexpr = 64
    for tile in range(outer_tiles):
        for k_iter in range(k_tiles):
            token = tile * k_tiles + k_iter
            slot = writer.acquire(token)
            tle.gpu.copy(
                a_desc,
                slot.a,
                (block_m, block_k),
                (tile * block_m, k_iter * block_k),
            )
            tle.gpu.copy(
                b_main_desc,
                slot.b_main,
                (block_k, main_n),
                (k_iter * block_k, 0),
            )
            tle.gpu.copy(
                b_tail_desc,
                slot.b_tail,
                (block_k, tail_n),
                (k_iter * block_k, main_n),
            )
            writer.commit(token)


@triton.jit
def _persistent_main_reader(
    reader,
    out,
    outer_tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 320
    half_n: tl.constexpr = 128
    for tile in range(outer_tiles):
        acc_lo = tl.zeros((block_m, half_n), tl.float32)
        acc_hi = tl.zeros((block_m, half_n), tl.float32)
        for k_iter in range(k_tiles):
            token = tile * k_tiles + k_iter
            ready = reader.wait(token)
            b_lo = ready.slot.b_main.slice(0, half_n, dim=1)
            b_hi = ready.slot.b_main.slice(half_n, half_n, dim=1)
            acc_lo = tle.gpu.wgmma(ready.slot.a, b_lo, acc_lo)
            acc_hi = tle.gpu.wgmma(ready.slot.a, b_hi, acc_hi)
            acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
            acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
            reader.release(token)

        rows = tile * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, half_n)
        tl.store(
            out + rows[:, None] * block_n + cols[None, :],
            acc_lo.to(out.dtype.element_ty),
        )
        tl.store(
            out + rows[:, None] * block_n + half_n + cols[None, :],
            acc_hi.to(out.dtype.element_ty),
        )


@triton.jit
def _persistent_tail_reader(
    reader,
    out,
    outer_tiles: tl.constexpr,
    k_tiles: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 320
    main_n: tl.constexpr = 256
    tail_n: tl.constexpr = 64
    for tile in range(outer_tiles):
        acc = tl.zeros((block_m, tail_n), tl.float32)
        for k_iter in range(k_tiles):
            token = tile * k_tiles + k_iter
            ready = reader.wait(token)
            acc = tle.gpu.wgmma(ready.slot.a, ready.slot.b_tail, acc)
            acc = tle.gpu.wgmma_wait(0, acc)
            reader.release(token)

        rows = tile * block_m + tl.arange(0, block_m)
        cols = main_n + tl.arange(0, tail_n)
        tl.store(
            out + rows[:, None] * block_n + cols[None, :],
            acc.to(out.dtype.element_ty),
        )


@triton.jit
def _persistent_multi_reader_kernel(
    a_desc,
    b_main_desc,
    b_tail_desc,
    out,
    OUTER_TILES: tl.constexpr,
    K_TILES: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_k: tl.constexpr = 64
    main_n: tl.constexpr = 256
    tail_n: tl.constexpr = 64
    stages: tl.constexpr = 2
    a = tle.gpu.alloc(
        (stages, block_m, block_k),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_main = tle.gpu.alloc(
        (stages, block_k, main_n),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_tail = tle.gpu.alloc(
        (stages, block_k, tail_n),
        dtype=tl.bfloat16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="persistent_multi_reader",
        readers=("tail", "main"),
        a=a,
        b_main=b_main,
        b_tail=b_tail,
    )
    tle.gpu.warp_specialize(
        [
            (
                _persistent_tail_reader,
                (
                    pipe.reader("tail"),
                    out,
                    OUTER_TILES,
                    K_TILES,
                ),
            ),
            (
                _persistent_main_reader,
                (
                    pipe.reader("main"),
                    out,
                    OUTER_TILES,
                    K_TILES,
                ),
            ),
            (
                _persistent_multi_writer,
                (
                    pipe.writer(),
                    a_desc,
                    b_main_desc,
                    b_tail_desc,
                    OUTER_TILES,
                    K_TILES,
                ),
            ),
        ],
        worker_num_warps=[16, 4],
        worker_num_regs=[128, 24],
    )


def _make_inputs(m, n, k):
    torch.manual_seed(20260925)
    a_cpu = torch.randn((m, k), dtype=torch.bfloat16)
    b_cpu = torch.randn((k, n), dtype=torch.bfloat16)
    return a_cpu, b_cpu, a_cpu.to("musa"), b_cpu.to("musa")


def test_tme_only_pipe_does_not_get_local_store_wait_groups():
    compiled = compile_musa(
        _persistent_single_reader_kernel,
        {
            # Tensor descriptor signature dimensions denote block shapes, not
            # the complete backing tensor extents.
            "a_desc": "tensordesc<bf16[256, 64]>",
            "b_desc": "tensordesc<bf16[64, 256]>",
            "out": "*bf16",
            "OUTER_TILES": "constexpr",
            "K_TILES": "constexpr",
        },
        {"OUTER_TILES": 18, "K_TILES": 9},
        options={"num_warps": 16, "num_stages": 1},
    )
    ttgir = compiled.asm["ttgir"]
    assert "musa_tle.pipe_local_store_group" not in ttgir


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_persistent_single_reader_pipe_reuses_ring_across_output_tiles():
    # Match the production addmm schedule's 18 persistent output tiles and
    # nine K publications per tile.  Short two-tile probes do not reliably
    # expose a duplicate CTA-wide barrier initialization.
    m, n, k = 4608, 256, 576
    outer_tiles, k_tiles = m // 256, k // 64
    a_cpu, b_cpu, a, b = _make_inputs(m, n, k)
    out = torch.empty((m, n), device="musa", dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_desc = TensorDescriptor.from_tensor(b, [64, 256])
    expected = a_cpu.float() @ b_cpu.float()

    for _ in range(3):
        out.fill_(float("nan"))
        compiled = _persistent_single_reader_kernel[(1,)](
            a_desc,
            b_desc,
            out,
            OUTER_TILES=outer_tiles,
            K_TILES=k_tiles,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=3e-1, rtol=1e-1)

    llir = compiled.asm["llir"]
    assert "llvm.musa.async.init.arrival" in llir
    assert "llvm.musa.async.arrive.none.phaseid" in llir
    # Three full and three empty pipe barriers are sufficient.  A local
    # pointer dependency in the producer must not be lowered into a seventh
    # per-K private barrier; the pipe acquire already supplies that edge.
    assert "llvm.musa.async.bar.record(i32 6)" in llir
    assert "call void @llvm.musa.lma.wait()" in llir
    init_calls = re.findall(
        r"(?:tail )?call void @llvm\.musa\.async\.init\.arrival\(", llir
    )
    assert len(init_calls) == 6


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_persistent_named_readers_release_every_epoch_before_ring_reuse():
    # Exercise enough outer epochs to wrap the alternating barrier phase more
    # than once.  Keeping a nested K loop also verifies that late lowering
    # derives the phase from the outer tile loop rather than the K loop.
    m, n, k = 1792, 320, 256
    outer_tiles, k_tiles = m // 256, k // 64
    a_cpu, b_cpu, a, b = _make_inputs(m, n, k)
    out = torch.empty((m, n), device="musa", dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_main_desc = TensorDescriptor.from_tensor(b, [64, 256])
    b_tail_desc = TensorDescriptor.from_tensor(b, [64, 64])
    expected = a_cpu.float() @ b_cpu.float()

    for _ in range(10):
        out.fill_(float("nan"))
        compiled = _persistent_multi_reader_kernel[(1,)](
            a_desc,
            b_main_desc,
            b_tail_desc,
            out,
            OUTER_TILES=outer_tiles,
            K_TILES=k_tiles,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=3e-1, rtol=1e-1)

    llir = compiled.asm["llir"]
    assert llir.count("llvm.musa.async.init.arrival") >= 4
    assert "llvm.musa.async.arrive.none.phaseid" in llir
    # The two-stage named-reader pipe already owns two arrival-count-20 empty
    # barriers.  The compiler-inferred consumer epoch rendezvous must add a
    # third one, shared by tail(4) + main(16) warps.
    epoch_barrier_ids = re.findall(
        r"call void @llvm\.musa\.async\.init\.arrival\(i32 (\d+), i32 20, i32 0\)",
        llir,
    )
    assert len(epoch_barrier_ids) == 3
    epoch_barrier_id = epoch_barrier_ids[-1]
    assert llir.count(
        f"call void @llvm.musa.async.arrive.none.phaseid(i32 {epoch_barrier_id})"
    ) == 2
    assert llir.count(
        f"call void @llvm.musa.async.wait(i32 {epoch_barrier_id},"
    ) == 2
