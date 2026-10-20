# Copyright 2026-     FlagOS Contributors
"""03-gemm-allreduce: fused GEMM+AllReduce (raw cannshmem).

References found and followed:
  1. Catcoc ``MatmulAllReduce``
     ``FlagTree-AscendNPU-IR-shmem/.../catcoc/dgemm/kernel/matmul_allreduce.h``
     - WORKSPACE_STAGES=2, AIC store → CrossCoreFlag → AIV
     - AIV: ``aclshmemx_barrier_all_vec`` then reduce (here: oneshot get+sum)
  2. DSA ``python/tutorials/tle/dsa/04-tle-cv-mix-pipeline.py``
     - Cube∥Vector double-buffer via ``sync_block_*``
     - Mix + cannshmem ``tle_raw.call`` currently hangs on this stack → not default
  3. NV ``python/tutorials/tle/raw/nvshmem/03-gemm-allreduce``
     - Producer GEMM tiles ∥ consumer AR (ready flags / streams)

Default ``GEMM_AR_MODE=soft``: Catcoc staging (STAGES=2) + NV soft streams.
``GEMM_AR_MODE=mix``: DSA-04 Mix + cannshmem AR (currently incorrect numerically).

  ASH_MASTER_PORT=18790 torchrun --nproc_per_node=2 --master_port=29790 \\
      python/tutorials/tle/raw/cannshmem/03-gemm-allreduce/gemm_ar.py
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cannshmem.utils import (
    bench_us,
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

DEVICE = "npu"
# mix = DSA-04/Catcoc Mix kernel (runs but wrong vs ref on this stack)
# soft = NV host dual-stream + Catcoc stages (correct default)
FUSED_MODE = os.environ.get("GEMM_AR_MODE", "soft").lower()
M, N, K = 1024, 1024, 256
BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
STAGES = 2  # Catcoc WORKSPACE_STAGES
TILE_ELEMS = BLOCK_M * N
NUM_TILES = M // BLOCK_M
pipe = tle.dsa.ascend.PIPE


@dialect(
    name="cann",
    library="cannshmem",
    file=Path(__file__).parent / "ar_oneshot.cpp",
    extern_func_name="cannshmem_ar_oneshot",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def ar_oneshot(*args, **kwargs):
    ...


@triton.jit
def mix_fused_gemm_ar_kernel(
    a_ptr,
    b_ptr,
    symm_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    TILE_ELEMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """DSA-04 Mix double-buffer + Catcoc AIV AR (sync_block = CrossCoreFlag)."""
    num_blocks = M // BLOCK_M
    tle.dsa.ascend.sync_block_set("vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set("vector", "cube", 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

    for block_idx in range(num_blocks):
        stage = block_idx % 2
        symm_off = stage * TILE_ELEMS
        start_m = block_idx * BLOCK_M
        out_off = start_m * N
        offs_am = start_m + tl.arange(0, BLOCK_M)
        local_m = tl.arange(0, BLOCK_M)

        tle.dsa.ascend.sync_block_wait("vector", "cube", stage, pipe.PIPE_MTE2, pipe.PIPE_FIX)
        for pid_n in range(0, tl.cdiv(N, BLOCK_N)):
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a = tl.load(
                    a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c = acc.to(tl.float16)
            tl.store(
                symm_ptr + symm_off + local_m[:, None] * N + offs_bn[None, :],
                c,
                mask=(offs_bn[None, :] < N),
            )
        tle.dsa.ascend.sync_block_set("cube", "vector", stage, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        tle.dsa.ascend.sync_block_wait("cube", "vector", stage, pipe.PIPE_FIX, pipe.PIPE_MTE2)
        dummy = tl.zeros([16], dtype=tl.float16)
        _ = tle_raw.call(
            ar_oneshot,
            [symm_ptr, out_ptr, symm_off, out_off, TILE_ELEMS, dummy],
            output_indices=[5],
        )
        tle.dsa.ascend.sync_block_set("vector", "cube", stage, pipe.PIPE_MTE2, pipe.PIPE_FIX)


@triton.jit
def gemm_tile_kernel(
    a_ptr,
    b_ptr,
    symm_ptr,
    start_m,
    symm_off,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Producer tile (NV gemm_ar_producer / Catcoc AIC) → symmetric stage."""
    offs_am = start_m + tl.arange(0, BLOCK_M)
    local_m = tl.arange(0, BLOCK_M)
    for pid_n in range(0, tl.cdiv(N, BLOCK_N)):
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
        c = acc.to(tl.float16)
        tl.store(
            symm_ptr + symm_off + local_m[:, None] * N + offs_bn[None, :],
            c,
            mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N),
        )


@triton.jit
def ar_tile_kernel(symm, out, symm_off, out_off, TILE_ELEMS: tl.constexpr):
    """Consumer tile (NV consumer / Catcoc AIV); barrier inside ar_oneshot."""
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(
        ar_oneshot,
        [symm, out, symm_off, out_off, TILE_ELEMS, dummy],
        output_indices=[5],
    )


@triton.jit
def sequential_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    num_blocks = M // BLOCK_M
    for block_idx in range(num_blocks):
        start_m = block_idx * BLOCK_M
        offs_am = start_m + tl.arange(0, BLOCK_M)
        for pid_n in range(0, tl.cdiv(N, BLOCK_N)):
            offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a = tl.load(
                    a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            c = acc.to(tl.float16)
            tl.store(
                c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
                c,
                mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N),
            )


@triton.jit
def sequential_ar_kernel(symm, out, N_ELEMS: tl.constexpr):
    dummy = tl.zeros([16], dtype=tl.float16)
    _ = tle_raw.call(ar_oneshot, [symm, out, 0, 0, N_ELEMS, dummy], output_indices=[5])


def _kw():
    return dict(enable_legacy_insert_load_store_for_mix_cv=True)


def _run_fused_mix(a, b, symm, out):
    mix_fused_gemm_ar_kernel[(1, 1, 1)](
        a, b, symm, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        TILE_ELEMS=TILE_ELEMS, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        **_kw(),
    )


def _run_fused_soft(a, b, symm, out):
    """Catcoc STAGES=2 + NV soft streams. Host barriers align multi-rank barrier counts."""
    dist.barrier()
    s_gemm = torch.npu.Stream()
    s_ar = torch.npu.Stream()
    gemm_done = [torch.npu.Event() for _ in range(STAGES)]
    ar_done = [torch.npu.Event() for _ in range(STAGES)]
    for s in range(STAGES):
        ar_done[s].record(s_ar)

    for block_idx in range(NUM_TILES):
        stage = block_idx % STAGES
        symm_off = stage * TILE_ELEMS
        start_m = block_idx * BLOCK_M
        out_off = start_m * N

        with torch.npu.stream(s_gemm):
            s_gemm.wait_event(ar_done[stage])
            gemm_tile_kernel[(1, )](
                a, b, symm, start_m, symm_off, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                **_kw(),
            )
            gemm_done[stage].record(s_gemm)

        with torch.npu.stream(s_ar):
            s_ar.wait_event(gemm_done[stage])
            ar_tile_kernel[(1, )](
                symm, out, symm_off, out_off, TILE_ELEMS=TILE_ELEMS, **_kw(),
            )
            ar_done[stage].record(s_ar)

    torch.npu.current_stream().wait_stream(s_gemm)
    torch.npu.current_stream().wait_stream(s_ar)
    torch.npu.synchronize()
    dist.barrier()


def _run_fused(a, b, symm, out):
    if FUSED_MODE == "soft":
        _run_fused_soft(a, b, symm, out)
    else:
        _run_fused_mix(a, b, symm, out)


def _run_sequential_raw(a, b, c_local, symm, out):
    sequential_gemm_kernel[(1, )](
        a, b, c_local, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c_local.stride(0), c_local.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, **_kw(),
    )
    symm.copy_(c_local.reshape(-1))
    sequential_ar_kernel[(1, )](symm, out, N_ELEMS=M * N, **_kw())


def main():
    init_cannshmem_by_torch_pg(ash_size=256 * 1024 * 1024)
    rank = dist.get_rank()
    world = dist.get_world_size()
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    g = torch.Generator(device="cpu")
    g.manual_seed(0xC0FFEE + rank)
    a = torch.randn(M, K, dtype=torch.float16, generator=g).to(DEVICE).contiguous()
    b = torch.randn(K, N, dtype=torch.float16, generator=g).to(DEVICE).contiguous()

    c_local = torch.empty(M, N, dtype=torch.float16, device=DEVICE)
    symm = create_symmetric_tensor(
        torch.empty(STAGES * TILE_ELEMS, dtype=torch.float16, device=DEVICE))
    symm_full = create_symmetric_tensor(torch.empty(M * N, dtype=torch.float16, device=DEVICE))
    out_fused = torch.empty(M * N, dtype=torch.float16, device=DEVICE)
    out_seq = torch.empty(M * N, dtype=torch.float16, device=DEVICE)

    ref = torch.mm(a.float(), b.float()).to(torch.float16)
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()

    dist.barrier()
    _run_fused(a, b, symm, out_fused)
    fused_out = out_fused.view(M, N)
    ok_fused = torch.allclose(fused_out.float(), ref.float(), rtol=3e-2, atol=3e-2)
    print(f"[rank {rank}] fused vs ref {'PASS' if ok_fused else 'FAIL'} "
          f"max_abs={(fused_out.float() - ref.float()).abs().max().item():.4f}")

    dist.barrier()
    _run_sequential_raw(a, b, c_local, symm_full, out_seq)
    torch.npu.synchronize()
    dist.barrier()
    seq_out = out_seq.view(M, N)
    ok_seq = torch.allclose(seq_out.float(), ref.float(), rtol=3e-2, atol=3e-2)
    print(f"[rank {rank}] sequential vs ref {'PASS' if ok_seq else 'FAIL'} "
          f"max_abs={(seq_out.float() - ref.float()).abs().max().item():.4f}")
    if not (ok_fused and ok_seq):
        cleanup_symmetric_tensor(symm)
        cleanup_symmetric_tensor(symm_full)
        raise SystemExit(1)

    try:
        _ = torch_npu.npu_mm_all_reduce_base(a, b, hcom, reduce_op="sum")
        torch.npu.synchronize()
        has_npu = True
    except Exception as exc:  # noqa: BLE001
        has_npu = False
        if rank == 0:
            print(f"torch_npu fused unavailable: {type(exc).__name__}")

    us_fused = bench_us(lambda: _run_fused(a, b, symm, out_fused), warmup=2, iters=5)
    us_seq = bench_us(lambda: _run_sequential_raw(a, b, c_local, symm_full, out_seq), warmup=2, iters=5)
    us_mm = bench_us(
        lambda: dist.all_reduce(torch.mm(a, b), op=dist.ReduceOp.SUM), warmup=2, iters=5)
    us_npu = None
    if has_npu:
        us_npu = bench_us(
            lambda: torch_npu.npu_mm_all_reduce_base(a, b, hcom, reduce_op="sum"),
            warmup=2, iters=5)

    if rank == 0:
        print(f"shape=({M},{K})x({K},{N}) world={world} stages={STAGES} tiles={NUM_TILES} mode={FUSED_MODE}")
        print(f"fused   {FUSED_MODE:8s} gemm-ar  {us_fused:8.1f} us")
        print(f"seq     gemm then AR       {us_seq:8.1f} us")
        print(f"torch   mm + all_reduce    {us_mm:8.1f} us")
        print(f"fused/seq speedup          {us_seq / us_fused:.2f}x  (>1 = fusion helps)")
        if us_npu is not None:
            print(f"torch_npu mm_all_reduce    {us_npu:8.1f} us")
        print("03-gemm-allreduce PASS")

    cleanup_symmetric_tensor(symm)
    cleanup_symmetric_tensor(symm_full)


if __name__ == "__main__":
    main()
