# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29501 ag-gemm.py
"""Allgather A on the copy engine, GEMM in Triton.

Local rows are marked ready with a raw store. Remote rows are pulled with
dushmem_ptr on a side stream, which overlaps the GEMM of rows that are already
ready. The Triton GEMM calls raw to wait before it reads a rank's rows.
"""

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.hcu.utils import (
    compile_host_library,
    init_dushmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    tensor_from_pointer,
)

HERE = Path(__file__).parent
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
M = 2048
N = 2048
K = 1024


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=HERE / "ag-gemm-device.hip", extern_func_name="ag_mark_local_ready", deferred=True)
def mark_local_ready(*args, **kwargs):
    ...


@dialect(name="hcu", file=HERE / "ag-gemm-device.hip", extern_func_name="ag_wait_ready", deferred=True)
def wait_ready(*args, **kwargs):
    ...


def _load_host():
    host = compile_host_library(HERE / "ag-gemm-host.hip", Path("/tmp/dushmem-ag-gemm-host.so"))
    host.ag_gemm_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    host.ag_gemm_workspace_create.restype = ctypes.c_int
    host.ag_gemm_peer_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_peer_ptr.restype = ctypes.c_void_p
    host.ag_gemm_copy_on_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    host.ag_gemm_copy_on_stream.restype = ctypes.c_int
    host.ag_gemm_signal_on_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    host.ag_gemm_signal_on_stream.restype = ctypes.c_int
    host.ag_gemm_barrier_on_stream.argtypes = [ctypes.c_void_p]
    host.ag_gemm_barrier_on_stream.restype = None
    host.ag_gemm_workspace_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    host.ag_gemm_workspace_destroy.restype = ctypes.c_int
    return host


@triton.jit
def set_local_ready(ready, rank):
    tle_raw.call(mark_local_ready, [ready, rank], output_indices=[])


@triton.jit
def consumer_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    ready,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    m_per_rank,
    RANK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    pid_ms_per_rank = m_per_rank // BLOCK_SIZE_M
    dtype = c_ptr.dtype.element_ty

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id - pid_m * num_pid_n
        pid_m = (pid_m + RANK * pid_ms_per_rank) % num_pid_m
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        source_rank = (pid_m * BLOCK_SIZE_M) // m_per_rank
        tle_raw.call(wait_ready, [ready, source_rank.to(tl.int32)], output_indices=[])

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in tl.range(k_tiles, flatten=False):
            accumulator = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, accumulator.to(dtype))


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise SystemExit("ag-gemm needs at least 2 ranks")
    if M % world_size or N % world_size:
        raise SystemExit("M and N must be divisible by the world size")
    m_per_rank = M // world_size
    n_per_rank = N // world_size
    if m_per_rank % BLOCK_M or n_per_rank % BLOCK_N or K % BLOCK_K:
        raise SystemExit("shards must be divisible by the tile size")

    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    host = _load_host()
    device = torch.device("cuda", local_rank)
    dtype = torch.float16

    workspace_ptr = ctypes.c_void_p()
    ready_ptr = ctypes.c_void_p()
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    status = host.ag_gemm_workspace_create(
        m_per_rank * K,
        dtype.itemsize,
        world_size,
        ctypes.byref(workspace_ptr),
        ctypes.byref(ready_ptr),
        ctypes.byref(mype),
        ctypes.byref(npes),
    )
    if status != 0:
        raise SystemExit(f"workspace create failed: {status}")
    if mype.value != rank or npes.value != world_size:
        raise SystemExit(f"rank mismatch torch={rank} dushmem={mype.value}")

    workspace = tensor_from_pointer(workspace_ptr, (M, K), dtype, device)
    ready = tensor_from_pointer(ready_ptr, (world_size, ), torch.uint64, device)

    torch.manual_seed(2000 + rank)
    a_local = torch.randn((m_per_rank, K), dtype=dtype, device=device) * 0.1
    b = torch.randn((K, n_per_rank), dtype=dtype, device=device) * 0.1
    c = torch.empty((M, n_per_rank), dtype=dtype, device=device)

    ready.zero_()
    workspace[rank * m_per_rank:(rank + 1) * m_per_rank].copy_(a_local)
    set_local_ready[(1, )](ready, rank, num_warps=1)

    current = torch.cuda.current_stream(device)
    host.ag_gemm_barrier_on_stream(current.cuda_stream)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    comm_stream.wait_stream(current)
    elements = m_per_rank * K
    nbytes = elements * dtype.itemsize
    for step in range(1, world_size):
        src_rank = (rank + step) % world_size
        peer_base = host.ag_gemm_peer_ptr(workspace_ptr, src_rank)
        if not peer_base:
            raise SystemExit(f"dushmem_ptr failed for peer {src_rank}")
        segment = src_rank * elements * dtype.itemsize
        src_ptr = peer_base + segment
        dst_ptr = workspace.data_ptr() + segment
        copy_rc = host.ag_gemm_copy_on_stream(comm_stream.cuda_stream, dst_ptr, src_ptr, nbytes)
        if copy_rc != 0:
            raise SystemExit(f"copy failed: {copy_rc}")
        signal_ptr = ready.data_ptr() + src_rank * ctypes.sizeof(ctypes.c_uint64)
        signal_rc = host.ag_gemm_signal_on_stream(comm_stream.cuda_stream, signal_ptr, 1)
        if signal_rc != 0:
            raise SystemExit(f"signal failed: {signal_rc}")

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(n_per_rank, BLOCK_N)
    num_sms = min(num_tiles, sms)
    consumer_gemm[(num_sms, )](
        workspace,
        b,
        c,
        ready,
        M,
        n_per_rank,
        K,
        workspace.stride(0),
        workspace.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        m_per_rank,
        RANK=rank,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=4,
    )
    current.wait_stream(comm_stream)
    torch.cuda.synchronize()

    shards = [torch.empty(m_per_rank, K, dtype=dtype) for _ in range(world_size)]
    torch.distributed.all_gather(shards, a_local.cpu(), group=group)
    full_a = torch.cat(shards, dim=0).to(device)
    expected = full_a @ b
    err = (c - expected).abs().max().item()
    ok = torch.allclose(c, expected, rtol=2e-2, atol=2e-2)
    print(f"ag-gemm pe={rank} allclose={bool(ok)} max_abs_err={err}", flush=True)

    destroy = host.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
    if destroy != 0:
        raise SystemExit(f"workspace destroy failed: {destroy}")
    torch.distributed.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
