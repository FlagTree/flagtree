import torch
import triton
import triton.language as tl
import triton.backends.mlu.driver as driver

import math
import random
import pytest
import json

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CLUSTER_NUM = _devprob.get("cluster_num")
TOTAL_CORE_NUM = TOTAL_CLUSTER_NUM * _devprob.get("core_num_per_cluster")
LLC_CACHE_SIZE = _devprob.get("max_l2_cache_size")

dld_configs = [
    triton.Config({"BLOCK_SIZE": bs, "SELECT_BLOCK_SIZE": 0, "C_BLOCK_SIZE": 0})
    for bs in [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768]
]

idx_check_configs = [
    triton.Config({"BLOCK_SIZE": bs}, num_stages=3, num_warps=1) for bs in [512, 1024, 2048, 4096, 8192, 16384, 32768]
]


@triton.autotune(configs=idx_check_configs, key=["idx_dim"])
@triton.jit(debug=True)
def idx_check_kernel(idx, idx_dim, select_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)

    total_num_blocks = tl.cdiv(idx_dim, BLOCK_SIZE)

    for block_id in range(pid, total_num_blocks, step):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < idx_dim
        cur_idx = tl.load(idx + offs, mask=mask, other=0)

        cur_idx_i32 = cur_idx.to(tl.int32)
        idx_lt_0 = cur_idx_i32 < 0
        idx_ge_size = cur_idx_i32 >= select_dim
        all_bool = idx_lt_0 | idx_ge_size

        tl.device_assert(tl.sum(all_bool) == 0, "[triton index_select]: invalid idx")


def dld_config_prune(configs, named_args, **kwargs):
    block_size_set = set()
    select_dim = named_args["select_dim"]
    c_dim = named_args["c_dim"]

    # TODO: add sweet-point config and do pruning based on it, i.e. only keep a few configs around it
    for config in configs:
        block_size = config.kwargs["BLOCK_SIZE"]
        # block is too large
        if block_size > select_dim * c_dim and select_dim * c_dim > 512:
            continue
        select_block_size = block_size // c_dim
        c_block_size = c_dim
        # block is too small compared with c_dim, so just discard it
        if select_block_size == 0 and block_size < 512:
            continue
        if select_block_size == 0:
            select_block_size = 1
            c_block_size = block_size
        block_size_set.add((select_block_size, c_block_size))

    pruned_configs = []
    for (select_block_size, c_block_size) in block_size_set:
        pruned_configs.append(
            triton.Config({"SELECT_BLOCK_SIZE": select_block_size, "C_BLOCK_SIZE": c_block_size}, num_stages=5,
                          num_warps=1))

    return pruned_configs


@triton.jit
def ld_dld(inp, block_id_batch, select_dim, c_dim, cur_idx, c_offs, ld_st_mask):
    inp_offs = block_id_batch * select_dim * c_dim + cur_idx[:, None] * c_dim + c_offs
    cur_inp = tl.load(inp + inp_offs, mask=ld_st_mask)

    return cur_inp


@triton.autotune(configs=dld_configs, key=["batch_dim", "select_dim", "c_dim", "idx_dim"],
                 prune_configs_by={"early_config_prune": dld_config_prune})
@triton.heuristics({
    "IS_INP_LARGE":
    lambda args: args["batch_dim"] * args["select_dim"] * args["c_dim"] * args["dtype_bytes"] >= 2**31,
    "LD_IDX_ONCE_FOR_ALL":
    lambda args: args["SELECT_BLOCK_SIZE"] >= args["idx_dim"],
})
@triton.jit
def idx_select_kernel_dld(inp, idx, out, batch_dim, select_dim, c_dim, idx_dim, select_dim_threshold, dtype_bytes,
                          SELECT_BLOCK_SIZE: tl.constexpr, C_BLOCK_SIZE: tl.constexpr, IS_INP_LARGE: tl.constexpr,
                          LD_IDX_ONCE_FOR_ALL: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)

    num_blocks_select = tl.cdiv(idx_dim, SELECT_BLOCK_SIZE)
    num_blocks_c = tl.cdiv(c_dim, C_BLOCK_SIZE)
    num_blocks_per_batch = num_blocks_select.to(tl.int64) * num_blocks_c.to(tl.int64)
    total_num_blocks = batch_dim * num_blocks_per_batch

    if LD_IDX_ONCE_FOR_ALL:
        select_offs = tl.arange(0, SELECT_BLOCK_SIZE)
        mask_select = select_offs < idx_dim
        cur_idx = tl.load(idx + select_offs, mask=mask_select, other=0)

    for block_id in range(pid, total_num_blocks, step):
        block_id_batch = block_id // num_blocks_per_batch
        block_id_select = (block_id - block_id_batch * num_blocks_per_batch) // num_blocks_c
        block_id_c = block_id % num_blocks_c

        if not LD_IDX_ONCE_FOR_ALL:
            select_offs = block_id_select * SELECT_BLOCK_SIZE + tl.arange(0, SELECT_BLOCK_SIZE)
            mask_select = select_offs < idx_dim
            cur_idx = tl.load(idx + select_offs, mask=mask_select, other=0)

        c_offs = block_id_c * C_BLOCK_SIZE + tl.arange(0, C_BLOCK_SIZE)
        mask_c = c_offs < c_dim

        ld_st_mask = mask_select[:, None] & mask_c[None, :]

        cur_idx_i32 = cur_idx.to(tl.int32)

        if not IS_INP_LARGE:
            cur_inp = ld_dld(inp, block_id_batch, select_dim, c_dim, cur_idx_i32, c_offs, ld_st_mask)
        else:
            cur_idx_max = tl.max(cur_idx_i32)
            if block_id_batch * select_dim + cur_idx_max < select_dim_threshold:
                cur_inp = ld_dld(inp, block_id_batch, select_dim, c_dim, cur_idx_i32, c_offs, ld_st_mask)
            else:
                cur_inp = ld_dld(inp, block_id_batch, select_dim, c_dim, cur_idx.to(tl.int64), c_offs, ld_st_mask)

        st_offs = block_id_batch * idx_dim * c_dim + select_offs[:, None] * c_dim + c_offs[None, :]
        tl.store(out + st_offs, cur_inp, mask=ld_st_mask)


vaa_configs = [
    triton.Config({
        "BLOCK_SIZE": bs,
        "BATCH_BLOCK_SIZE": 0,
        "SELECT_BLOCK_SIZE": 0,
    }) for bs in [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768]
]


def vaa_config_prune(configs, named_args, **kwargs):
    block_size_set = set()
    batch_dim = named_args["batch_dim"]
    idx_dim = named_args["idx_dim"]

    # TODO: add sweet-point config and do pruning based on it, i.e. only keep a few configs around it
    for config in configs:
        block_size = config.kwargs["BLOCK_SIZE"]
        # block is too large
        if block_size > batch_dim * idx_dim and batch_dim * idx_dim > 512:
            continue
        batch_block_size = block_size // idx_dim
        select_block_size = idx_dim
        # block is too small compared with idx_dim, so just discard it
        if batch_block_size == 0 and block_size < 512:
            continue
        if batch_block_size == 0:
            batch_block_size = 1
            select_block_size = block_size
        block_size_set.add((batch_block_size, select_block_size))

    pruned_configs = []
    for (batch_block_size, select_block_size) in block_size_set:
        pruned_configs.append(
            triton.Config({
                "BATCH_BLOCK_SIZE": batch_block_size,
                "SELECT_BLOCK_SIZE": select_block_size,
            }, num_stages=3, num_warps=1))

    return pruned_configs


def determin_ld_inp(idx_dim, select_dim, dtype_bytes, SELECT_BLOCK_SIZE):
    #TODO: find out the maximum size, currently set it to 64KB
    #TODO: currently tl.gather does not support mask, so it's only used for IDX_BLOCK_DIM1==idx_dim1
    #      lift this constraint once mask is supported
    return select_dim * dtype_bytes <= 64 * 1024 and \
           SELECT_BLOCK_SIZE == idx_dim and \
           idx_dim >= select_dim // (512 // dtype_bytes)


@triton.autotune(configs=vaa_configs, key=["batch_dim", "select_dim", "idx_dim"],
                 prune_configs_by={"early_config_prune": vaa_config_prune})
@triton.heuristics({
    "LD_USING_CACHE":
    lambda args:
    (args["select_dim"] * args["dtype_bytes"] <= LLC_CACHE_SIZE) and (args["idx_dim"] >= args["select_dim"] //
                                                                      (64 // args["dtype_bytes"])),
    "LD_IDX_ONCE_FOR_ALL":
    lambda args: args["SELECT_BLOCK_SIZE"] >= args["idx_dim"],
    "LD_INP":
    lambda args: determin_ld_inp(args["idx_dim"], args["select_dim"], args["dtype_bytes"], args["SELECT_BLOCK_SIZE"]),
    "SELECT_DIM":
    lambda args: args["select_dim"],
})
@triton.jit
def idx_select_kernel_vaa(
    inp,
    idx,
    out,
    batch_dim,
    select_dim,
    idx_dim,
    dtype_bytes,
    LD_USING_CACHE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    SELECT_BLOCK_SIZE: tl.constexpr,
    LD_INP: tl.constexpr,
    SELECT_DIM: tl.constexpr,
    LD_IDX_ONCE_FOR_ALL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)

    num_blocks_batch = tl.cdiv(batch_dim, BATCH_BLOCK_SIZE)
    num_blocks_select = tl.cdiv(idx_dim, SELECT_BLOCK_SIZE)
    total_num_blocks = num_blocks_batch.to(tl.int64) * num_blocks_select.to(tl.int64)

    if LD_IDX_ONCE_FOR_ALL:
        select_offs = tl.arange(0, SELECT_BLOCK_SIZE)
        mask_select = select_offs < idx_dim
        cur_idx = tl.load(idx + select_offs, mask=mask_select, other=0, cache_modifier=".ca")

    for block_id in range(pid, total_num_blocks, step):
        block_id_batch = block_id // num_blocks_select
        block_id_select = block_id % num_blocks_select

        batch_offs = block_id_batch * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
        mask_batch = batch_offs < batch_dim

        if not LD_IDX_ONCE_FOR_ALL:
            select_offs = block_id_select * SELECT_BLOCK_SIZE + tl.arange(0, SELECT_BLOCK_SIZE)
            mask_select = select_offs < idx_dim
            cur_idx = tl.load(idx + select_offs, mask=mask_select, other=0, cache_modifier=".ca")

        ld_st_mask = mask_batch[:, None] & mask_select[None, :]

        cur_idx_i32 = cur_idx.to(tl.int32)

        batch_offs_i32 = block_id_batch.to(tl.int32) * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
        if LD_INP:
            inp_offs = batch_offs_i32[:, None] * SELECT_DIM + tl.arange(0, SELECT_DIM)[None, :]
            cur_inp_whole = tl.load(inp + inp_offs, mask=mask_batch[:, None])
            cur_idx_i32_exp = tl.broadcast_to(cur_idx_i32[None, :], (BATCH_BLOCK_SIZE, SELECT_BLOCK_SIZE))

            cur_inp = tl.gather(cur_inp_whole, cur_idx_i32_exp, 1)
        else:
            inp_offs = batch_offs_i32[:, None] * select_dim + cur_idx_i32[None, :]
            if LD_USING_CACHE:
                cur_inp = tl.load(inp + inp_offs, mask=ld_st_mask, cache_modifier=".ca")
            else:
                cur_inp = tl.load(inp + inp_offs, mask=ld_st_mask)

        st_offs = batch_offs[:, None] * idx_dim + select_offs[None, :]
        tl.store(out + st_offs, cur_inp, mask=ld_st_mask)


def index_select(inp, dim, idx):
    assert dim >= -inp.ndim and dim < inp.ndim, "[triton index_select]: Invalid dim"
    assert idx.ndim <= 1, "Index should have dimension 1 or 0"

    dim = dim if dim >= 0 else (dim + inp.ndim)

    inp = inp.contiguous()
    inp_shape = inp.shape
    inp_numel = inp.numel()

    assert inp_shape[dim] < 2**31 - 1, f"inp.shape[{dim}] should be within int32 range, but now it's {inp_shape[dim]}"

    if idx.ndim == 0:
        idx = idx.unsqueeze(0)
    idx_dim = idx.numel()

    out_shape = list(inp_shape)
    out_shape[dim] = idx_dim
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # input  [batch_dim, select_dim, c_dim]
    # output [batch_dim, idx_dim, c_dim]
    batch_dim = math.prod(inp_shape[:dim])
    select_dim = inp_shape[dim]
    c_dim = math.prod(inp_shape[(dim + 1):])

    grid = lambda meta: (TOTAL_CORE_NUM, )

    # dld_kernel: launch a separte kernel to check idx so that larger block can be used and higher compute efficiency
    # vaa_kernel: remove redundant idx check
    idx_check_kernel[grid](idx, idx_dim, select_dim)

    if torch.is_floating_point(inp):
        dtype_bytes = torch.finfo(inp.dtype).bits // 8
    else:
        dtype_bytes = torch.iinfo(inp.dtype).bits // 8

    if c_dim > 1:
        select_dim_threshold = ((2**31 - 1) // dtype_bytes) // c_dim
        idx_select_kernel_dld[grid](inp, idx, out, batch_dim, select_dim, c_dim, idx_dim, select_dim_threshold,
                                    dtype_bytes)
    else:
        # c_dim is 1
        # input  [batch_dim, select_dim]
        # output [batch_dim, idx_dim]
        # essentially gather at dim=1
        batch_dim_threshold = ((2**31 - 1) // dtype_bytes) // select_dim
        assert batch_dim_threshold > 0, f"the last dim of inp is too large {inp_shape[-1]}, not supported yet"
        step = batch_dim_threshold

        inp = inp.reshape((batch_dim, select_dim))
        out_ = out.reshape((batch_dim, idx_dim))

        for batch_dim_off in range(0, batch_dim, step):
            bs = step if (batch_dim_off + step <= batch_dim) else (batch_dim - batch_dim_off)
            idx_select_kernel_vaa[grid](inp[batch_dim_off:(batch_dim_off + bs), :], idx,
                                        out_[batch_dim_off:(batch_dim_off + bs), :], bs, select_dim, idx_dim,
                                        dtype_bytes)

    return out


INP_SHAPES = [
    (64, 64),
    #(256, 256),
    #(1024, 1024),
    #(4096, 4096),
    #(1024, 65536),
    #(1229312, 512),
    #(1433600, 512),
]

dtypes = [torch.float32, torch.float16]


def gen_inp_idx_tensor(inp_shape, idx_shape, dtype, dim):
    inp = torch.randn(inp_shape, dtype=dtype, device="mlu")
    idx = torch.randint(0, inp_shape[dim], idx_shape, dtype=torch.long, device=inp.device)

    return (inp, idx)


def gen_inp_idx_dims(inp_shapes, dims):
    inp_idx_dims = []

    for inp_shape in inp_shapes:
        for dim in dims:
            inp_dim_size = inp_shape[dim]

            idx_size = random.randint(inp_dim_size // 2, inp_dim_size)
            idx_size = idx_size if idx_size > 0 else 1
            inp_idx_dims.append((inp_shape, (idx_size, ), dim))

            idx_size = random.randint(inp_dim_size, inp_dim_size * 2)
            inp_idx_dims.append((inp_shape, (idx_size, ), dim))

    return inp_idx_dims


@pytest.mark.parametrize("inp_idx_shape_dim", gen_inp_idx_dims(INP_SHAPES, [0, 1]))
@pytest.mark.parametrize("dtype", dtypes)
def test_op(inp_idx_shape_dim, dtype):
    print(inp_idx_shape_dim, dtype)
    inp_shape, idx_shape, dim = inp_idx_shape_dim
    inp, idx = gen_inp_idx_tensor(inp_shape, idx_shape, dtype, dim)

    out = index_select(inp, dim, idx)
    out_ref = torch.index_select(inp, dim, idx)

    torch.testing.assert_close(out_ref, out, atol=0, rtol=0)


def gen_perf_configs(inp_shapes, dims):
    inp_idx_dims_str = {dim: [] for dim in dims}
    inp_idx_dims = gen_inp_idx_dims(inp_shapes, dims)

    for inp_idx_dim in inp_idx_dims:
        inp_shape, idx_shape, dim = inp_idx_dim
        inp_shape = list(inp_shape)
        idx_shape = list(idx_shape)
        # To ensure the indices generated for triton and torch are the same
        rand_seed = random.randint(0, 2**32 - 1)

        # testing.py: df[first_x].to_numpy(): requires string or float, not list
        inp_idx_dims_str[dim].append((str(inp_shape), str(idx_shape), rand_seed))

    perf_configs = []
    for dim in dims:
        for dtype in dtypes:
            perf_configs.append(
                triton.testing.Benchmark(
                    x_names=["inp_shape", "idx_shape", "rand_seed"],
                    x_vals=inp_idx_dims_str[dim],
                    line_arg="provider",
                    line_vals=["Triton", "Torch"],
                    line_names=["Triton (ms)", "Torch (ms)"],
                    styles=[("yellow", "-"), ("red", "-")],
                    ylabel="Time",
                    plot_name=f"index_select-performance-dtype_{dtype}-dim_{dim}",
                    args={
                        "dtype": dtype,
                        "dim": dim,
                    },
                ))
    return perf_configs


def benchmark_base(inp_shape, idx_shape, rand_seed, dtype, dim, provider):
    print(provider, inp_shape, idx_shape, rand_seed, dtype, dim)
    inp_shape = json.loads(inp_shape)
    idx_shape = json.loads(idx_shape)

    torch.mlu.manual_seed(rand_seed)
    inp, idx = gen_inp_idx_tensor(inp_shape, idx_shape, dtype, dim)

    if provider == "Triton":
        ms_ = triton.testing.do_bench(lambda: index_select(inp, dim, idx))
        ms = triton.testing.do_bench(lambda: index_select(inp, dim, idx), warmup=10 * ms_, rep=102 * ms_)
    if provider == "Torch":
        ms_ = triton.testing.do_bench(lambda: torch.index_select(inp, dim, idx))
        # Torch can run for a very long time and get stuck somehow, we can not afford it
        if ms_ < 500:
            ms = triton.testing.do_bench(lambda: torch.index_select(inp, dim, idx), warmup=10 * ms_, rep=102 * ms_)
        else:
            ms = ms_
    return ms


@triton.testing.perf_report(gen_perf_configs(INP_SHAPES, [0]))
def benchmark_dim0(inp_shape, idx_shape, rand_seed, dtype, dim, provider):
    return benchmark_base(inp_shape, idx_shape, rand_seed, dtype, dim, provider)


@triton.testing.perf_report(gen_perf_configs(INP_SHAPES, [1]))
def benchmark_dim1(inp_shape, idx_shape, rand_seed, dtype, dim, provider):
    return benchmark_base(inp_shape, idx_shape, rand_seed, dtype, dim, provider)


if __name__ == "__main__":
    benchmark_dim0.run(show_plots=True, print_data=True)
    benchmark_dim1.run(show_plots=True, print_data=True)
