import torch
import triton
import triton.language as tl
import triton.backends.mlu.driver as driver

import random
import pytest
import json

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CLUSTER_NUM = _devprob.get("cluster_num")
TOTAL_CORE_NUM = TOTAL_CLUSTER_NUM * _devprob.get("core_num_per_cluster")
LLC_CACHE_SIZE = _devprob.get("max_l2_cache_size")

configs_dim0 = [
    triton.Config({"BLOCK_SIZE": bs, "IDX_BLOCK_DIM0": 0, "IDX_BLOCK_DIM1": 0})
    for bs in [64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768]
]


@triton.jit
def scatter_st(
    inp,
    cur_src,
    mask,
    dim: tl.constexpr,
    cur_idx,
    inp_dim1,
    idx_offs_dim0,
    idx_offs_dim1,
    ST_USING_CACHE: tl.constexpr,
):
    if dim == 0:
        inp_offs = cur_idx * inp_dim1 + idx_offs_dim1[None, :]
    else:
        inp_offs = idx_offs_dim0[:, None] * inp_dim1 + cur_idx

    if ST_USING_CACHE:
        tl.store(inp + inp_offs, cur_src, mask=mask, cache_modifier=".cg")
    else:
        tl.store(inp + inp_offs, cur_src, mask=mask)


@triton.jit
def scatter_ld_inp_st(
    inp,
    cur_src,
    mask_dim0,
    cur_idx_i32,
    idx_offs_dim0,
    INP_DIM1: tl.constexpr,
):
    mask = mask_dim0[:, None]
    inp_offs = idx_offs_dim0[:, None] * INP_DIM1 + tl.arange(0, INP_DIM1)[None, :]
    cur_inp = tl.load(inp + inp_offs, mask=mask)

    mod_inp = tl.scatter(cur_inp, cur_src, cur_idx_i32, 1)

    tl.store(inp + inp_offs, mod_inp, mask=mask)


def do_config_prune(configs, named_args, **kwargs):
    idx_block_dim01 = set()
    idx_dim0 = named_args["idx_dim0"]
    idx_dim1 = named_args["idx_dim1"]

    # TODO: add sweet-point config and do pruning based on it, i.e. only keep a few configs around it
    for config in configs:
        block_size = config.kwargs["BLOCK_SIZE"]
        # block is too large
        if block_size > idx_dim0 * idx_dim1 and idx_dim0 * idx_dim1 > 512:
            continue
        idx_block_dim0 = block_size // idx_dim1
        idx_block_dim1 = idx_dim1
        # block is too small compared with idx_dim1, so just discard it
        if idx_block_dim0 == 0 and block_size < 512:
            continue
        if idx_block_dim0 == 0:
            idx_block_dim0 = 1
            idx_block_dim1 = block_size
        idx_block_dim01.add((idx_block_dim0, idx_block_dim1))

    pruned_configs = []
    for (idx_block_dim0, idx_block_dim1) in idx_block_dim01:
        pruned_configs.append(
            triton.Config({"IDX_BLOCK_DIM0": idx_block_dim0, "IDX_BLOCK_DIM1": idx_block_dim1}, num_stages=3,
                          num_warps=1))

    return pruned_configs


def determine_using_cache(dim, idx_dim0, idx_dim1, inp_dim0, inp_dim1, dtype_bytes):
    if dim == 0:
        # in each gap (inp_dim0/idx_dim0), average distance is within 64B
        # v1 - - v4 -
        # - v2 - - v5
        # - - v3 - - v6
        return (inp_dim0 * inp_dim1 * dtype_bytes
                <= LLC_CACHE_SIZE) and (inp_dim0 / idx_dim0
                                        <= 64 / dtype_bytes) and (idx_dim1 >= (inp_dim0 / idx_dim0) * (512 / 64))
    else:
        # cache ON: minimum RMW is 512B
        # cache OFF: minimum RMW is 64B
        # cache ON is beneficial when there is at least one data point in each 64B within a cacheline
        return (inp_dim1 * dtype_bytes <= LLC_CACHE_SIZE) and (idx_dim1 >= inp_dim1 // (64 // dtype_bytes))


def determin_ld_inp(idx_dim1, inp_dim1, dtype_bytes, IDX_BLOCK_DIM1):
    #TODO: find out the maximum size, currently set it to 64KB
    #TODO: currently tl.gather/scatter does not support mask, so it's only used for IDX_BLOCK_DIM1==idx_dim1
    #      lift this constraint once mask is supported
    return inp_dim1 * dtype_bytes <= 64 * 1024 and \
           IDX_BLOCK_DIM1 == idx_dim1 and \
           idx_dim1 >= inp_dim1 // (512 // dtype_bytes)


@triton.autotune(
    configs=configs_dim0,
    key=["dim", "src_dim0", "idx_dim0", "idx_dim1", "inp_dim0", "inp_dim1"],
    prune_configs_by={"early_config_prune": do_config_prune},
)
@triton.heuristics({
    "IS_INP_LARGE":
    lambda args: args["inp_dim0"] * args["inp_dim1"] * args["src_dtype_bytes"] >= 2**31,
    "ST_USING_CACHE":
    lambda args: determine_using_cache(args["dim"], args["idx_dim0"], args["idx_dim1"], args["inp_dim0"], args[
        "inp_dim1"], args["src_dtype_bytes"]),
    "LD_INP":
    lambda args: determin_ld_inp(args["idx_dim1"], args["inp_dim1"], args["src_dtype_bytes"], args["IDX_BLOCK_DIM1"]),
    "INP_DIM1":
    lambda args: args["inp_dim1"],
})
@triton.jit(debug=True)
def scatter_kernel_2D(
    src,
    idx,
    inp,
    dim: tl.constexpr,
    src_dim1,
    idx_dim0,
    idx_dim1,
    inp_dim0,
    inp_dim1,
    src_dtype_bytes,
    inp_dim0_threshold,
    IS_INP_LARGE: tl.constexpr,
    ST_USING_CACHE: tl.constexpr,
    IDX_BLOCK_DIM0: tl.constexpr,
    IDX_BLOCK_DIM1: tl.constexpr,
    LD_INP: tl.constexpr,
    INP_DIM1: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)

    # idx size can be much larger than inp, so can be beyond int32
    num_blocks_dim0 = tl.cdiv(idx_dim0, IDX_BLOCK_DIM0)
    num_blocks_dim1 = tl.cdiv(idx_dim1, IDX_BLOCK_DIM1)
    total_num_blocks = num_blocks_dim0.to(tl.int64) * num_blocks_dim1.to(tl.int64)

    for block_id in range(pid, total_num_blocks, step):
        block_id_dim0 = block_id // num_blocks_dim1
        block_id_dim1 = block_id % num_blocks_dim1

        idx_offs_dim0 = block_id_dim0 * IDX_BLOCK_DIM0 + tl.arange(0, IDX_BLOCK_DIM0)
        idx_offs_dim1 = block_id_dim1 * IDX_BLOCK_DIM1 + tl.arange(0, IDX_BLOCK_DIM1)
        mask_dim0 = idx_offs_dim0 < idx_dim0
        mask_dim1 = idx_offs_dim1 < idx_dim1
        mask = mask_dim0[:, None] & mask_dim1[None, :]

        idx_offs = idx_offs_dim0[:, None] * idx_dim1 + idx_offs_dim1[None, :]
        cur_idx = tl.load(idx + idx_offs, mask=mask, other=0)

        src_offs = idx_offs_dim0[:, None] * src_dim1 + idx_offs_dim1[None, :]
        cur_src = tl.load(src + src_offs, mask=mask)

        # check the validity of idx
        cur_idx_i32 = cur_idx.to(tl.int32)
        idx_lt_0 = cur_idx_i32 < 0
        idx_ge_size = cur_idx_i32 >= (inp_dim0 if dim == 0 else inp_dim1)
        all_bool = idx_lt_0 | idx_ge_size
        tl.device_assert(tl.sum(all_bool) == 0, "[triton scatter]: invalid idx")

        idx_offs_dim0_i32 = block_id_dim0.to(tl.int32) * IDX_BLOCK_DIM0 + tl.arange(0, IDX_BLOCK_DIM0)
        idx_offs_dim1_i32 = block_id_dim1.to(tl.int32) * IDX_BLOCK_DIM1 + tl.arange(0, IDX_BLOCK_DIM1)
        cur_idx_i64 = cur_idx.to(tl.int64)
        if dim == 0:
            # TODO: if inp_dim0 is not large, better performance can be achieved by
            #       loading [inp_dim0, IDX_BLOCK_DIM1] onto RAM, just like scatter_ld_inp_st for dim=1
            if not IS_INP_LARGE:
                scatter_st(inp, cur_src, mask, dim, cur_idx_i32, inp_dim1, None, idx_offs_dim1_i32, ST_USING_CACHE)
            else:
                cur_idx_max = tl.max(cur_idx_i32)
                if cur_idx_max < inp_dim0_threshold:
                    scatter_st(inp, cur_src, mask, dim, cur_idx_i32, inp_dim1, None, idx_offs_dim1_i32, ST_USING_CACHE)
                else:
                    scatter_st(inp, cur_src, mask, dim, cur_idx_i64, inp_dim1, None, idx_offs_dim1, ST_USING_CACHE)
        else:
            if LD_INP:
                scatter_ld_inp_st(inp, cur_src, mask_dim0, cur_idx_i32, idx_offs_dim0, INP_DIM1)
            else:
                scatter_st(inp, cur_src, mask, dim, cur_idx_i32, inp_dim1, idx_offs_dim0_i32, None, ST_USING_CACHE)


def scatter(inp, dim, idx, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "[triton scatter]: Invalid dim"
    dim = dim if dim >= 0 else inp.dim() + dim
    assert src.dim() == 2, "the current scatter kernel is only 2D tensor"
    assert idx.dim() == src.dim(), "idx and src should have the same number of dimensions"
    assert inp.dim() == src.dim(), "input and src should have the same number of dimensions"

    inp = inp.contiguous()
    idx = idx.contiguous()
    src = src.contiguous()

    src_shape = src.shape
    idx_shape = idx.shape
    inp_shape = inp.shape

    assert inp_shape[dim] < 2**31 - 1, f"inp.shape[{dim}] should be within int32 range, but now it's {inp_shape[dim]}"

    grid = lambda meta: (TOTAL_CORE_NUM, )

    inp_dtype_bytes = torch.finfo(inp.dtype).bits // 8
    inp_dim0_threshold = ((2**31 - 1) // inp_dtype_bytes) // inp_shape[1]

    if dim == 0:
        scatter_kernel_2D[grid](
            src,
            idx,
            inp,
            dim,
            src_shape[1],
            idx_shape[0],
            idx_shape[1],
            inp_shape[0],
            inp_shape[1],
            torch.finfo(src.dtype).bits // 8,
            inp_dim0_threshold,
        )
    else:
        assert inp_dim0_threshold > 0, f"inp.shape[1] {inp_shape[1]} is too large, not supported yet"
        # compute inp offsets in int32 in each chunk
        dim0_step = inp_dim0_threshold
        for dim0_offs in range(0, idx_shape[0], dim0_step):
            idx_dim0 = dim0_step if (dim0_offs + dim0_step <= idx_shape[0]) else (idx_shape[0] - dim0_offs)
            src_dim0 = idx_dim0
            inp_dim0 = idx_dim0

            scatter_kernel_2D[grid](
                src[dim0_offs:(dim0_offs + src_dim0), :],
                idx[dim0_offs:(dim0_offs + idx_dim0), :],
                inp[dim0_offs:(dim0_offs + inp_dim0), :],
                dim,
                src_shape[1],
                idx_dim0,
                idx_shape[1],
                inp_dim0,
                inp_shape[1],
                torch.finfo(src.dtype).bits // 8,
                inp_dim0_threshold,
            )


@triton.jit
def gather_ld(
    inp,
    mask,
    dim: tl.constexpr,
    cur_idx,
    inp_dim1,
    idx_offs_dim0,
    idx_offs_dim1,
    LD_USING_CACHE: tl.constexpr,
):
    if dim == 0:
        inp_offs = cur_idx * inp_dim1 + idx_offs_dim1[None, :]
    else:
        inp_offs = idx_offs_dim0[:, None] * inp_dim1 + cur_idx

    if LD_USING_CACHE:
        cur_out = tl.load(inp + inp_offs, mask=mask, cache_modifier=".ca")
    else:
        cur_out = tl.load(inp + inp_offs, mask=mask)

    return cur_out


@triton.jit
def gather_ld_inp(inp, mask_dim0, cur_idx_i32, INP_DIM1: tl.constexpr, idx_offs_dim0):
    inp_offs = idx_offs_dim0[:, None] * INP_DIM1 + tl.arange(0, INP_DIM1)[None, :]
    cur_inp = tl.load(inp + inp_offs, mask=mask_dim0[:, None])

    cur_out = tl.gather(cur_inp, cur_idx_i32, 1)

    return cur_out


@triton.autotune(
    configs=configs_dim0,
    key=["idx_dim0", "idx_dim1", "inp_dim0", "inp_dim1"],
    prune_configs_by={"early_config_prune": do_config_prune},
)
@triton.heuristics({
    "IS_INP_LARGE":
    lambda args: args["inp_dim0"] * args["inp_dim1"] * args["inp_dtype_bytes"] >= 2**31,
    "LD_USING_CACHE":
    lambda args: determine_using_cache(args["dim"], args["idx_dim0"], args["idx_dim1"], args["inp_dim0"], args[
        "inp_dim1"], args["inp_dtype_bytes"]),
    "LD_INP":
    lambda args: determin_ld_inp(args["idx_dim1"], args["inp_dim1"], args["inp_dtype_bytes"], args["IDX_BLOCK_DIM1"]),
    "INP_DIM1":
    lambda args: args["inp_dim1"],
})
@triton.jit(debug=True)
def gather_kernel_2D(
    out,
    idx,
    inp,
    dim: tl.constexpr,
    idx_dim0,
    idx_dim1,
    inp_dim0,
    inp_dim1,
    inp_dtype_bytes,
    inp_dim0_threshold,
    IS_INP_LARGE: tl.constexpr,
    LD_USING_CACHE: tl.constexpr,
    IDX_BLOCK_DIM0: tl.constexpr,
    IDX_BLOCK_DIM1: tl.constexpr,
    LD_INP: tl.constexpr,
    INP_DIM1: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)

    # idx size can be much larger than inp, so can be beyond int32
    num_blocks_dim0 = tl.cdiv(idx_dim0, IDX_BLOCK_DIM0)
    num_blocks_dim1 = tl.cdiv(idx_dim1, IDX_BLOCK_DIM1)
    total_num_blocks = num_blocks_dim0.to(tl.int64) * num_blocks_dim1.to(tl.int64)

    for block_id in range(pid, total_num_blocks, step):
        block_id_dim0 = block_id // num_blocks_dim1
        block_id_dim1 = block_id % num_blocks_dim1

        idx_offs_dim0 = block_id_dim0 * IDX_BLOCK_DIM0 + tl.arange(0, IDX_BLOCK_DIM0)
        idx_offs_dim1 = block_id_dim1 * IDX_BLOCK_DIM1 + tl.arange(0, IDX_BLOCK_DIM1)
        mask_dim0 = idx_offs_dim0 < idx_dim0
        mask_dim1 = idx_offs_dim1 < idx_dim1
        mask = mask_dim0[:, None] & mask_dim1[None, :]

        idx_offs = idx_offs_dim0[:, None] * idx_dim1 + idx_offs_dim1[None, :]
        out_offs = idx_offs

        cur_idx = tl.load(idx + idx_offs, mask=mask, other=0)

        # check the validity of idx
        cur_idx_i32 = cur_idx.to(tl.int32)
        idx_lt_0 = cur_idx_i32 < 0
        idx_ge_size = cur_idx_i32 >= (inp_dim0 if dim == 0 else inp_dim1)
        all_bool = idx_lt_0 | idx_ge_size
        tl.device_assert(tl.sum(all_bool) == 0, "[triton scatter]: invalid idx")

        idx_offs_dim0_i32 = block_id_dim0.to(tl.int32) * IDX_BLOCK_DIM0 + tl.arange(0, IDX_BLOCK_DIM0)
        idx_offs_dim1_i32 = block_id_dim1.to(tl.int32) * IDX_BLOCK_DIM1 + tl.arange(0, IDX_BLOCK_DIM1)
        cur_idx_i64 = cur_idx.to(tl.int64)
        if dim == 0:
            # TODO: if inp_dim0 is not large, better performance can be achieved by
            #       loading [inp_dim0, IDX_BLOCK_DIM1] onto RAM, just like gather_ld_inp for dim=1
            if not IS_INP_LARGE:
                cur_out = gather_ld(inp, mask, dim, cur_idx_i32, inp_dim1, None, idx_offs_dim1_i32, LD_USING_CACHE)
            else:
                cur_idx_max = tl.max(cur_idx_i32)
                if cur_idx_max < inp_dim0_threshold:
                    cur_out = gather_ld(inp, mask, dim, cur_idx_i32, inp_dim1, None, idx_offs_dim1_i32, LD_USING_CACHE)
                else:
                    cur_out = gather_ld(inp, mask, dim, cur_idx_i64, inp_dim1, None, idx_offs_dim1, LD_USING_CACHE)
        else:
            if LD_INP:
                cur_out = gather_ld_inp(inp, mask_dim0, cur_idx_i32, INP_DIM1, idx_offs_dim0)
            else:
                cur_out = gather_ld(inp, mask, dim, cur_idx_i32, inp_dim1, idx_offs_dim0_i32, None, LD_USING_CACHE)
        tl.store(out + out_offs, cur_out, mask=mask)


def gather(inp, dim, idx):
    assert dim >= -inp.ndim and dim < inp.ndim, "[triton gather]: Invalid dim"
    dim = dim if dim >= 0 else inp.dim() + dim
    assert idx.dim() == inp.dim(), "idx and src should have the same number of dimensions"

    inp = inp.contiguous()
    idx = idx.contiguous()

    idx_shape = idx.shape
    inp_shape = inp.shape
    assert inp_shape[dim] < 2**31 - 1, f"inp.shape[{dim}] should be within int32 range, but now it's {inp_shape[dim]}"

    out = torch.empty(idx_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (TOTAL_CORE_NUM, )

    inp_dtype_bytes = torch.finfo(inp.dtype).bits // 8
    inp_dim0_threshold = ((2**31 - 1) // inp_dtype_bytes) // inp_shape[1]

    if dim == 0:
        gather_kernel_2D[grid](
            out,
            idx,
            inp,
            dim,
            idx_shape[0],
            idx_shape[1],
            inp_shape[0],
            inp_shape[1],
            torch.finfo(inp.dtype).bits // 8,
            inp_dim0_threshold,
        )
    else:
        assert inp_dim0_threshold > 0, f"inp.shape[1] {inp_shape[1]} is too large, not supported yet"
        # compute inp offsets in int32 in each chunk
        dim0_step = inp_dim0_threshold
        for dim0_offs in range(0, idx_shape[0], dim0_step):
            idx_dim0 = dim0_step if (dim0_offs + dim0_step <= idx_shape[0]) else (idx_shape[0] - dim0_offs)
            inp_dim0 = idx_dim0

            gather_kernel_2D[grid](
                out[dim0_offs:(dim0_offs + idx_dim0), :],
                idx[dim0_offs:(dim0_offs + idx_dim0), :],
                inp[dim0_offs:(dim0_offs + inp_dim0), :],
                dim,
                idx_dim0,
                idx_shape[1],
                inp_dim0,
                inp_shape[1],
                torch.finfo(inp.dtype).bits // 8,
                inp_dim0_threshold,
            )

    return out


def get_idx_shape(src_shape, inp_shape, dim, need_check_res, mode):
    if mode == "gather":
        # gather needs src and idx to have the same shape
        return src_shape

    # scatter
    num_dims = len(src_shape)
    idx_shape = []
    for i in range(num_dims):
        if i == dim and not need_check_res:
            size = src_shape[i]
        else:
            size = min(src_shape[i], inp_shape[i])
        idx_shape.append(random.randint(max(1, size // 2), size))
    return idx_shape


def get_in_out_tensor(src_shape, idx_shape, inp_shape, dtype, dim, need_check_res, mode):
    if dim < 0:
        # 2D
        dim = 2 + dim
    if idx_shape is None:
        idx_shape = get_idx_shape(src_shape, inp_shape, dim, need_check_res, mode)

    src = torch.randn(src_shape, dtype=dtype, device="mlu")
    inp = torch.randn(inp_shape, dtype=dtype, device=src.device)

    dim0, dim1 = idx_shape
    dim_size = idx_shape[dim]
    inp_dim_size = inp_shape[dim]
    if need_check_res:
        idx = torch.empty(idx_shape, dtype=torch.long, device=src.device)
        for i in range(1 if dim == 0 else dim0):
            for j in range(1 if dim == 1 else dim1):
                ij = [i, j]
                ij[dim] = slice(dim_size)
                if dim_size <= inp_dim_size:
                    idx[ij] = torch.randperm(inp_dim_size)[0:dim_size]
                else:
                    idx[ij] = torch.randperm(dim_size) % inp_dim_size
    else:
        idx = torch.randint(0, inp_dim_size, idx_shape, dtype=torch.long, device=src.device)

    return (src, idx, inp)


def gen_src_inp_shapes(inp_shapes, mode):
    """ For scatter dim 0/1, and gather dim 0
    """
    # if scatter, inp and src can have independent shapes
    # if gather, src should have no larger shapes for dims other than idx_dim
    ratio = 2 if mode == "scatter" else 1

    src_inp_shapes = []
    for inp_shape in inp_shapes:
        inp_dim0, inp_dim1 = inp_shape

        src_inp_shapes.append((inp_dim0 * 2, inp_dim1 * ratio, inp_dim0, inp_dim1))
        src_inp_shapes.append((inp_dim0 * 1, inp_dim1 * ratio, inp_dim0, inp_dim1))
        if inp_dim0 >= 2:
            src_inp_shapes.append((inp_dim0 // 2, inp_dim1 * ratio, inp_dim0, inp_dim1))
    return src_inp_shapes


def gen_src_inp_shapes_gather_dim1(inp_shapes):
    src_inp_shapes = []
    for inp_shape in inp_shapes:
        inp_dim0, inp_dim1 = inp_shape

        src_inp_shapes.append((inp_dim0, inp_dim1 * 1, inp_dim0, inp_dim1))
        src_inp_shapes.append((inp_dim0, inp_dim1 * 2, inp_dim0, inp_dim1))
        if inp_dim1 >= 2:
            src_inp_shapes.append((inp_dim0, inp_dim1 // 2, inp_dim0, inp_dim1))
    return src_inp_shapes


TEST_SHAPES = [
    (64, 64),
    #(256, 256),
    #(1024, 1024),
    #(4096, 4096),
    #(1024, 65536),
    #(1229312, 512),
    #(1433600, 512),
]

dtypes = [torch.float32, torch.float16]
dims = [0, 1]


@pytest.mark.parametrize("src_inp_shapes", gen_src_inp_shapes(TEST_SHAPES, "scatter"))
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dims)
def test_scatter(src_inp_shapes, dtype, dim):
    src_dim0, src_dim1, inp_dim0, inp_dim1 = src_inp_shapes
    need_check_res = True
    src, idx, inp = get_in_out_tensor((src_dim0, src_dim1), None, (inp_dim0, inp_dim1), dtype, dim, need_check_res,
                                      "scatter")
    print(src.shape, idx.shape, inp.shape, dtype, dim)
    inp_ref = inp.clone()

    scatter(inp, dim, idx, src)
    inp_ref.scatter_(dim, idx, src)

    torch.testing.assert_close(inp_ref, inp, atol=0, rtol=0)


@pytest.mark.parametrize("src_inp_shapes", gen_src_inp_shapes(TEST_SHAPES, "gather"))
@pytest.mark.parametrize("dtype", dtypes)
def test_gather_dim0(src_inp_shapes, dtype):
    dim = 0
    print(src_inp_shapes, dtype, dim)
    src_dim0, src_dim1, inp_dim0, inp_dim1 = src_inp_shapes
    need_check_res = True
    _, idx, inp = get_in_out_tensor((src_dim0, src_dim1), None, (inp_dim0, inp_dim1), dtype, dim, need_check_res,
                                    "gather")

    out = gather(inp, dim, idx)
    out_ref = torch.gather(inp, dim, idx)

    torch.testing.assert_close(out_ref, out, atol=0, rtol=0)


@pytest.mark.parametrize("src_inp_shapes", gen_src_inp_shapes_gather_dim1(TEST_SHAPES))
@pytest.mark.parametrize("dtype", dtypes)
def test_gather_dim1(src_inp_shapes, dtype):
    dim = 1
    print(src_inp_shapes, dtype, dim)
    src_dim0, src_dim1, inp_dim0, inp_dim1 = src_inp_shapes
    need_check_res = True
    _, idx, inp = get_in_out_tensor((src_dim0, src_dim1), None, (inp_dim0, inp_dim1), dtype, dim, need_check_res,
                                    "gather")

    out = gather(inp, dim, idx)
    out_ref = torch.gather(inp, dim, idx)

    torch.testing.assert_close(out_ref, out, atol=0, rtol=0)


def gen_perf_configs(mode, dim):
    src_idx_inp_shapes = []
    need_check_res = False

    if mode == "gather" and dim == 1:
        all_src_inp_shapes = gen_src_inp_shapes_gather_dim1(TEST_SHAPES)
    else:
        all_src_inp_shapes = gen_src_inp_shapes(TEST_SHAPES, mode)

    for src_inp_shapes in all_src_inp_shapes:
        src_dim0, src_dim1, inp_dim0, inp_dim1 = src_inp_shapes
        src_shape = [src_dim0, src_dim1]
        inp_shape = [inp_dim0, inp_dim1]
        idx_shape = get_idx_shape(src_shape, inp_shape, dim, need_check_res, mode)
        # To ensure the indices generated for triton and torch are the same
        rand_seed = random.randint(0, 2**32 - 1)
        # testing.py: df[first_x].to_numpy(): requires string or float, not list
        src_idx_inp_shapes.append((str(src_shape), str(idx_shape), str(inp_shape), str(rand_seed)))

    perf_configs = []
    for dtype in dtypes:
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=["src_shape" if mode == "scatter" else "out_shape", "idx_shape", "inp_shape", "rand_seed"],
                x_vals=src_idx_inp_shapes,
                line_arg="provider",
                line_vals=["Triton", "Torch"],
                line_names=["Triton (ms)", "Torch (ms)"],
                styles=[("yellow", "-"), ("red", "-")],
                ylabel="Time",
                plot_name=f"{mode}-performance-dtype_{dtype}-dim_{dim}",
                args={"dtype": dtype, "dim": dim, "need_check_res": need_check_res},
            ))
    return perf_configs


def benchmark_scatter_base(src_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider):
    print(provider, src_shape, idx_shape, inp_shape, rand_seed, dtype, dim)
    src_shape = json.loads(src_shape)
    idx_shape = json.loads(idx_shape)
    inp_shape = json.loads(inp_shape)
    rand_seed = json.loads(rand_seed)

    torch.mlu.manual_seed(rand_seed)
    src, idx, inp = get_in_out_tensor(src_shape, idx_shape, inp_shape, dtype, dim, need_check_res, "scatter")

    if provider == "Triton":
        ms_ = triton.testing.do_bench(lambda: scatter(inp, dim, idx, src))
        ms = triton.testing.do_bench(lambda: scatter(inp, dim, idx, src), warmup=10 * ms_, rep=102 * ms_)
    if provider == "Torch":
        inp_ref = inp.clone()
        ms_ = triton.testing.do_bench(lambda: inp_ref.scatter_(dim, idx, src))
        ms = triton.testing.do_bench(lambda: inp_ref.scatter_(dim, idx, src), warmup=10 * ms_, rep=102 * ms_)
    return ms


@triton.testing.perf_report(gen_perf_configs("scatter", 0))
def benchmark_scatter_dim0(src_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider):
    return benchmark_scatter_base(src_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider)


@triton.testing.perf_report(gen_perf_configs("scatter", 1))
def benchmark_scatter_dim1(src_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider):
    return benchmark_scatter_base(src_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider)


def benchmark_gather_base(out_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider):
    print(provider, out_shape, idx_shape, inp_shape, rand_seed, dtype, dim)
    out_shape = json.loads(out_shape)
    idx_shape = json.loads(idx_shape)
    inp_shape = json.loads(inp_shape)
    rand_seed = json.loads(rand_seed)

    torch.mlu.manual_seed(rand_seed)
    _, idx, inp = get_in_out_tensor(out_shape, idx_shape, inp_shape, dtype, dim, need_check_res, "gather")

    if provider == "Triton":
        ms_ = triton.testing.do_bench(lambda: gather(inp, dim, idx))
        ms = triton.testing.do_bench(lambda: gather(inp, dim, idx), warmup=10 * ms_, rep=102 * ms_)
    if provider == "Torch":
        inp_ref = inp.clone()
        ms_ = triton.testing.do_bench(lambda: torch.gather(inp, dim, idx))
        ms = triton.testing.do_bench(lambda: torch.gather(inp, dim, idx), warmup=10 * ms_, rep=102 * ms_)
    return ms


@triton.testing.perf_report(gen_perf_configs("gather", 0))
def benchmark_gather_dim0(out_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider):
    return benchmark_gather_base(out_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider)


@triton.testing.perf_report(gen_perf_configs("gather", 1))
def benchmark_gather_dim1(out_shape, idx_shape, inp_shape, dtype, rand_seed, dim, need_check_res, provider):
    return benchmark_gather_base(out_shape, idx_shape, inp_shape, rand_seed, dtype, dim, need_check_res, provider)


if __name__ == "__main__":
    benchmark_scatter_dim0.run(show_plots=True, print_data=True)
    benchmark_scatter_dim1.run(show_plots=True, print_data=True)
    benchmark_gather_dim0.run(show_plots=True, print_data=True)
    benchmark_gather_dim1.run(show_plots=True, print_data=True)
