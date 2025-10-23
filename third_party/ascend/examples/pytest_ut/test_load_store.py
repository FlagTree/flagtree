import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + x_index, tmp2, xmask)


# require: all data (4d and 5d) can be placed into but without ub overflow
@triton.jit
def triton_load_store_multi_d(
    in_ptr0, out_ptr0, 
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
    SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
    STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    offsets = tl.program_id(0)

    offsets = offsets + tl.arange(0, BLOCK_0) * STRIDE_0
    masks = tl.arange(0, BLOCK_0) < SHAPE_0
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        masks = masks[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        masks = masks[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        masks = masks[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        masks = masks[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    tmp_in = tl.load(in_ptr0 + offsets, masks)
    tmp_out = tmp_in
    tl.store(out_ptr0 + offsets, tmp_out, masks)


@triton.jit
def triton_load_store_sle_mask(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index <= xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + x_index, tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (8, 8, 4), 2, 128, 64],
                             ['float16', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 8, 4), 2, 128, 64],
                         ]
                         )
def test_load_store(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_ref = x0
    # triton结果
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store[(ncore, )](x0, y_cal, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (8, 4, 16, 16)],
                             ['float16', (8, 4, 16, 16)],
                             ['int8', (8, 4, 16, 16)],
                             ['float32', (8, 8, 4, 4)],
                             ['float16', (8, 8, 4, 4)],
                             ['int8', (8, 8, 4, 4)],
                             ['float32', (3, 8, 2, 16, 16)],
                             ['float16', (3, 8, 2, 16, 16)],
                             ['int8', (9, 8, 8, 16, 16)],
                             ['float32', (11, 8, 8, 4, 4)],
                             ['float16', (11, 8, 8, 4, 4)],
                             ['int8', (11, 8, 8, 4, 4)],
                         ]
                         )
def test_load_store_multi_d(param_list):
    # 生成数据
    dtype, shape = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_expect = x0
    y_actual = test_common.generate_tensor(shape, dtype).npu()
    # triton结果
    blocks = list(x0.size())
    shapes = list(x0.stride())
    while len(blocks) < 5:
        blocks.append(1)
        shapes.append(1)
    triton_load_store_multi_d[(1, )](x0, y_actual, *blocks, *blocks, *shapes)
    # 比较结果
    test_common.validate_cmp(dtype, y_actual, y_expect)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (8, 8, 4), 2, 128, 64],
                             ['float16', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 8, 4), 2, 128, 64],
                         ]
                         )
def test_load_store_sle_mask(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_ref = x0
    # triton结果
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store_sle_mask[(ncore, )](x0, y_cal, x0.numel() - 1, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, y_cal, y_ref)
