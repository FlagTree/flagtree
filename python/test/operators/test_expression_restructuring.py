import triton
import triton.language as tl
import torch

import pytest

VEC_SHAPES = [[64, 640], [32, 128], [128, 256]]


def custom_rand_strided(shape, strides, device, dtype, seed=0):
    torch.manual_seed(seed)
    total_size = sum((s - 1) * st for s, st in zip(shape, strides)) + 1
    storage = torch.randn(total_size, device=device, dtype=dtype)
    return torch.as_strided(storage, size=shape, stride=strides)


def torch_equivalent(arg_0, arg_1, arg_2, arg_3):
    reshaped_arg_0 = arg_0.view(arg_2.shape[0], arg_2.shape[0], arg_2.shape[2])
    reshaped_arg_3 = arg_3.squeeze(-1)
    tmp0 = -reshaped_arg_0
    tmp4 = arg_1 * arg_2
    tmp7 = reshaped_arg_3 + 1e-06
    tmp8 = tmp4 / tmp7.unsqueeze(-1)
    tmp9 = tmp8 / tmp7.unsqueeze(-1)
    result = tmp0 * tmp9
    return result


@triton.jit
def expression_restructuring_function_test(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, rnumel):
    XBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    RBLOCK: tl.constexpr = 1024
    xindex = tl.full([1], xoffset, tl.int32)
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask, other=0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp3 = tl.load(in_ptr2 + (r1 + (rnumel * x0)), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp4 = tmp2 * tmp3
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 / tmp7
    tmp9 = tmp8 / tmp7
    tmp10 = tmp1 * tmp9
    tl.store(out_ptr2 + (r1 + (rnumel * x0)), tmp10, rmask)


@pytest.mark.parametrize("vec_shape", VEC_SHAPES)
def test_accruacy_kernel(vec_shape):
    x = vec_shape[0]
    y = vec_shape[1]
    arg_0 = custom_rand_strided((x * x, y), (y, 1), dtype=torch.float32, device='cuda')
    arg_1 = custom_rand_strided((y, ), (1, ), dtype=torch.float32, device='cuda')
    arg_2 = custom_rand_strided((x, x, y), (x * y, y, 1), dtype=torch.float32, device='cuda')
    arg_3 = custom_rand_strided((x, x, 1), (x, 1, 1), dtype=torch.float32, device='cuda')
    triton_result = custom_rand_strided((x, x, y), (x * y, y, 1), dtype=torch.float32, device='cuda')
    grid = lambda meta: (x * x, )
    expression_restructuring_function_test[grid](arg_0, arg_1, arg_2, arg_3, triton_result, y)
    torch_result = torch_equivalent(arg_0, arg_1, arg_2, arg_3)
    torch.testing.assert_close(triton_result, torch_result)
