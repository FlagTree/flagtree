import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

types = [
    "float32",
    "float16",
    "bfloat16",
]

shapes = [
    # 3,
    # 32,
    37,
    # 256,
    # 781,
]


@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
@pytest.mark.parametrize("val", ['1.0', 'nan', 'inf', '-inf'])
def test_isfinited(sigtype, N, val):

    def torch_func(x0):
        res = torch.isfinite(x0)
        return res


    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = tl.math.isfinited(x0)
        tl.store(out_ptr0 + idx, ret)


    def triton_func(x0, N):
        out = torch.zeros(x0.shape, dtype=torch.bool).npu()
        triton_kernel[1, 1, 1](out, x0, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x0[1] = float(val)

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0, N)

    test_common.validate_cmp("bool", triton_cal, torch_ref)


@pytest.mark.parametrize("N", shapes)
@pytest.mark.parametrize("val", ['1.0', 'nan', 'inf', '-inf'])
def test_finitef(N, val):

    def torch_func(x0):
        res = torch.isfinite(x0)
        return res


    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = tl.math.finitef(x0)
        tl.store(out_ptr0 + idx, ret)


    def triton_func(x0, N):
        out = torch.zeros(x0.shape, dtype=torch.bool).npu()
        triton_kernel[1, 1, 1](out, x0, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype='float32').npu()
    x0[1] = float(val)

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0, N)
    test_common.validate_cmp("bool", triton_cal, torch_ref)


invalid_types = [
    "int32",
]


@pytest.mark.parametrize("sigtype", invalid_types)
@pytest.mark.parametrize("N", shapes)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Expected dtype fp16/fp32/bf16, but got")
def test_isfinited_invalid_dtype(sigtype, N):

    def torch_func(x0):
        res = torch.isfinite(x0)
        return res


    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = tl.math.isfinited(x0)
        tl.store(out_ptr0 + idx, ret)


    def triton_func(x0, N):
        out = torch.zeros(x0.shape, dtype=torch.bool).npu()
        triton_kernel[1, 1, 1](out, x0, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x0[1] = float('nan')

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0, N)
    test_common.validate_cmp("bool", triton_cal, torch_ref)
    assert triton_cal[1] == True
