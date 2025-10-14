import pytest
try:
    import paddle
except:
    pytest.skip("Paddle not installed — skipping tests.", allow_module_level=True)

import triton
import triton.language as tl
from test_core_paddle import (_test_binary, float_dtypes, int_dtypes, numpy_random,
                       uint_dtypes)


@pytest.mark.interpreter
@pytest.mark.parametrize(
    "dtype", int_dtypes + uint_dtypes + float_dtypes + ["bfloat16"]
)
@pytest.mark.parametrize("op", ["maximum", "minimum"])
def test_maximum_minium(dtype, op, device):
    expr = f"tl.{op}(x, y)"
    numpy_expr = f"np.{op}(x, y)"
    _test_binary(dtype, dtype, expr, numpy_expr, device=device)


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ["int32", "float16", "float32"])
def test_sort(M, N, descending, dtype_str, device):
    @triton.jit
    def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.sort(x, descending=descending)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = paddle.to_tensor(x).to(device)
    y = paddle.compat.sort(x, descending=descending)[0]
    z = paddle.empty_like(x)
    sort_kernel[
        1,
    ](x, z, N, M, descending, num_warps=8)
    assert (y == z).all(), (y, z)


@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("dtype_str", ["int32", "float16", "float32"])
def test_flip(M, N, dtype_str, device):
    @triton.jit
    def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.flip(x)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = paddle.to_tensor(x).to(device)
    y = paddle.flip(x=x, axis=(1,))
    z = paddle.empty_like(x, device=device)
    flip_kernel[
        1,
    ](x, z, N, M, num_warps=8)
    assert (y == z).all(), (y, z)
