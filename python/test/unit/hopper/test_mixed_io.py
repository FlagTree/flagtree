import pytest

HAS_TORCH = False
HAS_PADDLE = False
try:
    import torch
    from torch.testing import assert_close
    HAS_TORCH = True
    dtype_mapping = {
        'float16': torch.float16,
        'float32': torch.float32,
    }
    
except :
    import paddle
    HAS_PADDLE = True
    dtype_mapping = {
        'float16': paddle.float16,
        'float32': paddle.float32,
    }


import triton
import triton.language as tl




@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(n_elements, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    x = tl.load(x_block_ptr, boundary_check=(0, ), padding_option='zero')

    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize('SIZE,BLOCK_SIZE,dtype_str',
                         [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']])
def test_add(SIZE, BLOCK_SIZE, dtype_str):
    dtype = dtype_mapping[dtype_str]
    if HAS_TORCH:
        output = torch.empty(SIZE, device='cuda', dtype=dtype)
        x = torch.randn(SIZE, device='cuda', dtype=dtype)
        y = torch.randn(SIZE, device='cuda', dtype=dtype)
    else:
        output = paddle.zeros([SIZE], dtype=dtype)
        x = paddle.randn([SIZE], dtype=dtype)
        y = paddle.randn([SIZE], dtype=dtype)
    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    output_ref = x + y
    if HAS_TORCH:
        torch.set_printoptions(profile='full')
        assert_close(output, output_ref, rtol=1e-2, atol=1e-3, check_dtype=False)
    else:
        paddle.set_printoptions()
        paddle.allclose(output, output_ref, rtol=1e-2, atol=1e-3)


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x_ptr = tl.make_block_ptr(base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn), offsets=(0, 0),
                              block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)


@pytest.mark.parametrize('BLOCK_M,BLOCK_N,dtype_str', [(128, 64, dtype_str) for dtype_str in ['float16']])
def test_load_reduce(BLOCK_M, BLOCK_N, dtype_str):
    dtype = dtype_mapping[dtype_str]
    if HAS_TORCH:
        x = torch.randn((BLOCK_M, BLOCK_N), device='cuda', dtype=dtype)
        y = torch.empty((BLOCK_M, ), device='cuda', dtype=dtype)
    else:
        x = paddle.randn([BLOCK_M, BLOCK_N], dtype=dtype)
        y = paddle.empty([BLOCK_M], dtype=dtype)


    if HAS_TORCH:
        load_reduce_kernel[(1, )](x, y, x.stride(0), x.stride(1), y.stride(0), BLOCK_M, BLOCK_N)

        golden = x.max(dim=1)[0]
        torch.set_printoptions(profile='full')
        assert_close(y, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
    else:
        load_reduce_kernel[(1, )](x, y, x.strides[0], x.strides[1], y.strides[0], BLOCK_M, BLOCK_N)

        golden = x.max(axis=1)
        paddle.set_printoptions()
        paddle.allclose(y, golden, rtol=1e-2, atol=1e-3)
