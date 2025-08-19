try:
    import torch
    HAS_TORCH = True
    HAS_PADDLE = False
except Exception:
    import paddle
    HAS_TORCH = False
    HAS_PADDLE = True

import pytest
import triton
import triton.language as tl
import triton.ops
from triton.testing import get_dram_gbps, get_max_tensorcore_tflops, nvsmi


# ---- helpers: framework-agnostic wrappers ----

def device_capability_major():
    if HAS_TORCH:
        return torch.cuda.get_device_capability()[0]
    elif HAS_PADDLE:
        return paddle.device.cuda.get_device_capability()[0]
    else:
        raise RuntimeError("No frame(Paddle/Torch) detected.")

def set_stream_default():
    if HAS_TORCH:
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
    else:
        # Paddle: ensure we're on GPU; stream control is optional for tests
        paddle.device.set_device('gpu')

def manual_seed(seed: int):
    if HAS_TORCH:
        torch.manual_seed(seed)
    else:
        paddle.seed(seed)

def dtype_from_str(dtype_str: str):
    if HAS_TORCH:
        mapping = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'float64': torch.float64,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
        }
    else:
        # Paddle supported dtypes (no int8/bfloat16 in some versions)
        mapping = {
            'float16': paddle.float16,
            'float32': paddle.float32,
            'float64': paddle.float64,
            'bool': paddle.bool,
            'int16': paddle.int16,
            'int32': paddle.int32,
            'int64': paddle.int64,
        }
        # optional bfloat16 support
        if hasattr(paddle, "bfloat16"):
            mapping['bfloat16'] = paddle.bfloat16
    return mapping[dtype_str]

def empty(shape, dtype):
    if HAS_TORCH:
        return torch.empty(shape, dtype=dtype, device='cuda')
    else:
        return paddle.empty(shape, dtype=dtype)

def randn(shape, dtype):
    if HAS_TORCH:
        return torch.randn(shape, dtype=dtype, device='cuda')
    else:
        # ensure deterministic behavior matches seed
        return paddle.randn(shape, dtype=dtype)

def randn_like(x):
    if HAS_TORCH:
        return torch.randn_like(x)
    else:
        return paddle.randn(x.shape, dtype=x.dtype)

def randint(low, high, shape, dtype):
    if HAS_TORCH:
        return torch.randint(low, high, shape, dtype=dtype, device='cuda')
    else:
        # Paddle 只支持 int32 和 int64
        if dtype in ["int8", "uint8", "int16", paddle.int8, paddle.int16]:
            # 用 int32 生成，再 cast 回去
            x = paddle.randint(low=low, high=high, shape=shape, dtype="int32")
            return paddle.cast(x, dtype)
        elif dtype in ["int32", "int64", paddle.int32, paddle.int64]:
            return paddle.randint(low=low, high=high, shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Paddle randint does not support dtype {dtype}")

def transpose_2d(b):
    if HAS_TORCH:
        return b.t()
    else:
        return paddle.transpose(b, [1, 0])

def requires_grad_(x, flag=True):
    if HAS_TORCH:
        return x.requires_grad_(flag)
    else:
        x.stop_gradient = not flag
        return x

def numel(x):
    if HAS_TORCH:
        return x.numel()
    else:
        return int(paddle.numel(x))

def element_size_of_dtype(dtype):
    # bytes per element
    if HAS_TORCH:
        return torch.tensor([], dtype=dtype).element_size()
    else:
        # map sizes manually; extend if needed
        import numpy as np
        mapping = {
            getattr(paddle, "float16"): 2,
            getattr(paddle, "bfloat16", object()): 2,  # if present
            paddle.float32: 4,
            paddle.float64: 8,
            paddle.int16: 2,
            paddle.int32: 4,
            paddle.int64: 8,
            paddle.bool: 1,
        }
        return mapping[dtype]

def bench(fn):
    # prefer cudagraph bench when torch is available
    if hasattr(triton.testing, "do_bench_cudagraph") and HAS_TORCH:
        return triton.testing.do_bench_cudagraph(fn)
    else:
        return triton.testing.do_bench(fn)


# ---- device name ----

DEVICE_NAME = {7: 'v100', 8: 'a100'}[device_capability_major()]


#######################
# Utilities
#######################


def print_perf(cur_ms, cur_util, ref_util):
    # print on the same line cur_ms, cur_util and ref_util with 3 decimal places
    print(f'{cur_ms:.3f} ms \t cur: {cur_util:.3f} \t ref: {ref_util:.3f} \t dif={cur_util - ref_util:.3f}', end='\t')


#######################
# Matrix Multiplication
#######################

sm_clocks = {'v100': 1350, 'a100': 1350}
mem_clocks = {'v100': 877, 'a100': 1215}

matmul_data = {
    'a100': {
        # square
        (512, 512, 512): {'float16': 0.108, 'float32': 0.097, 'int8': 0.05},
        (1024, 1024, 1024): {'float16': 0.355, 'float32': 0.313, 'int8': 0.169},
        (2048, 2048, 2048): {'float16': 0.653, 'float32': 0.532, 'int8': 0.34},
        (8192, 8192, 8192): {'float16': 0.839, 'float32': 0.754, 'int8': 0.51},
        # tall-skinny
        (16, 1024, 1024): {'float16': 0.015, 'float32': 0.009, 'int8': 0.005},
        (16, 4096, 4096): {'float16': 0.080, 'float32': 0.051, 'int8': 0.026},
        (16, 8192, 8192): {'float16': 0.083, 'float32': 0.077, 'int8': 0.043},
        (64, 1024, 1024): {'float16': 0.045, 'float32': 0.023, 'int8': 0.017},
        (64, 4096, 4096): {'float16': 0.170, 'float32': 0.000, 'int8': 0.097},
        (64, 8192, 8192): {'float16': 0.227, 'float32': 0.000, 'int8': 0.174},
        (1024, 64, 1024): {'float16': 0.040, 'float32': 0.046, 'int8': 0.017},
        (4096, 64, 4096): {'float16': 0.160, 'float32': 0.214, 'int8': 0.102},
        (8192, 64, 8192): {'float16': 0.272, 'float32': 0.000, 'int8': 0.177},
        # test EVEN_K==False
        (8192, 8192, 8176): {'float16': 0.828, 'float32': 0.743, 'int8': 0.51},
    }
}


@pytest.mark.parametrize('M, N, K, dtype_str', [(M, N, K, dtype_str)
                                                for M, N, K in matmul_data[DEVICE_NAME].keys()
                                                for dtype_str in ['float16']])
def test_matmul(M, N, K, dtype_str):
    set_stream_default()
    if dtype_str in ['float32', 'int8'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test float32 & int8 on a100')
    if (M, N, K) in [(64, 4096, 4096), (64, 8192, 8192), (8192, 64, 8192)] and dtype_str == 'float32':
        pytest.skip('Out of shared memory in float32')
    dtype = dtype_from_str(dtype_str)
    manual_seed(0)
    ref_gpu_util = matmul_data[DEVICE_NAME][(M, N, K)][dtype_str]
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    if dtype_str == 'int8':
        a = randint(-128, 127, (M, K), dtype=dtype)
        b = randint(-128, 127, (N, K), dtype=dtype)
        b = transpose_2d(b)  # only test row-col layout
    else:
        a = randn((M, K), dtype=dtype)
        b = randn((K, N), dtype=dtype)
    fn = lambda: triton.ops.matmul(a, b)
    ms = bench(fn)
    cur_gpu_perf = 2. * M * N * K / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Element-Wise
#######################


@triton.jit
def _add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


elementwise_data = {
    'a100': {
        1024 * 16: {'float16': 0.031, 'float32': 0.060},
        1024 * 64: {'float16': 0.120, 'float32': 0.224},
        1024 * 256: {'float16': 0.394, 'float32': 0.691},
        1024 * 1024: {'float16': 1.06, 'float32': 1.453},
        1024 * 16384: {'float16': 0.832, 'float32': 0.862},
        1024 * 65536: {'float16': 0.873, 'float32': 0.882},
        # Non pow 2
        1020 * 100: {'float16': 0.173, 'float32': 0.327},
        10003 * 7007: {'float16': 0.522, 'float32': 0.873},
    }
}


@pytest.mark.parametrize('N', elementwise_data[DEVICE_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
def test_elementwise(N, dtype_str):
    set_stream_default()
    manual_seed(0)
    if dtype_str in ['bfloat16'] and DEVICE_NAME != 'a100':
        pytest.skip('Only test bfloat16 on a100')
    if HAS_PADDLE and (dtype_str == 'bfloat16') and not hasattr(paddle, "bfloat16"):
        pytest.skip('Paddle bfloat16 not supported in this build')
    dtype = dtype_from_str(dtype_str)
    ref_dtype_str = 'float16' if dtype_str == 'bfloat16' else dtype_str
    ref_gpu_util = elementwise_data[DEVICE_NAME][N][ref_dtype_str]
    max_gpu_perf = get_dram_gbps()
    z = empty((N, ), dtype=dtype)
    x = randn_like(z)
    y = randn_like(z)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _add[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = bench(fn)
    elem_size = element_size_of_dtype(dtype)
    cur_gpu_perf = 3. * N * elem_size / ms * 1e-6
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Flash-Attention
#######################

flash_attention_data = {
    "a100": {
        (4, 48, 4096, 64, True, True, 'forward', 'float16'): 0.542,
        (4, 48, 4096, 64, True, True, 'forward', 'bfloat16'): 0.471,
        (4, 48, 1024, 16, True, True, 'forward', 'float32'): 0.155,
        (4, 48, 4096, 64, True, True, 'backward', 'float16'): 0.232,
        (4, 48, 4096, 64, True, True, 'backward', 'bfloat16'): 0.231,
        (4, 48, 1024, 16, True, True, 'backward', 'float32'): 0.138,
        (4, 48, 4096, 64, True, False, 'forward', 'float16'): 0.306,
        (4, 48, 4096, 64, True, False, 'forward', 'bfloat16'): 0.266,
        (4, 48, 1024, 16, True, False, 'forward', 'float32'): 0.098,
        (4, 48, 4096, 64, True, False, 'backward', 'float16'): 0.157,
        (4, 48, 4096, 64, True, False, 'backward', 'bfloat16'): 0.157,
        (4, 48, 1024, 16, True, False, 'backward', 'float32'): 0.092,
        (4, 48, 4096, 64, False, True, 'forward', 'float16'): 0.541,
        (4, 48, 4096, 64, False, True, 'forward', 'bfloat16'): 0.471,
        (4, 48, 1024, 16, False, True, 'forward', 'float32'): 0.150,
        (4, 48, 4096, 64, False, True, 'backward', 'float16'): 0.291,
        (4, 48, 4096, 64, False, True, 'backward', 'bfloat16'): 0.255,
        (4, 48, 1024, 16, False, True, 'backward', 'float32'): 0.144,
        (4, 48, 4096, 64, False, False, 'forward', 'float16'): 0.306,
        (4, 48, 4096, 64, False, False, 'forward', 'bfloat16'): 0.266,
        (4, 48, 1024, 16, False, False, 'forward', 'float32'): 0.098,
        (4, 48, 4096, 64, False, False, 'backward', 'float16'): 0.159,
        (4, 48, 4096, 64, False, False, 'backward', 'bfloat16'): 0.159,
        (4, 48, 1024, 16, False, False, 'backward', 'float32'): 0.088,
    }
}


@pytest.mark.parametrize("dtype_str", ['float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize("mode", ['forward', 'backward'])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seq_par", [True, False])
@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [[4, 48, 4096, 64]])
def test_flash_attention(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str):
    set_stream_default()
    is_backward = mode == 'backward'
    capability = device_capability_major()
    if capability < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
    manual_seed(20)
    if HAS_PADDLE and dtype_str == 'bfloat16' and not hasattr(paddle, "bfloat16"):
        pytest.skip('Paddle bfloat16 not supported in this build')
    dtype = dtype_from_str(dtype_str)
    # init data
    if dtype_str == 'float32':
        N_CTX = 1024
        D_HEAD = 16
    q = empty((Z, H, N_CTX, D_HEAD), dtype=dtype)
    k = empty((Z, H, N_CTX, D_HEAD), dtype=dtype)
    v = empty((Z, H, N_CTX, D_HEAD), dtype=dtype)
    if HAS_TORCH:
        q = q.normal_(mean=0.1, std=0.2)
        k = k.normal_(mean=0.4, std=0.2)
        v = v.normal_(mean=0.3, std=0.2)
    else:
        q = q + randn(q.shape, dtype=dtype) * 0.2 + 0.1
        k = k + randn(k.shape, dtype=dtype) * 0.2 + 0.4
        v = v + randn(v.shape, dtype=dtype) * 0.2 + 0.3
    q = requires_grad_(q, True)
    k = requires_grad_(k, True)
    v = requires_grad_(v, True)
    sm_scale = 0.2
    # benchmark
    fn = lambda: triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    if is_backward:
        o = fn()
        do = randn_like(o)
        if HAS_TORCH:
            fn = lambda: o.backward(do, retain_graph=True)
        else:
            def _backward():
                o.backward(gradient=do, retain_graph=True)
            fn = _backward
    ms = bench(fn)
    # compute flops
    flops_per_matmul = 2. * Z * H * N_CTX * N_CTX * D_HEAD * 0.5
    total_flops = 2 * flops_per_matmul
    if is_backward:
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    cur_gpu_perf = total_flops / ms * 1e-9
    # maximum flops
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    ref_gpu_util = flash_attention_data[DEVICE_NAME][(Z, H, N_CTX, D_HEAD, seq_par, causal, mode, dtype_str)]
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)


#######################
# Reduction
#######################


@triton.jit
def _sum(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # run in a loop to only to make it compute bound.
    for i in range(100):
        x = tl.sum(x, axis=0) + y

    tl.store(output_ptr + offsets, x, mask=mask)


reduction_data = {
    'a100': {
        1024 * 16384: {'float16': 0.016, 'float32': 0.031, 'int16': 0.022, 'int32': 0.048},
        1024 * 65536: {'float16': 0.016, 'float32': 0.032, 'int16': 0.022, 'int32': 0.049},
    }
}


@pytest.mark.parametrize('N', reduction_data[DEVICE_NAME].keys())
@pytest.mark.parametrize("dtype_str", ['float16', 'float32', 'int16', 'int32'])
def test_reductions(N, dtype_str):
    set_stream_default()
    manual_seed(0)
    dtype = dtype_from_str(dtype_str)
    ref_gpu_util = reduction_data[DEVICE_NAME][N][dtype_str]
    cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
    max_gpu_perf = get_max_tensorcore_tflops(dtype, clock_rate=cur_sm_clock * 1e3)
    z = empty((N, ), dtype=dtype)
    if dtype_str in ['float16', 'float32']:
        x = randn_like(z)
        y = randn_like(z)
    else:
        # int16/int32 range
        if HAS_TORCH:
            info = torch.iinfo(dtype)
            low, high = info.min, info.max
        else:
            if dtype_str == 'int16':
                low, high = -32768, 32767
            else:
                low, high = -2147483648, 2147483647
        x = randint(low, high, (N, ), dtype=dtype)
        y = randint(low, high, (N, ), dtype=dtype)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE']), )
    fn = lambda: _sum[grid](x, y, z, N, BLOCK_SIZE=1024)
    ms = bench(fn)
    cur_gpu_perf = 100. * 2. * N / ms * 1e-9
    cur_gpu_util = cur_gpu_perf / max_gpu_perf
    print_perf(ms, cur_gpu_util, ref_gpu_util)
    triton.testing.assert_close(cur_gpu_util, ref_gpu_util, atol=0.02, rtol=0.01)

if __name__ == '__main__':
    test_matmul(M = 512, N = 512, K = 512, dtype_str = 'float16')