import pytest
HAS_TORCH = False
HAS_PADDLE = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    import paddle
    HAS_PADDLE = True
import os

import triton
import triton.ops


@pytest.mark.interpreter
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [  #
    (2, 4, 512, 16),
    (2, 4, 512, 32),
    (2, 4, 512, 64),
    (2, 4, 512, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16] if HAS_TORCH else ['float16', 'float32'])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('seq_par', [True, False])
def test_op(Z, H, N_CTX, D_HEAD, dtype, causal, seq_par, device):
    if HAS_TORCH:
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            pytest.skip("Flash attention only supported for compute capability >= 80")
        if dtype == torch.bfloat16 and os.environ.get("TRITON_INTERPRET", "0") == "1":
            pytest.skip("Flash attention bfloat16 not supported in interpreter mode")
        torch.manual_seed(20)
        q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = 0.5
        dout = torch.randn_like(q)
        # reference implementation
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        # p = torch.exp(p)
        ref_out = torch.matmul(p, v)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
        # # triton implementation
        tri_out = triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None
        # compare
        atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_out), dim=0),
                                torch.nn.functional.normalize(torch.flatten(tri_out), dim=0), atol=atol, rtol=0)
        torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dv), dim=0),
                                torch.nn.functional.normalize(torch.flatten(tri_dv), dim=0), atol=atol, rtol=0)
        torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dk), dim=0),
                                torch.nn.functional.normalize(torch.flatten(tri_dk), dim=0), atol=atol, rtol=0)
        torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dq), dim=0),
                                torch.nn.functional.normalize(torch.flatten(tri_dq), dim=0), atol=atol, rtol=0)
    else:
        capability = paddle.device.cuda.get_device_capability()
        if capability[0] < 8:
            pytest.skip("Flash attention only supported for compute capability >= 80")
        paddle.seed(20)
        # 创建 QKV：先 normal 再 astype
        q = paddle.normal(mean=0.0, std=0.5, shape=[Z, H, N_CTX, D_HEAD]).astype(dtype)
        q.stop_gradient = False

        k = paddle.normal(mean=0.0, std=0.5, shape=[Z, H, N_CTX, D_HEAD]).astype(dtype)
        k.stop_gradient = False

        v = paddle.normal(mean=0.0, std=0.5, shape=[Z, H, N_CTX, D_HEAD]).astype(dtype)
        v.stop_gradient = False


        sm_scale = 0.5
        dout = paddle.randn(shape=q.shape, dtype=dtype) 

       
        M = paddle.tril(paddle.ones([N_CTX, N_CTX], dtype='float32'))  # 掩码通常用 float32
        k_t = paddle.transpose(k, perm=[0, 1, 3, 2])
        p = paddle.matmul(q, k_t) * sm_scale

        if causal:
            mask = M == 0
            mask = mask.unsqueeze(0).unsqueeze(0)  # -> [1, 1, N_CTX, N_CTX]
            p = paddle.where(mask, paddle.full([], float('-inf'), dtype=p.dtype), p)

        p = paddle.nn.functional.softmax(p.astype('float32'), axis=-1).astype(dtype)
        ref_out = paddle.matmul(p, v)
        ref_out.backward(dout)
        ref_dv = v.grad.clone()

        ref_dk = k.grad.clone()

        ref_dq = q.grad.clone()
        
        
        
        q = q.clone().detach()
        k = k.clone().detach()
        v = v.clone().detach()
        q.stop_gradient = False
        k.stop_gradient = False
        v.stop_gradient = False
        
        tri_out = triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
        tri_out.backward(dout)

        tri_dv = v.grad.clone()
        v.clear_gradient()

        tri_dk = k.grad.clone()
        k.clear_gradient()

        tri_dq = q.grad.clone()
        q.clear_gradient()

        atol = 1e-1 if dtype == paddle.bfloat16 else 1e-2

        def normalize(x):
            x_flat = paddle.flatten(x)
            return paddle.nn.functional.normalize(x_flat, axis=0)

        paddle.testing.assert_allclose(
            normalize(ref_out),
            normalize(tri_out),
            atol=atol,
            rtol=0,
            err_msg="Output mismatch"
        )

        paddle.testing.assert_allclose(
            normalize(ref_dv),
            normalize(tri_dv),
            atol=atol,
            rtol=0,
            err_msg="Grad V mismatch"
        )

        paddle.testing.assert_allclose(
            normalize(ref_dk),
            normalize(tri_dk),
            atol=atol,
            rtol=0,
            err_msg="Grad K mismatch"
        )

        paddle.testing.assert_allclose(
            normalize(ref_dq),
            normalize(tri_dq),
            atol=atol,
            rtol=0,
            err_msg="Grad Q mismatch"
        )


try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
dtype = torch.float16 if HAS_TORCH else "float16"
configs = [
    triton.testing.Benchmark(
        x_names=['N_CTX'], x_vals=[2**i for i in range(10, 14)], line_arg='provider',
        line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
        line_names=['Triton'] + (['Flash'] if HAS_FLASH else []), styles=[('red', '-'), ('blue', '-')], ylabel='ms',
        plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-{casual}-{seq_par}', args={
            'H': N_HEADS,
            'BATCH': BATCH,
            'D_HEAD': D_HEAD,
            'dtype': dtype,
            'mode': mode,
            'casual': casual,
            'seq_par': seq_par,
        }) for mode in ['fwd', 'bwd'] for casual in [True, False] for seq_par in [True, False]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, casual, seq_par, provider, dtype=torch.float16 if HAS_TORCH else "float16", device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    sm_scale = 1.3
    if HAS_TORCH:
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if provider == "triton":
            fn = lambda: triton.ops.attention(q, k, v, casual, sm_scale, seq_par)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == "flash":
            lengths = torch.full((BATCH, ), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1, ), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            fn = lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=casual)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
    else:
        q = paddle.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = paddle.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = paddle.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if provider == "triton":
            fn = lambda: triton.ops.attention(q, k, v, casual, sm_scale, seq_par)
            if mode == 'bwd':
                o = fn()
                do = paddle.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == "flash":
            lengths = paddle.full((BATCH, ), fill_value=N_CTX, device=device)
            cu_seqlens = paddle.zeros((BATCH + 1, ), device=device, dtype=paddle.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            fn = lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=casual)
            if mode == 'bwd':
                o = fn()
                do = paddle.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms


# only works on post-Ampere GPUs right now
# bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    test_op(Z = 2, H = 4, N_CTX = 512, D_HEAD = 16, dtype = 'float32', causal = True, seq_par = True, device = 'cuda')