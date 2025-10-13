import pytest
try:
    import paddle
except ImportError:
    pytest.skip("Paddle not installed — skipping tests.", allow_module_level=True)

"""
Fused Attention
===============
persistent warp specialized version of python/tutorials/06-fused-attention.py.
As of now, it only supports non-persistent warp specialized version of _fwd kernel.
"""

import triton
import triton.language as tl


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)], key=["Q", "K", "V"]
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (
        off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    off_k = (
        off_hz * stride_qh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    )
    off_v = (
        off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float("-inf"))
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        l_prev = l_curr
        m_prev = m_curr
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = (
        off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess(
    Out, DO, L, NewDO, Delta, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float("-inf"))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            ds = p * dp * sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


empty = paddle.empty(128, device="cuda")


class _attention(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = paddle.empty_like(q)
        grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1
        L = paddle.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.place, dtype=paddle.float32
        )
        m = paddle.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.place, dtype=paddle.float32
        )
        num_warps = 4 if Lk <= 64 else 8
        assert num_warps == 4
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            m,
            o,
            q.strides[0],
            q.strides[1],
            q.strides[2],
            q.strides[3],
            k.strides[0],
            k.strides[1],
            k.strides[2],
            k.strides[3],
            v.strides[0],
            v.strides[1],
            v.strides[2],
            v.strides[3],
            o.strides[0],
            o.strides[1],
            o.strides[2],
            o.strides[3],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q, k, v, o, l, m = ctx.saved_tensor()
        do = do.contiguous()
        dq = paddle.zeros_like(q, dtype=paddle.float32)
        dk = paddle.empty_like(k)
        dv = paddle.empty_like(v)
        do_scaled = paddle.empty_like(do)
        delta = paddle.empty_like(l)
        _bwd_preprocess[
            ctx.grid[0] * ctx.grid[1],
        ](o, do, l, do_scaled, delta, BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
        _bwd_kernel[ctx.grid[1],](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            l,
            m,
            delta,
            q.strides[0],
            q.strides[1],
            q.strides[2],
            q.strides[3],
            k.strides[0],
            k.strides[1],
            k.strides[2],
            k.strides[3],
            v.strides[0],
            v.strides[1],
            v.strides[2],
            v.strides[3],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(4, 48, 1024, 64)])
@pytest.mark.skipif(
    paddle.device.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9",
)
def test_op(Z, H, N_CTX, D_HEAD, dtype=paddle.float16):
    paddle.seed(20)
    out_0 = paddle.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.1, std=0.2
    )
    out_0.stop_gradient = not True
    q = out_0
    out_1 = paddle.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.4, std=0.2
    )
    out_1.stop_gradient = not True
    k = out_1
    out_2 = paddle.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(
        mean=0.3, std=0.2
    )
    out_2.stop_gradient = not True
    v = out_2
    sm_scale = 0.2
    dout = paddle.randn(shape=q.shape, dtype=q.dtype)
    M = paddle.tril(paddle.ones((N_CTX, N_CTX), device="cuda"))
    p = paddle.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            mask = (M == 0).unsqueeze(0).unsqueeze(0)  # [1,1,128,128]
            p = p.masked_fill(mask, float('-inf'))
    p = paddle.softmax(p.float(), dim=-1).half()
    ref_out = paddle.matmul(p, v)
    ref_out.backward(grad_tensor=dout)
    ref_dv = v.grad.clone()
    v.clear_gradient()
    ref_dk= k.grad.clone()
    k.clear_gradient()
    ref_dq= q.grad.clone()
    q.clear_gradient()
    tri_out = attention(q, k, v, sm_scale)
    tri_out.backward(grad_tensor=dout)
    tri_dv = v.grad.clone()
    v.clear_gradient()
    tri_dk = k.grad.clone()
    k.clear_gradient()
    tri_dq = q.grad.clone()
    q.clear_gradient()
    assert paddle.allclose(x=ref_out, y=tri_out, atol=0.01, rtol=0.).item()
    atol = 0.04 if is_hip_mi200() else 0.01
    assert paddle.allclose(x=ref_dv, y=tri_dv, atol=atol, rtol=0.).item()
    assert paddle.allclose(x=ref_dk, y=tri_dk, atol=0.01, rtol=0.).item()
    assert paddle.allclose(x=ref_dq.astype(tri_dq.dtype), y=tri_dq, atol=0.01, rtol=0.).item()


try:
    pass
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[(2**i) for i in range(10, 11)],
        line_arg="provider",
        line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton"] + (["Flash"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "dtype": paddle.float16,
            "mode": mode,
        },
    )
    for mode in ["fwd"]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=paddle.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 0
    rep = 1
    if provider == "triton":
        q = paddle.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        k = paddle.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        v = paddle.randn(
            (BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
        )
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = paddle.randn(shape=o.shape, dtype=o.dtype)
            fn = lambda: o.backward(grad_tensor=do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "flash":
        lengths = paddle.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = paddle.zeros((BATCH + 1,), device=device, dtype=paddle.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = paddle.randn(
            (BATCH * N_CTX, 3, H, D_HEAD),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: paddle.nn.functional.flash_attention.flash_attention(
            query=qkv, key=cu_seqlens, value=0.0, dropout=N_CTX, causal=True
        )[0]
        if mode == "bwd":
            o = fn()
            do = paddle.randn(shape=o.shape, dtype=o.dtype)
            fn = lambda: o.backward(grad_tensor=do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
