"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import torch_mlu

import triton
import triton.language as tl
from triton.backends.mlu.driver import BangDriver


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr, IS_DIVISIBLE: tl.constexpr):
    if CAUSAL:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(k, q)
        if CAUSAL and (start_n + BLOCK_N) >= start_m * BLOCK_M:
            mask = offs_m[None, :] >= (start_n + offs_n[:, None])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij[None, :]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 0) * qk_scale)
            qk = qk * qk_scale - m_ij[None, :]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 0)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        #qkv = tl.dot(tl.trans(p.to(tl.float16)), v)
        if fp8_v:
            qk_wram = p.to(tl.float8e5)
        else:
            qk_wram = p.to(tl.float16)
        qkv = tl.dot(v, qk_wram)
        # -- update output accumulator --
        acc = acc * alpha[None, :]
        acc += qkv
        # update m_i and l_i
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({}, pipeline_strategies=p, num_stages=s, num_warps=1)
        for p in [[""], ["reduce_delay"]]
        for s in [3, 5]
    ], key=['N_CTX', 'BLOCK_DMODEL', 'CAUSAL'])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              CAUSAL: tl.constexpr,  #
              IS_DIVISIBLE: tl.constexpr):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core

    for head_idx in range(head_begin, head_end):
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty != tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        #acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        CAUSAL, offs_m, offs_n, N_CTX,  #
                                        V.dtype.element_ty == tl.float8e5, IS_DIVISIBLE)
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[None, :]
        acc = tl.trans(acc)
        m_ptrs = M + off_hz * N_CTX + offs_m
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    N_CTX: tl.constexpr,
    N_HEAD: tl.constexpr,
    BATCH: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    if IS_DIVISIBLE:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    else:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :], mask=(off_m < N_CTX * N_HEAD * BATCH)[:, None],
                    other=0.).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :], mask=(off_m < N_CTX * N_HEAD * BATCH)[:, None],
                     other=0.).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    if IS_DIVISIBLE:
        tl.store(Delta + off_m, delta)
    else:
        tl.store(Delta + off_m, delta, mask=off_m < N_CTX * N_HEAD * BATCH)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': m, 'BLOCK_N': n}, pipeline_strategies=p, num_stages=s, num_warps=1)
        for p in [["force_bottleneck"]]
        for m in [128]
        for n in [64, 128] if m >= n for s in [6]
    ], key=['N_CTX', 'BLOCK_DMODEL', 'CAUSAL'], reset_to_zero=["DQ_ptr"])
@triton.heuristics({
    'IS_DIVISIBLE':
    lambda args: args['N_CTX'] % args['BLOCK_M'] == 0 and args['N_CTX'] % args['BLOCK_N'] == 0,
})
@triton.jit
def _bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    sm_scale,
    Out,
    DO_ptr,
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    L,
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
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
    Z,
    H,
    N_CTX: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core

    for head_idx in range(head_begin, head_end):
        start_n = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H

        qk_scale = sm_scale * 1.44269504
        # offset pointers for batch/head
        Q = Q_ptr + off_z * stride_qz + off_h * stride_qh
        K = K_ptr + off_z * stride_kz + off_h * stride_kh
        V = V_ptr + off_z * stride_vz + off_h * stride_vh
        DO = DO_ptr + off_z * stride_doz + off_h * stride_doh
        DQ = DQ_ptr + off_z * stride_dqz + off_h * stride_dqh
        DK = DK_ptr + off_z * stride_dkz + off_h * stride_dkh
        DV = DV_ptr + off_z * stride_dvz + off_h * stride_dvh

        if CAUSAL:
            lo = start_n * BLOCK_M
        else:
            lo = 0
        # initialize row/col offsets
        offs_q = lo + tl.arange(0, BLOCK_N)
        offs_kv = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_q[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_kv[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_kv[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_q[:, None] * stride_dom + offs_k[None, :] * stride_dok)
        dq_ptrs = DQ + (offs_q[:, None] * BLOCK_DMODEL + offs_k[None, :] * stride_dqk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        if IS_DIVISIBLE:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=(offs_kv < N_CTX)[:, None], other=0.)
            v = tl.load(v_ptrs, mask=(offs_kv < N_CTX)[:, None], other=0.)
        # loop over rows
        hi = tl.minimum(context_num * BLOCK_M, N_CTX)
        for start_n in range(lo, hi, BLOCK_N):
            curr_offs = start_n + offs_n
            # load q, k, v, do on-chip
            if IS_DIVISIBLE:
                q = tl.load(q_ptrs)
            else:
                q = tl.load(q_ptrs, mask=(curr_offs < N_CTX)[:, None], other=0.)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                if IS_DIVISIBLE:
                    qk = tl.where(curr_offs[None, :] >= (offs_kv[:, None]), float(0.), float("-inf"))
                else:
                    qk = tl.where((curr_offs[None, :] >= (offs_kv[:, None]))
                                  & (curr_offs[None, :] < N_CTX), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(k, tl.trans(q))
            qk *= qk_scale
            if IS_DIVISIBLE:
                l_i = tl.load(l_ptrs + curr_offs[None, :])
            else:
                l_i = tl.load(tl.broadcast_to(l_ptrs + curr_offs[None, :], BLOCK_M, BLOCK_N),
                              mask=(curr_offs < N_CTX)[None, :] & (offs_kv < N_CTX)[:, None], other=1e8)
            p = tl.math.exp2(qk - l_i)
            p_half = p.to(Q.dtype.element_ty)
            # compute dv
            if IS_DIVISIBLE:
                do = tl.load(do_ptrs)
            else:
                do = tl.load(do_ptrs, mask=(curr_offs < N_CTX)[:, None], other=0.)
            dv += tl.dot(p_half, do)
            # compute dp = dot(v, do)
            if IS_DIVISIBLE:
                Di = tl.load(D_ptrs + curr_offs[None, :])
            else:
                Di = tl.load(tl.broadcast_to(D_ptrs + curr_offs[None, :], BLOCK_M, BLOCK_N),
                             mask=(curr_offs < N_CTX)[None, :] & (offs_kv < N_CTX)[:, None], other=0.)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di
            dp += tl.dot(v, tl.trans(do))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            #ds_trans = tl.trans(ds)
            ds_half = ds.to(Q.dtype.element_ty)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds_half, q)
            # compute dq
            dq = tl.dot(tl.trans(ds_half), k)
            if IS_DIVISIBLE:
                tl.atomic_add(dq_ptrs, dq)
            else:
                tl.atomic_add(dq_ptrs, dq, mask=(curr_offs < N_CTX)[:, None])
            # increment pointers
            dq_ptrs += BLOCK_N * stride_dqm
            q_ptrs += BLOCK_N * stride_qm
            do_ptrs += BLOCK_N * stride_dom
        # write-back
        dv_ptrs = DV + (offs_kv[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
        dk_ptrs = DK + (offs_kv[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
        if IS_DIVISIBLE:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_kv < N_CTX)[:, None])
            tl.store(dk_ptrs, dk, mask=(offs_kv < N_CTX)[:, None])


empty = torch.empty(128, device="mlu")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 128
        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        N_CTX = q.shape[2]
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX,  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=HEAD_DIM_K,  #
            CAUSAL=causal,  #
            IS_DIVISIBLE=IS_DIVISIBLE,  #
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.BLOCK_DMODEL == 64:
            BLOCK_M, BLOCK_N = 128, 64
        else:
            assert ctx.BLOCK_DMODEL == 128
            BLOCK_M, BLOCK_N = 128, 64
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        BATCH, N_HEAD, N_CTX = q.shape[:3]

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True

        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
        _bwd_preprocess[(triton.cdiv(N_CTX, BLOCK_M) * N_HEAD * BATCH, )](
            o,
            do,
            delta,
            N_CTX,
            N_HEAD,
            BATCH,
            IS_DIVISIBLE,
            BLOCK_M=BLOCK_M,
            D_HEAD=ctx.BLOCK_DMODEL,
        )

        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
        _bwd_kernel[(grid)](q, k, v, ctx.sm_scale, o, do, dq, dk, dv, L, delta, q.stride(0), q.stride(1), q.stride(2),
                            q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1),
                            v.stride(2), v.stride(3), do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dk.stride(0), dk.stride(1),
                            dk.stride(2), dk.stride(3), dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                            q.shape[0], q.shape[1], q.shape[2], BLOCK_DMODEL=ctx.BLOCK_DMODEL, CAUSAL=ctx.causal,
                            bottleneck="simd")
        return dq, dk, dv, None, None


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(4, 48, 1024, 64), (4, 48, 1024, 128), (2, 32, 2800, 64)])
@pytest.mark.parametrize("causal", [True, False])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="mlu"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

ENABLE_FP8 = (BangDriver().get_current_target().arch // 100) > 5
TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        for D_HEAD in [64, 128]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 and ENABLE_FP8 else []) +
                    (["flash"] if HAS_FLASH else []),
                    line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 and ENABLE_FP8 else []) +
                    (["Flash-2"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, device="mlu"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 50
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="mlu", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="mlu", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="mlu", requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.cpu().to(torch.float8_e5m2).view(torch.int8).to(device).view(torch.float8_e5m2)
            k = k.cpu().to(torch.float8_e5m2).view(torch.int8).to(device).view(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2)
            v = v.cpu().to(torch.float8_e5m2).view(torch.int8).to(device).view(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    bench_flash_attention.run(save_path=".", print_data=True)
