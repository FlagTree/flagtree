"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import torch

from triton import cdiv, jit
from triton import language as tl


@jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    CAUSAL: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, IS_DIVISIBLE: tl.constexpr):
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
        qkv = tl.dot(v, p.to(v.dtype))
        # -- update output accumulator --
        acc = acc * alpha[None, :]
        acc += qkv
        # update m_i and l_i
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@jit
def _fwd_kernel(Q, K, V, sm_scale, L, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
                stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H,
                N_CTX, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                CAUSAL: tl.constexpr,  #
                IS_DIVISIBLE: tl.constexpr):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = tl.cdiv(N_CTX, BLOCK_M)
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
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
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
                                        IS_DIVISIBLE)
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[None, :]
        acc = tl.trans(acc)
        m_ptrs = L + off_hz * N_CTX + offs_m
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


@jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@jit
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
    Z,
    H,
    N_CTX: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = tl.cdiv(N_CTX, BLOCK_M)
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
        K = K_ptr + off_z * stride_qz + off_h * stride_qh
        V = V_ptr + off_z * stride_qz + off_h * stride_qh
        DO = DO_ptr + off_z * stride_qz + off_h * stride_qh
        DQ = DQ_ptr + off_z * stride_qz + off_h * stride_qh
        DK = DK_ptr + off_z * stride_qz + off_h * stride_qh
        DV = DV_ptr + off_z * stride_qz + off_h * stride_qh

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
        do_ptrs = DO + (offs_q[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_q[:, None] * BLOCK_DMODEL + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_n in range(lo, context_num * BLOCK_M, BLOCK_N):
            curr_offs = start_n + offs_n
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(curr_offs[None, :] >= (offs_kv[:, None]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(k, tl.trans(q))
            qk *= qk_scale
            l_i = tl.load(l_ptrs + curr_offs)
            p = tl.math.exp2(qk - l_i[None, :])
            p_half = p.to(Q.dtype.element_ty)
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(p_half, do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + curr_offs)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[None, :]
            dp += tl.dot(v, tl.trans(do))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            #ds_trans = tl.trans(ds)
            ds_half = ds.to(Q.dtype.element_ty)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds_half, q)
            # compute dq
            dq = tl.dot(tl.trans(ds_half), k)
            tl.atomic_add(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_N * stride_qm
            q_ptrs += BLOCK_N * stride_qm
            do_ptrs += BLOCK_N * stride_qm
        # write-back
        dv_ptrs = DV + (offs_kv[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_kv[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        BLOCK_M = 128
        BLOCK_N = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert q.dtype == torch.float16, "only for f16 test now"
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        assert q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1
        o = torch.empty_like(q)
        num_warps = 1
        num_stages = 4
        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        IS_DIVISIBLE = False
        N_CTX = q.shape[2]
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True
        _fwd_kernel[grid](q, k, v, sm_scale, L, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0),
                          k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                          o.stride(0), o.stride(1), o.stride(2), o.stride(3), q.shape[0], q.shape[1], q.shape[2],
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk, CAUSAL=causal, IS_DIVISIBLE=IS_DIVISIBLE,
                          num_warps=num_warps, num_stages=num_stages)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.BLOCK_DMODEL == 64:
            BLOCK_M, BLOCK_N = 128, 128
        elif ctx.BLOCK_DMODEL == 128:
            BLOCK_M, BLOCK_N = 128, 64
        else:
            BLOCK_M, BLOCK_N = 64, 64
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        assert N_CTX % BLOCK_M == 0
        _bwd_preprocess[(N_CTX // BLOCK_M * N_HEAD * BATCH, )](
            o,
            do,
            delta,
            BLOCK_M=BLOCK_M,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)
        _bwd_kernel[(grid)](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do,
            dq,
            dk,
            dv,
            L,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CAUSAL=ctx.causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=1,
            num_stages=0,
        )

        return dq, dk, dv, None, None, None


attention = _attention.apply
