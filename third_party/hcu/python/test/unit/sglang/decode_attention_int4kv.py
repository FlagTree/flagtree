"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py
import torch
import triton
import triton.language as tl


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit(do_not_specialize=["Req_to_tokens", "B_req_idx", "B_Start_Loc", "B_Seqlen", "stride_req_to_tokens_b", "att_stride_h"])
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    K_Scales_Buffer,  # New argument for K scales
    k_gamas,
    quant_group_size: tl.constexpr, # quant_group_size
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_scales_buf_kbs,  # New stride for K scales bs
    stride_scales_buf_kh,   # New stride for K scales head
    stride_gamas_buf_kh,  # New stride for K scales bs
    stride_gamas_buf_kd,   # New stride for K scales head
    att_stride_h,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    scales_dtype = K_Scales_Buffer.dtype.element_ty

    cur_kv_head = cur_head // kv_group_num

    offs_d_q = tl.arange(0, BLOCK_DMODEL * 2)
    offs_d_s = tl.arange(0, BLOCK_DMODEL * 2 // quant_group_size)
    offs_d_kv = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d_q
    off_gama = cur_head * stride_gamas_buf_kh + offs_d_q[None, :]
    gamas = tl.load(k_gamas + off_gama)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark).to(reduce_dtype)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d_kv[None, :]
        )
        k_int8_tmp = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d_kv[None, :] < Lk),
            other=0.0,
        )

        # split int8 into two int4 values
        k_int8_low = (k_int8_tmp & 0xF).to(tl.int8)
        k_int8_high = ((k_int8_tmp & 0xF0) >> 4).to(tl.int8)
        k_int8 = tl.interleave(k_int8_low, k_int8_high)

        # Load K scales and dequantize
        offs_scales_k = (
            k_loc[:, None] * stride_scales_buf_kbs
            + cur_kv_head * stride_scales_buf_kh
            + offs_d_s[None, :]
        )
        k_scales = tl.load(
            K_Scales_Buffer + offs_scales_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=1.0
        )
        k_zeros = tl.load(
            K_Scales_Buffer + offs_scales_k + BLOCK_DMODEL * 2 // quant_group_size, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )
        k_int8_reshape = k_int8.to(tl.float32).reshape(k_int8.shape[0], k_int8.shape[1] // quant_group_size, quant_group_size)
        k_tmp = (k_int8_reshape - k_zeros.to(tl.float32).reshape(k_zeros.shape[0], k_zeros.shape[1], 1)).to(scales_dtype) * k_scales.reshape(k_scales.shape[0], k_scales.shape[1], 1)  # Dequantize K
        k_gama = k_tmp.reshape(k_tmp.shape[0], k_tmp.shape[1] * k_tmp.shape[2]).to(reduce_dtype)
        k = k_gama * gamas

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit(do_not_specialize=["Req_to_tokens", "B_req_idx", "B_Start_Loc", "B_Seqlen", "stride_logic_h", "stride_req_to_token_b"])
def _fwd_kernel_stage2(
    logits,
    V_Buffer,
    V_Scales_Buffer,  # New argument for V scales
    quant_group_size: tl.constexpr, # quant_group_size
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_scales_buf_vbs,  # New stride for V scales bs
    stride_scales_buf_vh,   # New stride for V scales head
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    scales_dtype = V_Scales_Buffer.dtype.element_ty

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_DMODEL * 2)
    offs_d_s = tl.arange(0, BLOCK_DMODEL * 2 // quant_group_size)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL * 2], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens
            + cur_batch_req_idx * stride_req_to_token_b
            + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        qk = tl.load(
            logits
            + cur_head * stride_logic_h
            + (cur_batch_start_loc + start_n + offs_n),
            mask=start_n + offs_n < cur_batch_seq_len,
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v_int8_tmp = tl.load(
            v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv)
        )

        # split int8 into two int4 values
        v_int8_low = (v_int8_tmp & 0xF).to(tl.int8)
        v_int8_high = ((v_int8_tmp & 0xF0) >> 4).to(tl.int8)
        v_int8 = tl.interleave(v_int8_low, v_int8_high)

        # Load V scales and dequantize
        offs_scales_v = (
            v_index[:, None] * stride_scales_buf_vbs
            + cur_kv_head * stride_scales_buf_vh
            + offs_d_s[None, :]
        )
        mask_n = start_n + offs_n < cur_batch_seq_len
        v_scales = tl.load(
            V_Scales_Buffer + offs_scales_v, mask=mask_n[:, None], other=1.0
        )
        v_zeros = tl.load(
            V_Scales_Buffer + offs_scales_v + BLOCK_DMODEL * 2 // quant_group_size, mask=mask_n[:, None], other=0.0
        )
        v_int8_reshape = v_int8.to(tl.float32).reshape(v_int8.shape[0], v_int8.shape[1] // quant_group_size, quant_group_size)
        v_tmp = (v_int8_reshape - v_zeros.to(tl.float32).reshape(v_zeros.shape[0], v_zeros.shape[1], 1)).to(scales_dtype) * v_scales.reshape(v_scales.shape[0], v_scales.shape[1], 1)  # Dequantize V
        v = v_tmp.reshape(v_tmp.shape[0], v_tmp.shape[1] * v_tmp.shape[2])

        p = p.to(v.dtype)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d_q
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_d_q < Lv * 2))


def _decode_att_m_fwd(
    q,
    k_buffer,
    k_scales_buffer,
    k_gamas,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    quant_group_size,
    stage1_best_config,
):
    Lq, Lk = q.shape[-1], k_buffer.shape[-1]

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    BLOCK_N = stage1_best_config['BLOCK_N']
    BLOCK = stage1_best_config['BLOCK']
    num_stages = stage1_best_config['num_stages']
    num_warps = stage1_best_config['num_warps']

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        k_scales_buffer,
        k_gamas,
        quant_group_size,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_scales_buffer.stride(0),
        k_scales_buffer.stride(1),
        k_gamas.stride(0),
        k_gamas.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
    )


def _decode_softmax_reducev_fwd(
    logits,
    v_buffer,
    v_scales_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    quant_group_size,
    stage2_best_config,
):

    batch, head = b_seq_len.shape[0], logits.shape[0]

    BLOCK_N = stage2_best_config['BLOCK_N']
    num_stages = stage2_best_config['num_stages']
    num_warps = stage2_best_config['num_warps']

    grid = (batch, head, 1)
    kv_group_num = logits.shape[0] // v_buffer.shape[1]

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_kernel_stage2[grid](
        logits,
        v_buffer,
        v_scales_buffer,
        quant_group_size,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logits.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_scales_buffer.stride(0),
        v_scales_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        Lv=Lv,
    )


@triton.jit(do_not_specialize=["Req_to_tokens", "B_req_idx", "B_Start_Loc", "B_Seqlen", "stride_req_to_tokens_b", "att_stride_h"])
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    K_Scales_Buffer,  # New argument for K scales
    quant_group_size: tl.constexpr, # quant_group_size
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_scales_buf_kbs,  # New stride for K scales bs
    stride_scales_buf_kh,   # New stride for K scales head
    att_stride_h,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    scales_dtype = K_Scales_Buffer.dtype.element_ty

    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d_q = tl.arange(0, BLOCK_DMODEL * 2)
    offs_d_s = tl.arange(0, BLOCK_DMODEL * 2 // quant_group_size)
    offs_d_kv = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d_q[None, :]

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(
            Q + offs_q + start_mark, mask=(mask_h[:, None]) & (offs_d_q[None, :] < Lk * 2)
        ).to(reduce_dtype)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (
            k_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d_kv[:, None]
        )
        k_int8_tmp = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[None, :] < cur_batch_end_index) & (offs_d_kv[:, None] < Lk),
            other=0.0,
        )

        # split int8 into two int4 values
        k_int8_low = tl.trans((k_int8_tmp & 0xF).to(tl.int8))
        k_int8_high = tl.trans(((k_int8_tmp & 0xF0) >> 4).to(tl.int8))
        k_int8 = tl.trans(tl.interleave(k_int8_low, k_int8_high))

        # Load K scales and dequantize
        offs_scales_k = (
            k_loc[None, :] * stride_scales_buf_kbs
            + cur_kv_head * stride_scales_buf_kh
            + offs_d_s[:, None]
        )
        k_scales = tl.load(
            K_Scales_Buffer + offs_scales_k, mask=offs_n_new[None, :] < cur_batch_end_index, other=1.0
        )
        k_zeros = tl.load(
            K_Scales_Buffer + offs_scales_k + BLOCK_DMODEL * 2 // quant_group_size, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0
        )
        k_int8_reshape = k_int8.to(tl.float32).reshape(k_int8.shape[0] // quant_group_size, quant_group_size, k_int8.shape[1])
        k_tmp = (k_int8_reshape - k_zeros.to(tl.float32).reshape(k_zeros.shape[0], 1, k_zeros.shape[1])).to(scales_dtype) * k_scales.reshape(k_scales.shape[0], 1, k_scales.shape[1])  # Dequantize K
        k = k_tmp.reshape(k_tmp.shape[0] * k_tmp.shape[1], k_tmp.shape[2]).to(reduce_dtype)

        qk = tl.dot(q, k)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        offs_o = cur_head[:, None] * att_stride_h + (
            cur_batch_in_all_start_index + offs_n[None, :]
        )

        tl.store(
            Att_Out + offs_o,
            qk,
            mask=mask_h[:, None] & (offs_n_new[None, :] < cur_batch_end_index),
        )


@triton.jit(do_not_specialize=["Req_to_tokens", "B_req_idx", "B_Start_Loc", "B_Seqlen", "stride_logic_h", "stride_req_to_token_b"])
def _fwd_grouped_kernel_stage2(
    logits,
    V_Buffer,
    V_Scales_Buffer,  # New argument for K scales
    quant_group_size: tl.constexpr, # quant_group_size
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_scales_buf_vbs,  # New stride for V scales bs
    stride_scales_buf_vh,   # New stride for V scales head
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    scales_dtype = V_Scales_Buffer.dtype.element_ty

    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d_q = tl.arange(0, BLOCK_DMODEL * 2)
    offs_d_s = tl.arange(0, BLOCK_DMODEL * 2 // quant_group_size)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL * 2], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens
            + cur_batch_req_idx * stride_req_to_token_b
            + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        offs_qk = cur_head[:, None] * stride_logic_h + (
            cur_batch_start_loc + start_n + offs_n[None, :]
        )

        qk = tl.load(
            logits + offs_qk,
            mask=mask_h[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len),
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * old_scale + tl.sum(p, 1)
        v_int8_tmp = tl.load(
            v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv)
        )

        # split int8 into two int4 values
        v_int8_low = (v_int8_tmp & 0xF).to(tl.int8)
        v_int8_high = ((v_int8_tmp & 0xF0) >> 4).to(tl.int8)
        v_int8 = tl.interleave(v_int8_low, v_int8_high)

        # Load V scales and dequantize
        offs_scales_v = (
            v_index[:, None] * stride_scales_buf_vbs
            + cur_kv_head * stride_scales_buf_vh
            + offs_d_s[None, :]
        )
        mask_n = start_n + offs_n < cur_batch_seq_len
        v_scales = tl.load(
            V_Scales_Buffer + offs_scales_v, mask=mask_n[:, None], other=1.0
        )
        v_zeros = tl.load(
            V_Scales_Buffer + offs_scales_v + BLOCK_DMODEL * 2 // quant_group_size, mask=mask_n[:, None], other=0.0
        )
        v_int8_reshape = v_int8.to(tl.float32).reshape(v_int8.shape[0], v_int8.shape[1] // quant_group_size, quant_group_size)
        v_tmp = (v_int8_reshape - v_zeros.to(tl.float32).reshape(v_zeros.shape[0], v_zeros.shape[1], 1)).to(scales_dtype) * v_scales.reshape(v_scales.shape[0], v_scales.shape[1], 1)  # Dequantize V
        v = v_tmp.reshape(v_tmp.shape[0], v_tmp.shape[1] * v_tmp.shape[2])
        
        p = p.to(v.dtype)
        acc = acc * old_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    acc = acc / e_sum[:, None]
    off_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_d_q[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(mask_h[:, None]) & (offs_d_q[None, :] < Lv * 2))


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    k_scales_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    quant_group_size,
    stage1_best_config,
):
    Lq, Lk = q.shape[-1], k_buffer.shape[-1]

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DPE = 0

    batch, head_num = B_req_idx.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_N = stage1_best_config['BLOCK_N']
    BLOCK = stage1_best_config['BLOCK']
    num_stages = stage1_best_config['num_stages']
    num_warps = stage1_best_config['num_warps']

    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        triton.cdiv(max_len_in_batch, BLOCK),
    )

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        k_scales_buffer,
        quant_group_size,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_scales_buffer.stride(0),
        k_scales_buffer.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
    )


def _decode_grouped_softmax_reducev_fwd(
    logits,
    v_buffer,
    v_scales_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    quant_group_size,
    stage2_best_config,
):
   
    batch, head_num = b_seq_len.shape[0], logits.shape[0]
    kv_group_num = logits.shape[0] // v_buffer.shape[1]

    BLOCK_N = stage2_best_config['BLOCK_N']
    num_stages = stage2_best_config['num_stages']
    num_warps = stage2_best_config['num_warps']

    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), 1)

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_grouped_kernel_stage2[grid](
        logits,
        v_buffer,
        v_scales_buffer,
        quant_group_size,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logits.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_scales_buffer.stride(0),
        v_scales_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        Lv=Lv,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def decode_attention_fwd_int4kv(
    q,
    k_buffer,
    v_buffer,
    k_scales_buffer,
    v_scales_buffer,
    k_gamas,
    o,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    attn_logits,
    max_len_in_batch,
    sm_scale,
    k_quant_group_size,
    v_quant_group_size,
    best_config,
    logit_cap=0.0,
):
    if best_config['kernel_kind'] == "v1_2stages":
        # MHA
        _decode_att_m_fwd(
            q,
            k_buffer,
            k_scales_buffer,
            k_gamas,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            k_quant_group_size,
            best_config['best_config']['stage1'],
        )
        _decode_softmax_reducev_fwd(
            attn_logits,
            v_buffer,
            v_scales_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            v_quant_group_size,
            best_config['best_config']['stage2'],
        )
    elif best_config['kernel_kind'] == "v1_2stages_tc":
        # GQA/MQA/MLA
        _decode_grouped_att_m_fwd(
            q,
            k_buffer,
            k_scales_buffer,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
            k_quant_group_size,
            best_config['best_config']['stage1'],
        )
        _decode_grouped_softmax_reducev_fwd(
            attn_logits,
            v_buffer,
            v_scales_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            v_quant_group_size,
            best_config['best_config']['stage2'],
        )
    else:
        print("C4 attention only surpport kernel kind v1_2stages and v1_2stages_tc for now")
        assert False



@triton.jit
def _bwd_kernel_destindex_dequantize_int4_kv(
    Quantized,
    Scale,
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_g,
    stride_q_d,
    stride_s_bs,
    stride_s_h,
    stride_s_g,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index
    scales_dtype = Scale.dtype.element_ty

    # 加载量化数据
    q_data = tl.load(
        Quantized + cur_index * stride_q_bs + cur_head * stride_q_h + offs_g[:, None] * stride_q_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )

    # 分离 int8 的低 4 位（int4 数据 0）和高 4 位（int4 数据 1）
    low_4 = q_data & 0xF
    high_4 = (q_data >> 4) & 0xF

    src_data_0 = low_4.to(tl.int8)
    src_data_1 = high_4.to(tl.int8)

    # 加载反量化比例因子（scale）
    scale = tl.load(Scale + dest_index * stride_s_bs + cur_head * stride_s_h + offs_g, mask=offs_g < group_size)
    zero = tl.load(Scale + dest_index * stride_s_bs + cur_head * stride_s_h + offs_g + BLOCK_GROUP_NUM, mask=offs_g < group_size)

    # 反量化
    dequant_data_0 = (src_data_0.to(tl.float32) - zero.to(tl.float32)[:, None]).to(scales_dtype) * scale[:, None]
    dequant_data_1 = (src_data_1.to(tl.float32) - zero.to(tl.float32)[:, None]).to(scales_dtype) * scale[:, None]

    # 存储反量化的 float 数据
    o_ptrs_0 = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * 2
    o_ptrs_1 = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * 2 + 1

    tl.store(o_ptrs_0, dequant_data_0, mask=offs_g[:, None] < group_size)
    tl.store(o_ptrs_1, dequant_data_1, mask=offs_g[:, None] < group_size)


@torch.no_grad()
def destindex_dequantize_int4kv(Quantized, Scale, Out, quant_group_dim):
    bs = Out.shape[0]
    head_num = Quantized.shape[1]
    head_dim = Out.shape[2]

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy dequant kv"
    grid = (bs, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    Quantized = Quantized.view((Quantized.shape[0], Quantized.shape[1], group_size, group_dim // 2))
    Scale = Scale.view((Scale.shape[0], Scale.shape[1], group_size * 2))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim
    )  # Out 是 float16 类型，解压缩时需要两个 int4 恢复成 float16，所以 group_dim

    _bwd_kernel_destindex_dequantize_int4_kv[grid](
        Quantized,
        Scale,
        Out,
        Quantized.stride(0),
        Quantized.stride(1),
        Quantized.stride(2),
        Quantized.stride(3),
        Scale.stride(0),
        Scale.stride(1),
        Scale.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return

@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_kv_init(
    K,
    Out,
    Out_scale_zero,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)

    dest_index = cur_index

    offs_kv = cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2

    src_data_0 = tl.load(
        K + offs_kv,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    ).to(tl.float32)
    src_data_1 = tl.load(
        K + offs_kv + 1,
        mask=offs_g[:, None] < group_size,
        other=0.0,
    ).to(tl.float32)

    # 计算量化因子并量化数据
    min_data = tl.minimum(tl.min(src_data_0, axis=1), tl.min(src_data_1, axis=1))  
    max_data = tl.maximum(tl.max(src_data_0, axis=1), tl.max(src_data_1, axis=1))

    scale = (max_data - min_data) / 15.0
    zero_point = -1 * min_data / scale

    # 非对称量化
    q_src_data_0 = ((src_data_0 / scale[:, None]) + zero_point[:, None]).to(tl.int8)
    q_src_data_0 = tl.clamp(q_src_data_0.to(tl.float32), 0.0, 15.0).to(tl.int8)

    q_src_data_1 = ((src_data_1 / scale[:, None]) + zero_point[:, None]).to(tl.int8)
    q_src_data_1 = tl.clamp(q_src_data_1.to(tl.float32), 0.0, 15.0).to(tl.int8)

    # 合并为 int4 数据
    low_4 = q_src_data_0 & 0xF
    high_4 = (q_src_data_1 & 0xF) << 4
    out_data = low_4 | high_4

    scale = scale.to(Out_scale_zero.dtype.element_ty)
    zero_point = zero_point.to(Out_scale_zero.dtype.element_ty)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale_zero + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    oz_ptrs = Out_scale_zero + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g + BLOCK_GROUP_NUM
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, scale, mask=offs_g < group_size)
    tl.store(oz_ptrs, zero_point, mask=offs_g < group_size)


@torch.no_grad()
def destindex_copy_quantize_int4kv_init(K, Out, Out_scale, quant_group_dim):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (bs_seq, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim // 2
    )  # OUt 是 int8 类型， 两个int4组一个int8，所以 group_dim // 2

    _fwd_kernel_destindex_copy_quantize_int4_kv_init[grid](
        K,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return



@triton.jit
def _fwd_kernel_gama_int4kv_init(
    K,
    Gama,
    K_G,
    total_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_h,
    stride_o_d,
    alpha: tl.constexpr,
):
    cur_head = tl.program_id(0)

    offs_t = tl.arange(0, total_tokens)
    offs_d = tl.arange(0, head_dim)

    offs_k = offs_t[None, :] * stride_k_bs + cur_head * stride_k_h + offs_d[:, None]
    src_data = tl.load(
        K + offs_k,
        mask=offs_d[:, None] < head_dim,
        other=0.0,
    ).to(tl.float32)

    # 计算gama因子
    max_data = tl.max(tl.abs(src_data), axis=1, keep_dims=True)
    gama_data = tl.sqrt(max_data)
    out_data = src_data / gama_data

    og_ptrs = Gama + cur_head * stride_o_h + offs_d[:, None]
    o_ptrs = K_G + offs_k
    tl.store(og_ptrs, gama_data, offs_d[:, None] < head_dim)
    tl.store(o_ptrs, out_data, offs_d[:, None] < head_dim)
    


@torch.no_grad()
def gama_int4kv_init(K, Gama, K_G, alpha=0.5):
    total_tokens = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]

    grid = (head_num,)
    num_warps = 4

    _fwd_kernel_gama_int4kv_init[grid](
        K,
        Gama,
        K_G,
        total_tokens,
        head_dim,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Gama.stride(0),
        Gama.stride(1),
        alpha,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_degama_int4kv_init(
    K,
    Gama,
    K_G,
    total_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_h,
    stride_o_d,
    alpha: tl.constexpr,
):
    cur_head = tl.program_id(0)

    offs_t = tl.arange(0, total_tokens)
    offs_d = tl.arange(0, head_dim)

    offs_k = offs_t[None, :] * stride_k_bs + cur_head * stride_k_h + offs_d[:, None]
    offs_g = cur_head * stride_o_h + offs_d[:, None]

    src_data = tl.load(
        K_G + offs_k,
        mask=offs_d[:, None] < head_dim,
        other=0.0,
    ).to(tl.float32)
    gama_data = tl.load(
        Gama + offs_g,
        mask=offs_d[:, None] < head_dim,
        other=0.0,
    ).to(tl.float32)

    out_data = src_data * gama_data

    o_ptrs = K + offs_k
    tl.store(o_ptrs, out_data, offs_d[:, None] < head_dim)
    


@torch.no_grad()
def degama_int4kv_init(K_G, Gama, K, alpha=0.5):
    total_tokens = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]

    grid = (head_num,)
    num_warps = 4

    _fwd_kernel_degama_int4kv_init[grid](
        K,
        Gama,
        K_G,
        total_tokens,
        head_dim,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Gama.stride(0),
        Gama.stride(1),
        alpha,
        num_warps=num_warps,
        num_stages=1,
    )
    return
