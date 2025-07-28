import torch
import triton
from triton import language as tl


@triton.jit
def _softmax_qk(max_logits, exp_sums, qk):
    qk_max = tl.maximum(max_logits, tl.max(qk, 1))
    alpha = tl.exp(max_logits - qk_max)
    soft_qk = tl.exp(qk - qk_max[:, None])
    exp_sums = exp_sums * alpha + tl.sum(soft_qk, 1)
    return qk_max, exp_sums, alpha, soft_qk


@triton.jit
def _softmax_kq(max_logits, exp_sums, qk):
    qk_max = tl.maximum(max_logits, tl.max(qk, 0))
    alpha = tl.exp(max_logits - qk_max)
    soft_qk = tl.exp(qk - qk_max[None, :])
    exp_sums = exp_sums * alpha + tl.sum(soft_qk, 0)
    return qk_max, exp_sums, alpha, soft_qk


@triton.jit
def _attn_inner_qkv(
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    seq_q,
    seq_kv,
    start_m,
    offset_m,
    alibi_slope,
    HAS_CAUSAL: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
):
    max_logits = tl.full([BLOCK_M * GQA_SHARED_HEADS], float("-inf"), dtype=tl.float32)
    exp_sums = tl.zeros([BLOCK_M * GQA_SHARED_HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * GQA_SHARED_HEADS, BLOCK_DHEAD], dtype=tl.float32)

    end_n = (start_m + 1) * BLOCK_M + seq_kv - seq_q if HAS_CAUSAL else seq_kv

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offset_n = start_n + tl.arange(0, BLOCK_N)

        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        if HAS_ALIBI:
            alibi_dist = offset_n - seq_kv + 1
            alibi = (alibi_slope[:, None] * alibi_dist[None, :]).reshape(GQA_SHARED_HEADS, 1, BLOCK_N)
            qk = qk.reshape(GQA_SHARED_HEADS, BLOCK_M, BLOCK_N) + alibi
            qk = qk.reshape(GQA_SHARED_HEADS * BLOCK_M, BLOCK_N)

        if HAS_CAUSAL and (start_n + BLOCK_N) >= (start_m * BLOCK_M + seq_kv - seq_q):
            causal_mask = offset_n[None, :] <= (offset_m + seq_kv - seq_q)
            qk = qk + tl.where(causal_mask, 0, -1.0e6)

        mask = True if IS_DIVISIBLE else (offset_m < seq_q * GQA_SHARED_HEADS) and (offset_n[None, :] < end_n)
        qk = qk + tl.where(mask, 0, -1.0e6)

        max_logits, exp_sums, alpha, soft_qk = _softmax_qk(max_logits, exp_sums, qk)

        qkv = tl.dot(soft_qk.to(v.dtype), v)
        acc = acc * alpha[:, None] + qkv

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, exp_sums


@triton.jit
def _attn_inner_vkq(
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    seq_q,
    seq_kv,
    start_m,
    offset_m,
    alibi_slope,
    HAS_CAUSAL: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
):
    max_logits = tl.full([BLOCK_M * GQA_SHARED_HEADS], float("-inf"), dtype=tl.float32)
    exp_sums = tl.zeros([BLOCK_M * GQA_SHARED_HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DHEAD, BLOCK_M * GQA_SHARED_HEADS], dtype=tl.float32)

    offset_m = offset_m.trans()

    end_n = (start_m + 1) * BLOCK_M + seq_kv - seq_q if HAS_CAUSAL else seq_kv

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offset_n = start_n + tl.arange(0, BLOCK_N)

        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(k, tl.trans(q))
        qk = qk * qk_scale

        if HAS_ALIBI:
            alibi_dist = offset_n - seq_kv + 1
            alibi = (alibi_slope[None, :] * alibi_dist[:, None]).reshape(BLOCK_N, GQA_SHARED_HEADS, 1)
            qk = qk.reshape(BLOCK_N, GQA_SHARED_HEADS, BLOCK_M) + alibi
            qk = qk.reshape(BLOCK_N, GQA_SHARED_HEADS * BLOCK_M)

        if HAS_CAUSAL and (start_n + BLOCK_N) >= (start_m * BLOCK_M + seq_kv - seq_q):
            causal_mask = offset_n[:, None] <= (offset_m + seq_kv - seq_q)
            qk = qk + tl.where(causal_mask, 0, -1.0e6)

        mask = True if IS_DIVISIBLE else (offset_m < seq_q * GQA_SHARED_HEADS) and (offset_n[:, None] < end_n)
        qk = qk + tl.where(mask, 0, -1.0e6)

        max_logits, exp_sums, alpha, soft_qk = _softmax_kq(max_logits, exp_sums, qk)

        qkv = tl.dot(tl.trans(v), soft_qk.to(v.type))
        acc = acc * alpha[None, :] + qkv

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return tl.trans(acc), exp_sums


@triton.jit
def _attn_inner(
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    seq_q,
    seq_kv,
    start_m,
    offset_m,
    alibi_slope,
    HAS_CAUSAL: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DHEAD: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    if COMPUTE_MODE == 0:
        return _attn_inner_qkv(q, K_block_ptr, V_block_ptr, qk_scale, seq_q, seq_kv, start_m, offset_m, alibi_slope,
                               HAS_CAUSAL, IS_DIVISIBLE, HAS_ALIBI, GQA_SHARED_HEADS, BLOCK_M, BLOCK_N, BLOCK_DHEAD)
    if COMPUTE_MODE == 1:
        return _attn_inner_vkq(q, K_block_ptr, V_block_ptr, qk_scale, seq_q, seq_kv, start_m, offset_m, alibi_slope,
                               HAS_CAUSAL, IS_DIVISIBLE, HAS_ALIBI, GQA_SHARED_HEADS, BLOCK_M, BLOCK_N, BLOCK_DHEAD)


@triton.jit
def _tile_range(
    bs,
    num_q_heads,
    num_kv_heads,
    seq_q,
    BLOCK_M: tl.constexpr,
    LOAD_MODE: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    task_id = tl.program_id(0)
    num_task = tl.num_programs(0)

    num_m = (seq_q + BLOCK_M - 1) // BLOCK_M
    num_heads = num_kv_heads if TILE_MODE == 0 else num_q_heads
    num_group = bs * num_heads * num_m

    if LOAD_MODE == 0:
        begin = task_id
        end = num_group
        step = num_task

    if LOAD_MODE == 1:
        group_per_task = num_group // num_task
        remain_group = num_group - group_per_task * num_task
        group_per_task += 1
        begin = task_id * group_per_task
        if task_id >= remain_group:
            group_per_task -= 1
            begin = task_id * group_per_task + remain_group
        end = begin + group_per_task
        step = 1

    return begin, end, step


@triton.jit
def _attn_kernel_split_k_head(
    Q,
    K,
    V,
    Out,
    Ctx_lens,
    Ctx_indexes,
    Alibi_slope,
    qk_scale,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_a0,
    stride_a1,
    bs,
    num_q_heads,
    num_kv_heads,
    seq_q,
    seq_kv,
    HAS_CAUSAL: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    # tuning config
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    LOAD_MODE: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    num_m = (seq_q + BLOCK_M - 1) // BLOCK_M
    begin, end, step = _tile_range(bs, num_q_heads, num_kv_heads, seq_q, BLOCK_M, LOAD_MODE, TILE_MODE)

    for group_id in range(begin, end, step):
        bs_id = group_id // (num_kv_heads * num_m)
        head_id = group_id % (num_kv_heads * num_m) // num_m
        start_m = group_id % num_m

        ctx_len = tl.load(Ctx_lens + bs_id) if Ctx_lens is not None else seq_kv
        ctx_index = tl.load(Ctx_indexes + bs_id) if Ctx_indexes is not None else bs_id

        Q_block_ptr = tl.make_block_ptr(
            base=Q + bs_id * stride_q0 + head_id * GQA_SHARED_HEADS * stride_q1,
            shape=(GQA_SHARED_HEADS, seq_q, D_HEAD),
            strides=(stride_q1, stride_q2, stride_q3),
            offsets=(0, start_m * BLOCK_M, 0),
            block_shape=(GQA_SHARED_HEADS, BLOCK_M, D_HEAD),
            order=(2, 1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + ctx_index * stride_k0 + head_id * stride_k1,
            shape=(ctx_len, D_HEAD),
            strides=(stride_k2, stride_k3),
            offsets=(0, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + ctx_index * stride_v0 + head_id * stride_v1,
            shape=(ctx_len, D_HEAD),
            strides=(stride_v2, stride_v3),
            offsets=(0, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0),
        )
        Out_block_ptr = tl.make_block_ptr(
            base=Out + bs_id * stride_o0 + head_id * GQA_SHARED_HEADS * stride_o1,
            shape=(GQA_SHARED_HEADS, seq_q, D_HEAD),
            strides=(stride_o1, stride_o2, stride_o3),
            offsets=(0, start_m * BLOCK_M, 0),
            block_shape=(GQA_SHARED_HEADS, BLOCK_M, D_HEAD),
            order=(2, 1, 0),
        )

        alibi_slope = None
        if HAS_ALIBI:
            Alibi_slope_block_ptr = tl.make_block_ptr(
                base=Alibi_slope + bs_id * stride_a0 + head_id * GQA_SHARED_HEADS * stride_a1,
                shape=(GQA_SHARED_HEADS, ),
                strides=(stride_a1, ),
                offsets=(0, ),
                block_shape=(GQA_SHARED_HEADS, ),
                order=(0, ),
            )
            alibi_slope = tl.load(Alibi_slope_block_ptr)

        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
        q = q.reshape(GQA_SHARED_HEADS * BLOCK_M, D_HEAD)

        offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        if GQA_SHARED_HEADS == 1:
            offset_m = offset_m.expand_dims(1)
            offset_m = offset_m.broadcast_to(BLOCK_M, BLOCK_N)
        else:
            offset_m = offset_m.reshape(1, BLOCK_M, 1)
            offset_m = offset_m.broadcast_to(GQA_SHARED_HEADS, BLOCK_M, BLOCK_N)
            offset_m = offset_m.reshape(GQA_SHARED_HEADS * BLOCK_M, BLOCK_N)

        acc, exp_sums = _attn_inner(q, K_block_ptr, V_block_ptr, qk_scale, seq_q, ctx_len, start_m, offset_m,
                                    alibi_slope, HAS_CAUSAL, IS_DIVISIBLE, HAS_ALIBI, GQA_SHARED_HEADS, BLOCK_M,
                                    BLOCK_N, D_HEAD, COMPUTE_MODE)

        acc = acc / (exp_sums[:, None] + 1e-6)
        acc = acc.reshape(GQA_SHARED_HEADS, BLOCK_M, D_HEAD)
        if IS_DIVISIBLE:
            tl.store(Out_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(Out_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1, 2))


@triton.jit
def _attn_kernel_split_q_head(
    Q,
    K,
    V,
    Out,
    Ctx_lens,
    Ctx_indexes,
    Alibi_slope,
    qk_scale,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_a0,
    stride_a1,
    bs,
    num_q_heads,
    num_kv_heads,
    seq_q,
    seq_kv,
    HAS_CAUSAL: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    # tuning config
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    LOAD_MODE: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    num_m = (seq_q + BLOCK_M - 1) // BLOCK_M
    begin, end, step = _tile_range(bs, num_q_heads, num_kv_heads, seq_q, BLOCK_M, LOAD_MODE, TILE_MODE)

    for group_id in range(begin, end, step):
        bs_id = group_id // (num_q_heads * num_m)
        head_id = group_id % (num_q_heads * num_m) // num_m
        start_m = group_id % num_m

        ctx_len = tl.load(Ctx_lens + bs_id) if Ctx_lens is not None else seq_kv
        ctx_index = tl.load(Ctx_indexes + bs_id) if Ctx_indexes is not None else bs_id

        Q_block_ptr = tl.make_block_ptr(
            base=Q + bs_id * stride_q0 + head_id * GQA_SHARED_HEADS * stride_q1,
            shape=(GQA_SHARED_HEADS, seq_q, D_HEAD),
            strides=(stride_q1, stride_q2, stride_q3),
            offsets=(0, start_m * BLOCK_M, 0),
            block_shape=(GQA_SHARED_HEADS, BLOCK_M, D_HEAD),
            order=(2, 1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + ctx_index * stride_k0,
            shape=(ctx_len, D_HEAD),
            strides=(stride_k2, stride_k3),
            offsets=(0, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + ctx_index * stride_v0,
            shape=(ctx_len, D_HEAD),
            strides=(stride_v2, stride_v3),
            offsets=(0, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0),
        )
        Out_block_ptr = tl.make_block_ptr(
            base=Out + bs_id * stride_o0 + head_id * GQA_SHARED_HEADS * stride_o1,
            shape=(GQA_SHARED_HEADS, seq_q, D_HEAD),
            strides=(stride_o1, stride_o2, stride_o3),
            offsets=(0, start_m * BLOCK_M, 0),
            block_shape=(GQA_SHARED_HEADS, BLOCK_M, D_HEAD),
            order=(2, 1, 0),
        )

        alibi_slope = None
        if HAS_ALIBI:
            Alibi_slope_block_ptr = tl.make_block_ptr(
                base=Alibi_slope + bs_id * stride_a0 + head_id * GQA_SHARED_HEADS * stride_a1,
                shape=(GQA_SHARED_HEADS, ),
                strides=(stride_a1, ),
                offsets=(0, ),
                block_shape=(GQA_SHARED_HEADS, ),
                order=(0, ),
            )
            alibi_slope = tl.load(Alibi_slope_block_ptr)

        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
        q = q.reshape(GQA_SHARED_HEADS * BLOCK_M, D_HEAD)

        offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        if GQA_SHARED_HEADS == 1:
            offset_m = offset_m.expand_dims(1)
            offset_m = offset_m.broadcast_to(BLOCK_M, BLOCK_N)
        else:
            offset_m = offset_m.reshape(1, BLOCK_M, 1)
            offset_m = offset_m.broadcast_to(GQA_SHARED_HEADS, BLOCK_M, BLOCK_N)
            offset_m = offset_m.reshape(GQA_SHARED_HEADS * BLOCK_M, BLOCK_N)

        acc, exp_sums = _attn_inner(q, K_block_ptr, V_block_ptr, qk_scale, seq_q, ctx_len, start_m, offset_m,
                                    alibi_slope, HAS_CAUSAL, IS_DIVISIBLE, HAS_ALIBI, GQA_SHARED_HEADS, BLOCK_M,
                                    BLOCK_N, D_HEAD, COMPUTE_MODE)

        acc = acc / (exp_sums[:, None] + 1e-6)
        acc = acc.reshape(GQA_SHARED_HEADS, BLOCK_M, D_HEAD)
        if IS_DIVISIBLE:
            tl.store(Out_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(Out_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1, 2))


def config_prune(configs, named_args, **kwargs):
    seq_q = named_args["seq_q"]
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, COMPUTE_MODE = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["COMPUTE_MODE"],
        )
        if seq_q <= 32:
            if BLOCK_M == 1 and BLOCK_N == 384 and COMPUTE_MODE == 0:
                pruned_configs.append(config)
            else:
                continue
        else:
            if COMPUTE_MODE == 1 and BLOCK_N <= 256:
                pruned_configs.append(config)
            else:
                continue
    return pruned_configs


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": m,
                "BLOCK_N": n,
                "COMPUTE_MODE": compute,
                "LOAD_MODE": load,
            },
            num_warps=1,
            num_stages=s,
        ) for m in [1, 128] for n in [128, 256, 384] for compute in [0, 1] for load in [0, 1] for s in [3]
    ],
    key=["seq_q", "seq_kv"],
    prune_configs_by={
        "early_config_prune": config_prune,
    },
)
@triton.heuristics({
    "IS_DIVISIBLE":
    lambda meta: False if meta["Ctx_lens"] is not None else
    ((meta["seq_q"] % meta["BLOCK_M"] == 0) and (meta["seq_kv"] % meta["BLOCK_M"] == 0) and
     (meta["seq_kv"] % meta["BLOCK_N"] == 0)),
    "HAS_ALIBI":
    lambda meta: True if meta["Alibi_slope"] is not None else False,
    "TILE_MODE":
    lambda meta: 0 if meta["num_kv_heads"] != 1 else 1,
})
@triton.jit
def _attn_kernel(
    Q,
    K,
    V,
    Out,
    Ctx_lens,
    Ctx_indexes,
    Alibi_slope,
    qk_scale,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_a0,
    stride_a1,
    bs,
    num_q_heads,
    num_kv_heads,
    seq_q,
    seq_kv,
    HAS_CAUSAL: tl.constexpr,
    GQA_SHARED_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    # tuning config
    IS_DIVISIBLE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    LOAD_MODE: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    if TILE_MODE == 0:
        _attn_kernel_split_k_head(Q, K, V, Out, Ctx_lens, Ctx_indexes, Alibi_slope, qk_scale, stride_q0, stride_q1,
                                  stride_q2, stride_q3, stride_k0, stride_k1, stride_k2, stride_k3, stride_v0,
                                  stride_v1, stride_v2, stride_v3, stride_o0, stride_o1, stride_o2, stride_o3,
                                  stride_a0, stride_a1, bs, num_q_heads, num_kv_heads, seq_q, seq_kv, HAS_CAUSAL,
                                  GQA_SHARED_HEADS, D_HEAD, IS_DIVISIBLE, HAS_ALIBI, BLOCK_M, BLOCK_N, COMPUTE_MODE,
                                  LOAD_MODE, TILE_MODE)
    if TILE_MODE == 1:
        _attn_kernel_split_q_head(Q, K, V, Out, Ctx_lens, Ctx_indexes, Alibi_slope, qk_scale, stride_q0, stride_q1,
                                  stride_q2, stride_q3, stride_k0, stride_k1, stride_k2, stride_k3, stride_v0,
                                  stride_v1, stride_v2, stride_v3, stride_o0, stride_o1, stride_o2, stride_o3,
                                  stride_a0, stride_a1, bs, num_q_heads, num_kv_heads, seq_q, seq_kv, HAS_CAUSAL, 1,
                                  D_HEAD, IS_DIVISIBLE, HAS_ALIBI, BLOCK_M, BLOCK_N, COMPUTE_MODE, LOAD_MODE, TILE_MODE)


def decoder_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float,
                      context_lens: torch.Tensor = None, context_indexes: torch.Tensor = None,
                      alibi_slope: torch.Tensor = None, has_causal: bool = True) -> torch.Tensor:
    """Implementation of decoder attention. Support multi head attention, multi
    query attention, group query attention and flash attention.

    :math:`(\text{out} = \text{Softmax}\(\text{q} \times \text{k}^T \times {sm_scale}\) \times \text{v})`

    Note:
        bs: Batch size, also known as `Z`.
        num_q_heads: Number of query heads, also known as `H_q`.
        num_kv_heads: Number of key or value heads, also known as `H_kv`.
        seq_q: Number of queries per head, also known as `M`.
        seq_kv: Number of keys per head, also known as `N`.
        d_head: Dimension of model, also known as `D_model` or `D`.

    Args:
        q: Query tensor of shape (bs, num_q_heads, seq_q, d_head). Must be one
           of the following types: `float16`, `float32`, `bfloat16`.
        k: Key tensor of shape (bs, num_kv_heads, seq_kv, d_head). Must be one
           of the following types: `float16`, `float32`, `bfloat16`.
        v: Value tensor of shape (bs, num_kv_heads, seq_kv, d_head). Must be one
           of the following types: `float16`, `float32`, `bfloat16`.
        sm_scale: Scaling factor.

    Keyword args:
        context_lens: Context lens tensor of shape (bs, ). Default is `None`.
                      If context_lens is not None, the number of keys per head
                      for each batch is `context_lens[bs_id]`.
        context_indexes: Context indexes tensor of shape (bs, ). Default is `None`.
                         If context_indexes is not None, the start index of `seq_kv`
                         for each batch is `context_indexes[bs_id]`.
        alibi_slope: Alibi slope tensor of shape (bs, num_q_heads). Default is `None`.
        has_causal: Default is `True`.

    Returns:
        Out tensor of shape (bs, num_q_heads, seq_q, d_head).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    stride_a0 = 0
    stride_a1 = 0
    if alibi_slope is not None:
        alibi_slope = alibi_slope.contiguous()
        stride_a0 = alibi_slope.stride(0)
        stride_a1 = alibi_slope.stride(1)

    bs, num_q_heads, seq_q, d_head = q.shape
    _, num_kv_heads, seq_kv, _ = k.shape

    assert q.shape[0] == k.shape[0]
    assert q.shape[3] == k.shape[3]

    assert num_q_heads % num_kv_heads == 0
    GQA_SHARED_HEADS = num_q_heads // num_kv_heads

    HAS_CAUSAL = seq_q > 1 and has_causal

    processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
    grid = lambda meta: (min(bs * num_kv_heads * ((seq_q + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]), processor_count), )
    _attn_kernel[grid](q, k, v, out, context_lens, context_indexes, alibi_slope, sm_scale, q.stride(0), q.stride(1),
                       q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0),
                       v.stride(1), v.stride(2), v.stride(3), out.stride(0), out.stride(1), out.stride(2),
                       out.stride(3), stride_a0, stride_a1, bs, num_q_heads, num_kv_heads, seq_q, seq_kv, HAS_CAUSAL,
                       GQA_SHARED_HEADS, d_head)
    return out
