import torch
import triton
import triton.language as tl


@triton.jit
def single_query_cached_kv_mlu(
    output_ptr,  # [num_seqs, num_heads, head_size]
    q_ptr,  # [num_seqs, num_heads, head_size]
    k_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_size]
    v_cache_ptr,  # [num_blocks, num_kv_heads, block_size, head_size]
    head_mapping_ptr,  # [num_heads]
    scale,
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    max_num_blocks_per_seq,
    alibi_slops_ptr,  # [num_heads]
    head_size,
    output_stride_seq,
    q_stride_seq,
    kv_stride_block,
    kv_stride_head,
    block_size: tl.constexpr,
    pad_head_size: tl.constexpr,
    blocks_per_group: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    kv_head_id = tl.load(head_mapping_ptr + pid_head)
    context_len = tl.load(context_lens_ptr + pid_seq)

    block_tables_ptr += pid_seq * max_num_blocks_per_seq
    headdim_offset = tl.arange(0, pad_head_size)

    q_ptr = q_ptr + pid_seq * q_stride_seq + pid_head * head_size
    k_cache_ptr = k_cache_ptr + kv_head_id * kv_stride_head
    v_cache_ptr = v_cache_ptr + kv_head_id * kv_stride_head
    output_ptr = output_ptr + pid_seq * output_stride_seq + pid_head * head_size

    q = tl.load(q_ptr + headdim_offset[None, :], mask=(headdim_offset[None, :] < head_size), other=0.0)
    qk_max, qk_sum = float('-inf'), 0.0

    group_block_size = blocks_per_group * block_size
    num_groups = (context_len + group_block_size - 1) // group_block_size
    qkv = tl.zeros(shape=(pad_head_size, ), dtype=tl.float32)
    offsets = tl.arange(0, blocks_per_group * block_size)
    block_offset = offsets % block_size
    group_offsets = offsets // block_size

    for group_id in range(0, num_groups):
        total_blocks = min(context_len - group_id * group_block_size, group_block_size)
        block_ids = tl.load(block_tables_ptr + group_id * blocks_per_group + group_offsets, mask=offsets < total_blocks,
                            other=0)

        k_begin = k_cache_ptr + block_ids[:, None] * kv_stride_block
        k = tl.load(k_begin + block_offset[:, None] * head_size + headdim_offset[None, :],
                    mask=(offsets[:, None] < total_blocks) & (headdim_offset[None, :] < head_size), other=0.0)

        qk = tl.view(tl.dot(q, tl.trans(k)), [k.shape[0]]) * scale
        qk = tl.where(offsets < total_blocks, qk, float('-inf'))
        max_val = tl.maximum(tl.max(qk, 0), qk_max)
        old_scale = tl.exp(qk_max - max_val)
        p = tl.exp(qk - max_val)
        qk_sum = qk_sum * old_scale + tl.sum(p, 0)
        qk_max = max_val

        v_begin = v_cache_ptr + block_ids[:, None] * kv_stride_block
        v = tl.load(v_begin + block_offset[:, None] * head_size + headdim_offset[None, :],
                    mask=(offsets[:, None] < total_blocks) & (headdim_offset[None, :] < head_size), other=0.0)

        qkv = qkv * old_scale + tl.view(tl.dot(p[None, :].to(tl.float16), v), [v.shape[1]])

    qkv = qkv / qk_sum
    tl.store(output_ptr + headdim_offset, qkv, mask=headdim_offset < head_size)


def single_query_cached_kv_attention(output, query, key_cache, value_cache, head_mapping, scale, block_tables,
                                     context_lens, block_size, block_size_per_group, max_context_len):  # ALiBi slopes.
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    grid = lambda meta: (num_seqs, num_heads)

    output_stride_seq = output.stride(0)
    q_stride_seq = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = key_cache.stride(1)

    single_query_cached_kv_mlu[grid](
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_tables.shape[1],
        None,
        head_size,
        output_stride_seq,
        q_stride_seq,
        kv_block_stride,
        kv_head_stride,
        block_size,
        triton.next_power_of_2(head_size),
        block_size_per_group,
    )
