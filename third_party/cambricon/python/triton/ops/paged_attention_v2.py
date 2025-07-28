import torch
import torch_mlu
import triton
import triton.language as tl
import time


# (num_kv_heads, bs, max_num_partitions)
@triton.jit
def paged_attention_v2_partial(output_ptr,  # (bs, seq_q, num_heads, head_size)
                               partial_out_ptr,  # (bs, seq_q, num_heads, max_num_partitions, head_size)
                               exp_sums_ptr,  # (bs, seq_q, num_heads, max_num_partitions)
                               max_logits_ptr,  # (bs, seq_q, num_heads, max_num_partitions)
                               query_ptr,  # (bs, seq_q, num_heads, head_size)
                               key_cache_ptr,  # (num_blocks, num_kv_heads, block_size, head_size)
                               value_cache_ptr,  # (num_blocks, num_kv_heads, block_size, head_size)
                               contxt_lens_ptr,  # (bs, )
                               block_table_ptr,  # (bs, max_num_blocks_per_seq)
                               alibi_slope_ptr,  # (bs, num_heads)
                               seq_q, max_num_blocks_per_seq, num_heads, num_kv_heads, output_seq_stride,
                               query_seq_stride, window_size_left, window_size_right, qk_scale, partition_size,
                               HEAD_REPEATS: tl.constexpr, BLOCK_SIZE: tl.constexpr, SEQ_Q_BLOCK: tl.constexpr,
                               HEAD_SIZE: tl.constexpr, GROUP_K_BLOCK: tl.constexpr, HAS_ALIBI: tl.constexpr):

    pid_kv_head = tl.program_id(axis=0)
    pid_bs = tl.program_id(axis=1)
    partition_id = tl.program_id(axis=2)
    max_num_partitions = tl.num_programs(axis=2)

    total_context_len = tl.load(contxt_lens_ptr + pid_bs)
    partition_begin = partition_id * partition_size
    context_len = min(total_context_len - partition_begin, partition_size)

    if context_len <= 0:
        return

    num_partitions = (total_context_len + partition_size - 1) // partition_size

    query_bs_stride = seq_q * query_seq_stride

    partial_out_head_stride = max_num_partitions * HEAD_SIZE
    partial_out_seq_stride = num_heads * partial_out_head_stride
    partial_out_bs_stride = seq_q * partial_out_seq_stride

    max_logits_head_stride = max_num_partitions
    max_logits_seq_stride = num_heads * max_logits_head_stride
    max_logits_bs_stride = seq_q * max_logits_seq_stride

    head_offsets = tl.arange(0, HEAD_SIZE)
    head_repeat_offsets = tl.arange(0, HEAD_REPEATS)
    head_begin = pid_kv_head * HEAD_REPEATS

    cache_block_stride = num_kv_heads * BLOCK_SIZE * HEAD_SIZE

    query_begin = query_ptr + pid_bs * query_bs_stride + head_begin * HEAD_SIZE
    k_cache_begin = key_cache_ptr + pid_kv_head * BLOCK_SIZE * HEAD_SIZE
    v_cache_begin = value_cache_ptr + pid_kv_head * BLOCK_SIZE * HEAD_SIZE

    partition_blocks = partition_size // BLOCK_SIZE
    REPEAT_HEAD_SIZE: tl.constexpr = HEAD_REPEATS * HEAD_SIZE
    BLOCK_HEAD_SIZE: tl.constexpr = BLOCK_SIZE * HEAD_SIZE
    SEQ_K_BLOCK: tl.constexpr = GROUP_K_BLOCK * BLOCK_SIZE

    query_head_offset = tl.arange(0, REPEAT_HEAD_SIZE)
    block_head_offset = tl.arange(0, BLOCK_HEAD_SIZE)
    seq_q_offset = tl.arange(0, SEQ_Q_BLOCK)
    seq_k_offset = tl.arange(0, SEQ_K_BLOCK)

    if HAS_ALIBI:
        alibi_slope_begin = alibi_slope_ptr + pid_bs * num_heads + head_begin
        alibi_slope = tl.load(alibi_slope_begin + head_repeat_offsets)
        alibi_dist = partition_begin + seq_k_offset - total_context_len + 1

    max_logits_begin = max_logits_ptr + pid_bs * max_logits_bs_stride + \
                        head_begin * max_logits_head_stride + \
                        partition_id
    exp_sums_begin = exp_sums_ptr + pid_bs * max_logits_bs_stride + \
                      head_begin * max_logits_head_stride + \
                      partition_id
    partial_out_begin = partial_out_ptr + pid_bs * partial_out_bs_stride + \
                        head_begin * partial_out_head_stride + \
                        partition_id * HEAD_SIZE
    block_table_begin  = block_table_ptr + pid_bs * max_num_blocks_per_seq + \
                          partition_id * partition_blocks

    total_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_q_blocks = (seq_q + SEQ_Q_BLOCK - 1) // SEQ_Q_BLOCK

    for q_block_id in range(num_q_blocks):
        seq_q_begin = q_block_id * SEQ_Q_BLOCK
        cur_seq_q = min(seq_q - seq_q_begin, SEQ_Q_BLOCK)
        query_seq_begin = query_begin + seq_q_begin * query_seq_stride

        partial_out_seq_begin = partial_out_begin + seq_q_begin * partial_out_seq_stride
        exp_sums_seq_begin = exp_sums_begin + seq_q_begin * max_logits_seq_stride
        max_logits_seq_begin = max_logits_begin + seq_q_begin * max_logits_seq_stride

        # load query: (SEQ_Q_BLOCK, HEAD_REPEATS * HEAD_SIZE)
        query_offsets = seq_q_offset[:, None] * query_seq_stride + query_head_offset[None, :]
        query_masks = (seq_q_offset[:, None] < cur_seq_q)
        query = tl.load(query_seq_begin + query_offsets, query_masks)
        query_view = tl.view(query, (SEQ_Q_BLOCK * HEAD_REPEATS, HEAD_SIZE))

        max_logits = tl.full((SEQ_Q_BLOCK * HEAD_REPEATS, ), float('-inf'), tl.float32)
        exp_sums = tl.zeros((SEQ_Q_BLOCK * HEAD_REPEATS, ), tl.float32)
        partial_out = tl.zeros((SEQ_Q_BLOCK * HEAD_REPEATS, HEAD_SIZE), tl.float32)

        num_k_blocks = (total_blocks + GROUP_K_BLOCK - 1) // GROUP_K_BLOCK
        for k_block_id in range(num_k_blocks):
            k_block_begin = k_block_id * GROUP_K_BLOCK
            seq_k_begin = k_block_begin * BLOCK_SIZE
            seq_k_end = (k_block_id + 1) * GROUP_K_BLOCK * BLOCK_SIZE
            cur_k_blocks = min(total_blocks - k_block_begin, GROUP_K_BLOCK)
            cur_seq_k = min(context_len - seq_k_begin, SEQ_K_BLOCK)

            block_offsets = tl.arange(0, GROUP_K_BLOCK)
            # block_masks = block_offsets < cur_k_blocks
            block_ids = tl.load(block_table_begin + k_block_begin + block_offsets, block_offsets < cur_k_blocks)

            # load cache: (GROUP_K_BLOCK, BLOCK_SIZE * HEAD_SIZE)
            cache_offsets = block_ids[:, None] * cache_block_stride + block_head_offset[None, :]
            cache_masks = block_offsets[:, None] < cur_k_blocks
            key_cache = tl.load(k_cache_begin + cache_offsets, cache_masks)
            key_cache_view = tl.view(key_cache, (SEQ_K_BLOCK, HEAD_SIZE))

            # mlu impl
            qk = tl.dot(query_view, tl.trans(key_cache_view))
            # gpu impl
            # qk = tl.sum(query_view[:, None, :].to(tl.float32) * key_cache_view[None, :, :].to(tl.float32), 2)
            qk = qk_scale * qk

            if HAS_ALIBI:
                alibi_dist_begin = alibi_dist + seq_k_begin
                alibi = alibi_slope[:, None] * alibi_dist_begin[None, :]
                logits_view = tl.view(qk, (SEQ_Q_BLOCK, HEAD_REPEATS * SEQ_K_BLOCK))
                alibi_view = tl.view(alibi, (1, HEAD_REPEATS * SEQ_K_BLOCK))
                qk = tl.view(logits_view + alibi_view, (SEQ_Q_BLOCK * HEAD_REPEATS, SEQ_K_BLOCK))

            # casul mask for seq_q > 1
            # (SEQ_Q_BLOCK * HEAD_REPEATS, SEQ_K_BLOCK)
            valid_context_len = total_context_len - seq_q + 1 + seq_q_begin
            if seq_q > 1 and valid_context_len < (partition_begin + seq_k_end):
                valid_q_lens = seq_q_offset + valid_context_len
                qk_view = tl.where((seq_k_offset[None, None, :] + partition_begin + seq_k_begin) < valid_q_lens[:, None,
                                                                                                                None],
                                   tl.view(qk, (SEQ_Q_BLOCK, HEAD_REPEATS, SEQ_K_BLOCK)), float(-1e30))
                qk = tl.view(qk_view, (SEQ_Q_BLOCK * HEAD_REPEATS, SEQ_K_BLOCK))

            if cur_seq_k < SEQ_K_BLOCK:
                qk = tl.where(seq_k_offset[None, :] < cur_seq_k, qk, float(-1e30))
            qk_max = tl.maximum(tl.max(qk, 1), max_logits)
            qk = tl.exp(qk - qk_max[:, None])
            penalty_scale = tl.exp(max_logits - qk_max)
            exp_sums = exp_sums * penalty_scale + tl.sum(qk, 1)
            max_logits = qk_max

            value_cache = tl.load(v_cache_begin + cache_offsets, cache_masks)
            value_cache_view = tl.view(value_cache, (SEQ_K_BLOCK, HEAD_SIZE))
            # qkv = tl.sum(qk[:, None, :] * tl.trans(value_cache_view)[None, :, :], 2)
            qkv = tl.dot(qk.to(value_cache.dtype), value_cache_view)
            partial_out = partial_out * penalty_scale[:, None] + qkv
        partial_out = partial_out / (exp_sums[:, None] + 1e-6)

        if num_partitions == 1:
            output_head_stride = HEAD_SIZE
            output_bs_stride = seq_q * output_seq_stride
            output_begin = output_ptr + pid_bs * output_bs_stride + seq_q_begin * output_seq_stride + head_begin * output_head_stride
            output_offsets = seq_q_offset[:, None] * output_seq_stride + query_head_offset[None, :]
            tl.store(output_begin + output_offsets,
                     tl.view(partial_out, (SEQ_Q_BLOCK, REPEAT_HEAD_SIZE)).to(query.dtype), query_masks)
        else:
            logits_offsets = seq_q_offset[:, None] * max_logits_seq_stride \
                            + head_repeat_offsets[None, :] * max_logits_head_stride
            logits_masks = seq_q_offset[:, None] < cur_seq_q
            max_logits_view = tl.view(max_logits, (SEQ_Q_BLOCK, HEAD_REPEATS))
            tl.store(max_logits_seq_begin + logits_offsets, max_logits_view, logits_masks)

            exp_sums_view = tl.view(exp_sums, (SEQ_Q_BLOCK, HEAD_REPEATS))
            tl.store(exp_sums_seq_begin + logits_offsets, exp_sums_view, logits_masks)

            partial_offset = seq_q_offset[:, None, None] * partial_out_seq_stride + \
                            head_repeat_offsets[None, :, None] * partial_out_head_stride + \
                            head_offsets[None, None, :]
            partial_mask = seq_q_offset[:, None, None] < cur_seq_q
            partial_out_view = tl.view(partial_out, (SEQ_Q_BLOCK, HEAD_REPEATS, HEAD_SIZE))
            tl.store(partial_out_seq_begin + partial_offset, partial_out_view, partial_mask)


# Grid: (num_heads, bs)
@triton.jit
def paged_attention_v2_reduce(output_ptr,  # (bs, seq_q, num_heads, head_size)
                              partial_out_ptr,  # (bs, seq_q, num_heads, max_num_partitions, head_size)
                              exp_sums_ptr,  # (bs, seq_q, num_heads, max_num_partitions)
                              max_logits_ptr,  # (bs, seq_q, num_heads, max_num_partitions)
                              context_lens_ptr,  # (bs, )
                              seq_q, partition_size, max_num_partitions, output_seq_stride, HEAD_SIZE: tl.constexpr,
                              SEQ_Q_BLOCK: tl.constexpr, PARTITION_BLOCK: tl.constexpr):

    pid_head = tl.program_id(0)
    pid_bs = tl.program_id(1)
    num_heads = tl.num_programs(0)

    context_len = tl.load(context_lens_ptr + pid_bs)
    num_partitions = (context_len + partition_size - 1) // partition_size
    if num_partitions == 1:
        return

    output_head_stride = HEAD_SIZE
    output_bs_stride = seq_q * output_seq_stride

    partial_out_head_stride = max_num_partitions * HEAD_SIZE
    partial_out_seq_stride = num_heads * partial_out_head_stride
    partial_out_bs_stride = seq_q * partial_out_seq_stride

    max_logits_head_stride = max_num_partitions
    max_logits_seq_stride = num_heads * max_logits_head_stride
    max_logits_bs_stride = seq_q * max_logits_seq_stride

    output_begin = output_ptr + pid_bs * output_bs_stride + pid_head * output_head_stride
    partial_out_begin = partial_out_ptr + pid_bs * partial_out_bs_stride + pid_head * partial_out_head_stride
    exp_sums_begin = exp_sums_ptr + pid_bs * max_logits_bs_stride + pid_head * max_logits_head_stride
    max_logits_begin = max_logits_ptr + pid_bs * max_logits_bs_stride + pid_head * max_logits_head_stride

    seq_offsets = tl.arange(0, SEQ_Q_BLOCK)
    head_offsets = tl.arange(0, HEAD_SIZE)
    partition_offsets = tl.arange(0, PARTITION_BLOCK)
    PARTITION_HEAD_SIZE: tl.constexpr = PARTITION_BLOCK * HEAD_SIZE
    partition_head_offsets = tl.arange(0, PARTITION_HEAD_SIZE)

    num_q_blocks = (seq_q + SEQ_Q_BLOCK - 1) // SEQ_Q_BLOCK
    for q_block_id in range(num_q_blocks):
        seq_q_begin = q_block_id * SEQ_Q_BLOCK
        cur_seq_q = min(seq_q - seq_q_begin, SEQ_Q_BLOCK)

        output_seq_begin = output_begin + seq_q_begin * output_seq_stride
        partial_out_seq_begin = partial_out_begin + seq_q_begin * partial_out_seq_stride
        exp_sums_seq_begin = exp_sums_begin + seq_q_begin * max_logits_seq_stride
        max_logits_seq_begin = max_logits_begin + seq_q_begin * max_logits_seq_stride

        global_max_logit = tl.full((SEQ_Q_BLOCK, ), float('-inf'), tl.float32)
        out_acc = tl.zeros((SEQ_Q_BLOCK, HEAD_SIZE), tl.float32)
        sum_acc = tl.zeros((SEQ_Q_BLOCK, ), tl.float32)
        num_partition_blocks = (num_partitions + PARTITION_BLOCK - 1) // PARTITION_BLOCK
        for partition_id in range(num_partition_blocks):
            partition_begin = partition_id * PARTITION_BLOCK
            cur_partitions = min(num_partitions - partition_begin, PARTITION_BLOCK)

            partial_out_part_begin = partial_out_seq_begin + partition_begin * HEAD_SIZE
            exp_sums_part_begin = exp_sums_seq_begin + partition_begin
            max_logits_part_begin = max_logits_seq_begin + partition_begin

            partial_out_offsets = seq_offsets[:, None] * partial_out_seq_stride + partition_head_offsets[None, :]
            partial_out_masks = (seq_offsets[:, None] < cur_seq_q) & \
                                (partition_head_offsets[None, :] < cur_partitions * HEAD_SIZE)
            # (SEQ_Q_BLOCK, PARTITION_BLOCK * HEAD_SIZE)
            partial_out = tl.load(partial_out_part_begin + partial_out_offsets, partial_out_masks)
            partial_out_view = tl.view(partial_out, (SEQ_Q_BLOCK, PARTITION_BLOCK, HEAD_SIZE))

            # (SEQ_Q_BLOCK, PARTITION_BLOCK)
            logits_offsets = seq_offsets[:, None] * max_logits_seq_stride + partition_offsets[None, :]
            logits_masks = (seq_offsets[:, None] < cur_seq_q) & (partition_offsets[None, :] < cur_partitions)
            logits = tl.load(max_logits_part_begin + logits_offsets, logits_masks, other=float('-inf'))

            max_logit = tl.maximum(tl.max(logits, 1), global_max_logit)
            scale = tl.exp(global_max_logit - max_logit)
            logits = tl.exp(logits - max_logit[:, None])
            global_max_logit = max_logit

            # (SEQ_Q_BLOCK, PARTITION_BLOCK)
            exp_sums = tl.load(exp_sums_part_begin + logits_offsets, logits_masks, other=0.0)
            sum_acc = tl.sum(exp_sums * logits, 1) + sum_acc * scale

            # (SEQ_Q_BLOCK, PARTITION_BLOCK, HEAD_SIZE)
            partial_out_view = partial_out_view * exp_sums[:, :, None] * logits[:, :, None]
            out_acc = out_acc * scale[:, None] + tl.sum(partial_out_view, 1)

        out_acc = out_acc / (sum_acc[:, None] + 1e-6)
        output_offsets = seq_offsets[:, None] * output_seq_stride + head_offsets[None, :]
        output_masks = (seq_offsets[:, None] < cur_seq_q)
        tl.store(output_seq_begin + output_offsets, out_acc, output_masks)


def launch_paged_attention_v2(output: torch.Tensor, partial_out: torch.Tensor, exp_sums: torch.Tensor,
                              max_logits: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                              key_cache: torch.Tensor, value_cache: torch.Tensor, context_lens: torch.Tensor,
                              block_tables: torch.Tensor, slot_mapping: torch.Tensor, alibi_slopes: torch.Tensor,
                              window_size_left: int, window_size_right: int, qk_scale: float, PARTITION_SIZE: int,
                              SEQ_Q_BLOCK: int, GROUP_K_BLOCK: int, num_stages: int):

    assert partial_out.dim() == 5
    assert key_cache.dim() == 4
    assert block_tables.dim() == 2

    bs, seq_q, num_heads, max_num_partitions, head_size = partial_out.size()
    num_blocks, num_kv_heads, block_size, _ = key_cache.size()
    _, max_num_blocks_per_seq = block_tables.size()

    assert num_heads % num_kv_heads == 0

    # check size
    assert exp_sums.size() == (bs, seq_q, num_heads, max_num_partitions)
    assert max_logits.size() == (bs, seq_q, num_heads, max_num_partitions)
    assert query.size() == (bs, seq_q, num_heads, head_size)
    if key is not None:
        assert key.size() == (bs, seq_q, num_kv_heads, head_size)
    if value is not None:
        assert value.size() == (bs, seq_q, num_kv_heads, head_size)
    assert key_cache.size() == (num_blocks, num_kv_heads, block_size, head_size)
    assert value_cache.size() == (num_blocks, num_kv_heads, block_size, head_size)
    assert context_lens.size() == (bs, )
    assert block_tables.size() == (bs, max_num_blocks_per_seq)
    if slot_mapping is not None:
        assert slot_mapping.size() == (bs, seq_q)
    if alibi_slopes is not None:
        assert alibi_slopes.size() == (bs, num_heads)
    assert window_size_left >= 0 and window_size_right == 0

    # check stride
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()
    assert context_lens.is_contiguous()
    assert block_tables.is_contiguous()
    if alibi_slopes is not None:
        assert alibi_slopes.is_contiguous()

    # check dtype
    assert max_logits.dtype == torch.float32
    assert exp_sums.dtype == torch.float32
    assert partial_out.dtype == torch.float32
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert key_cache.dtype == value_cache.dtype == query.dtype
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32
    if alibi_slopes is not None:
        assert alibi_slopes.dtype == torch.float32

    query_seq_stride = query.stride(1)
    output_seq_stride = output.stride(1)
    head_repeats = num_heads // num_kv_heads

    HAS_ALIBI = alibi_slopes is not None
    SEQ_Q_BLOCK = min(triton.next_power_of_2(seq_q), SEQ_Q_BLOCK)
    grid_partial = lambda meta: (num_kv_heads, bs, max_num_partitions)
    paged_attention_v2_partial[grid_partial](output, partial_out, exp_sums, max_logits, query, key_cache, value_cache,
                                             context_lens, block_tables, alibi_slopes, seq_q, max_num_blocks_per_seq,
                                             num_heads, num_kv_heads, output_seq_stride, query_seq_stride,
                                             window_size_left, window_size_right, qk_scale, PARTITION_SIZE,
                                             head_repeats, block_size, SEQ_Q_BLOCK, head_size, GROUP_K_BLOCK, HAS_ALIBI,
                                             num_stages=num_stages)

    grid_reduce = lambda meta: (num_heads, bs)
    PARTITION_BLOCK = min(triton.next_power_of_2(max_num_partitions), 8)
    if max_num_partitions > 1:
        paged_attention_v2_reduce[grid_reduce](output, partial_out, exp_sums, max_logits, context_lens, seq_q,
                                               PARTITION_SIZE, max_num_partitions, output_seq_stride, head_size,
                                               SEQ_Q_BLOCK, PARTITION_BLOCK)
