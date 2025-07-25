import torch
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_mlu(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_heads, block_size, head_size]
    value_cache_ptr,  # [num_blocks, num_heads, block_size, head_size]
    slot_mapping_ptr,  # [num_tokens]
    key_stride,
    value_stride,
    num_tokens,
    num_heads,
    block_size,
    tokens_per_group: tl.constexpr,
    pad_num_heads: tl.constexpr,
    head_size: tl.constexpr,
    pad_tokens_per_group: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    token_begin = pid * tokens_per_group
    slot_offset = token_begin + \
                  tl.arange(0, pad_tokens_per_group)[:, None, None]
    slot_idx = tl.load(slot_mapping_ptr + slot_offset, mask=slot_offset < num_tokens, other=0)

    num_heads_offset = tl.arange(0, pad_num_heads)[None, :, None]
    head_size_offset = tl.arange(0, head_size)[None, None, :]

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    kv_offsets = slot_offset * key_stride + num_heads_offset * head_size + head_size_offset * 1
    cache_offsets = (block_idx * num_heads * block_size  + block_offset) * head_size + \
                    (num_heads_offset * block_size * head_size) + \
                    (head_size_offset * 1)

    mask = (slot_idx >= 0) & (slot_offset < num_tokens) & (num_heads_offset < num_heads)
    cache = tl.load(key_ptr + kv_offsets, mask=mask)
    tl.store(key_cache_ptr + cache_offsets, cache, mask=mask)

    kv_offsets = slot_offset * value_stride + num_heads_offset * head_size + head_size_offset * 1
    cache = tl.load(value_ptr + kv_offsets, mask=mask)
    tl.store(value_cache_ptr + cache_offsets, cache, mask=mask)


def reshape_and_cache(
    key: torch.tensor,
    value: torch.tensor,
    key_cache: torch.tensor,
    value_cache: torch.tensor,
    slot_mapping: torch.tensor,
    num_tokens_per_group: int,
) -> None:
    """
        key: [num_tokens, num_heads, head_size]
        value: [num_tokens, num_heads, head_size]
        key_cache: [num_blocks, num_heads, block_size, head_size]
        value_cache: [num_blocks, num_heads, block_size, head_size]
        slot_mapping: [num_tokens]
    """
    num_tokens = slot_mapping.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[2]
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]

    grid = lambda meta: (triton.cdiv(num_tokens, num_tokens_per_group), )
    reshape_and_cache_mlu[grid](key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride,
                                num_tokens, num_heads, block_size, num_tokens_per_group,
                                triton.next_power_of_2(num_heads), head_size,
                                triton.next_power_of_2(num_tokens_per_group))
