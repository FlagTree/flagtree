import random
import pytest
import torch
from decode_attention_int4kv import (
    decode_attention_fwd_int4kv,
    destindex_copy_quantize_int4kv_init,
    destindex_dequantize_int4kv,
    gama_int4kv_init,
    degama_int4kv_init
)
from decode_attention import decode_attention_fwd_ori


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="function", autouse=True)
def setup():
    print("invoke setUp")
    set_all_seeds(42)


def decode_attention_once(B, KV_SEQ, H_Q, H_KV, D):
    print("invoke _test_decode_attention_once")
    dtype = torch.float16
    seq_len = KV_SEQ  # This represents the number of tokens already in the sequence (kv seq)
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    k_buffer_g = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    k_buffer_deq = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer_deq = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    k_buffer_int4 = torch.empty(
        (total_tokens, H_KV, D // 2), dtype=torch.int8, device="cuda"
    )
    v_buffer_int4 = torch.empty(
        (total_tokens, H_KV, D // 2), dtype=torch.int8, device="cuda"
    )

    k_quant_group_size = int(2)
    v_quant_group_size = int(2)

    k_buffer_scales = torch.empty(
        (total_tokens, H_KV, D // k_quant_group_size * 2), dtype=dtype, device="cuda"
    )
    v_buffer_scales = torch.empty(
        (total_tokens, H_KV, D // v_quant_group_size * 2), dtype=dtype, device="cuda"
    )

    k_gamas = torch.empty(
        (H_KV, D), dtype=dtype, device="cuda"
    )

    gama_int4kv_init(k_buffer, k_gamas, k_buffer_g, alpha=0.5)
    # degama_int4kv_init(k_buffer_g, k_gamas, k_buffer_deq, alpha=0.5)
    # print(k_buffer)
    # print(k_buffer_deq)

    destindex_copy_quantize_int4kv_init(
        k_buffer_g,
        k_buffer_int4,
        k_buffer_scales,
        k_quant_group_size,
    )
    # destindex_dequantize_int4kv(
    #     k_buffer_int4,
    #     k_buffer_scales,
    #     k_buffer_deq,
    #     k_quant_group_size
    # )
    # print(k_buffer)
    # print(k_buffer_deq)

    # self.assertTrue(torch.allclose(k_buffer, k_buffer_deq, atol=1e-1, rtol=1e-1))

    destindex_copy_quantize_int4kv_init(
        v_buffer,
        v_buffer_int4,
        v_buffer_scales,
        v_quant_group_size,
    )
    # destindex_dequantize_int4kv(
    #     v_buffer_int4,
    #     v_buffer_scales,
    #     v_buffer_deq,
    #     v_quant_group_size
    # )

    # print(v_buffer)
    # print(v_buffer_deq)

    # print(k_buffer_scales)
    # print(v_buffer_scales)

    # self.assertTrue(torch.allclose(v_buffer, v_buffer_deq, atol=1e-1, rtol=1e-1))

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
    o_ori = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

    req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
    b_req_idx = torch.arange(B, device="cuda")
    b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
    b_seq_len = torch.full((B,), seq_len, device="cuda")

    attn_logits = torch.empty(
        (H_Q, total_tokens),
        dtype=torch.float16,
        device="cuda")

    decode_attention_fwd_ori(
        q,
        k_buffer,
        v_buffer,
        o_ori,
        req_to_token,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        attn_logits,
        seq_len,
        sm_scale,
    )
    torch.cuda.synchronize()

    best_config = {
        "kernel_kind": "v1_2stages",
        "best_config": {
            "stage1": {
                "BLOCK_N": 16,
                "BLOCK": 16,
                "num_stages": 4,
                "num_warps": 2
            },
            "stage2": {
                "BLOCK_N": 64,
                "num_stages": 4,
                "num_warps": 1
            }
        }
    }

    decode_attention_fwd_int4kv(
        q,
        k_buffer_int4,
        v_buffer_int4,
        k_buffer_scales,
        v_buffer_scales,
        k_gamas,
        o,
        req_to_token,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        attn_logits,
        seq_len,
        sm_scale,
        k_quant_group_size,
        v_quant_group_size,
        best_config,
    )

    print(o_ori)
    print(o)

    assert torch.allclose(o, o_ori, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("B,KV_SEQ,H_Q,H_KV,D", [
    (2, 2, 2, 2, 16),
    # (8, 2000, 32, 32, 128),
    # (64, 2000, 32, 32, 128),
    # (128, 100, 32, 32, 128),
    # (1, 100, 32, 8, 128),      # GQA
    # (1, 1000, 32, 8, 128),
    # (16, 2000, 32, 8, 128),
    # (128, 1000, 32, 8, 128),
])
def test_decode_attention(B, KV_SEQ, H_Q, H_KV, D):
    print(f"B, KV_SEQ, H_Q, H_KV, D: {B} {KV_SEQ} {H_Q} {H_KV} {D}")
    decode_attention_once(B, KV_SEQ, H_Q, H_KV, D)


if __name__ == "__main__":
    pytest.main()