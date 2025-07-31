from typing import Optional, Union

import torch
import triton
import triton.language as tl
import triton.backends.mlu.driver as driver
import torch_mlu_ops as tmo

import pytest
import torch_mlu


def get_configs():
    configs = []
    for block_m in range(3, 10):
        configs.append(triton.Config({"BLOCK_M": 2**block_m}, num_stages=3))
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["bs", "seqlen", "head_num", "head_size"],
)
@triton.jit
def apply_cross_rotary_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, token_offsets,
        # dims
        bs, seqlen, head_num, head_size,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1)
    m_begin = pid_m * BLOCK_M

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]

    rope_offsets = tl.load(token_offsets + pid_bs * seqlen + m_offsets, mask=m_offsets < seqlen, other=0)

    cos_begin = cos_emb + rope_offsets * head_size
    sin_begin = sin_emb + rope_offsets * head_size

    cs_offset = tl.arange(0, BLOCK_K)[None, :]
    cs_mask = (m_offsets < seqlen) & (cs_offset < head_size)
    cos = tl.load(cos_begin + cs_offset, mask=cs_mask, other=1.0).to(tl.float32)

    sin = tl.load(sin_begin + cs_offset, mask=cs_mask, other=1.0).to(tl.float32)
    cos = tl.view(cos, (BLOCK_M, BLOCK_K))
    sin = tl.view(sin, (BLOCK_M, BLOCK_K))

    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    for pid_head in range(head_num):
        x_offsets = m_offsets * stride_input_seqlen + k_offsets * stride_input_headsize
        input_ptr = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        # x = [x1, x2, x3, ..., xm]
        x_mask = m_offsets < seqlen
        x = tl.load(input_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

        # x1 = [0, x1, x2, x3, ..., xm-1]
        x_offsets = m_offsets * stride_input_seqlen + (k_offsets - 1) * stride_input_headsize
        x1_mask = (m_offsets < seqlen) & (k_offsets > 0)
        x1 = tl.load(input_ptr + x_offsets, mask=x1_mask, other=0).to(tl.float32)

        # x2 = [x2, x3, x4, ..., xm, 0]
        x_offsets = m_offsets * stride_input_seqlen + (k_offsets + 1) * stride_input_headsize
        x2_mask = (m_offsets < seqlen) & (k_offsets < head_size - 1)
        x2 = tl.load(input_ptr + x_offsets, mask=x2_mask, other=0.0).to(tl.float32)

        # x_cross = [-x2, x1, -x4, x3, ....]
        x_cross = tl.where(k_offsets % 2 == 0, x2, x1)
        x_rope_coef = 1 - (k_offsets + 1) % 2 * 2
        x_cross = x_cross * x_rope_coef

        x = x * cos + x_cross * sin

        out_offsets = m_offsets * stride_out_seqlen + k_offsets * stride_out_headsize
        output_ptr = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum

        tl.store(output_ptr + out_offsets, x, mask=(m_offsets < seqlen) & (k_offsets < head_size))


@triton.autotune(
    configs=get_configs(),
    key=["bs", "seqlen", "head_num", "head_size"],
)
@triton.jit
def apply_fold_rotary_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, token_offsets,
        # dims
        bs, seqlen, head_num, head_size,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1)
    m_begin = pid_m * BLOCK_M

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    in_out_k_offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = (m_offsets < seqlen)

    ro_offsets = tl.load(token_offsets + pid_bs * seqlen + m_offsets, mask=mask, other=0)

    cos_begin = cos_emb + ro_offsets * head_size
    sin_begin = sin_emb + ro_offsets * head_size
    cos = tl.load(cos_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)
    input_offsets = m_offsets * stride_input_seqlen + in_out_k_offsets * stride_input_headsize
    for pid_head in range(head_num):
        input_ptr = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.view(x, (BLOCK_M, 2, BLOCK_K // 2))
        o = tl.empty([BLOCK_M, 2, BLOCK_K // 2], dtype=tl.float32)
        o[:, 0, :] = -x[:, 1, :]
        o[:, 1, :] = x[:, 0, :]
        x = tl.view(x, (BLOCK_M, BLOCK_K))
        o = tl.view(o, (BLOCK_M, BLOCK_K))
        x = x * cos + o * sin
        out_offsets = m_offsets * stride_out_seqlen + in_out_k_offsets * stride_out_headsize
        output_ptr = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        tl.store(output_ptr + out_offsets, x, mask=mask)


def apply_cross_rotary(output: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       token_offsets: Union[int, torch.Tensor] = 0) -> torch.Tensor:
    bs, seqlen, head_num, head_size = x.shape

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype

    assert token_offsets.dtype in [torch.int32, torch.int64]
    assert token_offsets.shape == (bs, seqlen)

    assert head_size % 2 == 0, "head_size must be even for this kernel"

    BLOCK_K = head_size
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), bs)
    apply_cross_rotary_kernel[grid](
        # data ptr
        output, x, cos, sin, token_offsets,
        # dims
        bs, seqlen, head_num, head_size,
        # strides
        output.stride(0), output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0), x.stride(-3),
        x.stride(-2), x.stride(-1),
        # meta parameters
        BLOCK_K)

    return output


def apply_fold_rotary(
    output: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_offsets: Union[int, torch.Tensor] = 0,
) -> torch.Tensor:
    bs, seqlen, head_num, head_size = x.shape

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype

    assert token_offsets.dtype in [torch.int32, torch.int64]
    assert token_offsets.shape == (bs, seqlen)

    assert head_size % 2 == 0, "head_size must be even for this kernel"

    BLOCK_K = head_size
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), bs)
    apply_fold_rotary_kernel[grid](
        # data ptr
        output, x, cos, sin, token_offsets,
        # dims
        bs, seqlen, head_num, head_size,
        # strides
        output.stride(0), output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0), x.stride(-3),
        x.stride(-2), x.stride(-1),
        # meta parameters
        BLOCK_K)
    return output


def apply_rotary(
    output: torch.Tensor,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_offsets: Union[int, torch.Tensor] = 0,
    interleaved=False,
) -> torch.Tensor:
    """
    Arguments:
        input: (batch, seqlen, num_heads, head_size) if cu_seqlen is None
            else (total_seqlen, num_heads, head_size) for pack mode.
        cos: (seqlen_ro, head_size)
        sin: (seqlen_ro, head_size)
        token_offsets: integer or integer tensor
        interleaved: cross or fold
    """
    apply_rotary_func = apply_cross_rotary if interleaved else apply_fold_rotary

    return apply_rotary_func(output, input, cos, sin, token_offsets)


def apply_rotary_torch(out: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       token_offsets: Union[int, torch.Tensor] = 0, interleaved=False, inplace=False) -> torch.Tensor:
    batch, seqlen, num_heads, head_size = x.shape
    tmo.apply_rotary(x, sin, cos, token_offsets, None, interleaved, True, False, seqlen, out)


def do_rotary_benchmark(provider, x, sin, cos, token_offset, bs, seqlen, nheads, head_size, interleaved,
                        dtype=torch.float16):
    torch.manual_seed(1)
    token_offset_torch = token_offset.reshape(bs * seqlen, )
    y = torch.empty_like(x)
    y_torch, x_torch, cos_torch, sin_torch = torch.empty_like(x), x.clone(), cos.clone(), sin.clone()
    if provider == 'triton':
        ms_triton = triton.testing.do_bench(lambda: apply_rotary(y, x, cos, sin, token_offset, interleaved=interleaved))
    if provider == 'torch':
        ms_triton = triton.testing.do_bench(lambda: apply_rotary_torch(y_torch, x_torch, cos_torch, sin_torch,
                                                                       token_offset_torch, interleaved=interleaved))
    gbps_io = 2 * bs * seqlen * nheads * head_size * x.element_size()
    gbps_cs = 2 * token_offset.numel() * head_size * x.element_size()
    gbps = (gbps_io + gbps_cs) * 1e-6
    return gbps / ms_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['bs', 'seqlen', 'nheads', 'head_size', 'interleaved'],
        x_vals=[[16, 1024, 8, 128, True], [16, 1024, 8, 128, False], [16, 2048, 8, 128,
                                                                      True], [16, 2048, 8, 128, False],
                [32, 1024, 8, 128, True], [32, 1024, 8, 128, False], [32, 2048, 8, 128,
                                                                      True], [32, 2048, 8, 128, False],
                [16, 4096, 8, 128, True], [16, 4096, 8, 128, False], [16, 8192, 8, 128,
                                                                      True], [16, 8192, 8, 128, False],
                [32, 4096, 8, 128, True], [32, 4096, 8, 128, False], [32, 8192, 8, 128, True],
                [32, 8192, 8, 128, False]],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg`
        line_vals=['torch', 'triton'],
        # label name for the lines
        line_names=["torch", "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="band_width(GB/s)",  # label name for the y-axis
        plot_name="Rotary-Embedding-1d-performance",
        # name for the plot. Used also as a file name for saving the plot
        args={},
    ))
def bench_rotary_embedding(bs, seqlen, nheads, head_size, interleaved, provider):
    dtype = torch.float
    x = torch.rand((bs, seqlen, nheads, head_size), dtype=dtype, device='mlu') * 100
    cos = torch.rand((seqlen, head_size), dtype=dtype, device='mlu') * 2 - 1
    sin = torch.rand((seqlen, head_size), dtype=dtype, device='mlu') * 2 - 1

    token_offset = torch.randint(low=0, high=seqlen, size=(bs, seqlen), dtype=torch.int32, device='mlu')
    band_width = do_rotary_benchmark(provider, x, sin, cos, token_offset, bs, seqlen, nheads, head_size, interleaved,
                                     dtype)
    return band_width


@pytest.mark.parametrize("bs, seqlen, nheads, head_size", [(48, 1024, 1, 24), (4, 2048, 2, 18)])
@pytest.mark.parametrize("interleaved", [True, False])
def precision_test(bs, seqlen, nheads, head_size, interleaved):
    head_size = head_size
    seqlen = seqlen
    dtype = torch.float
    x = torch.rand((bs, seqlen, nheads, head_size), dtype=dtype, device='mlu') * 100
    cos = torch.rand((seqlen, head_size), dtype=dtype, device='mlu') * 2 - 1
    sin = torch.rand((seqlen, head_size), dtype=dtype, device='mlu') * 2 - 1

    token_offset = torch.randint(low=0, high=seqlen, size=(bs, seqlen), dtype=torch.int32, device='mlu')
    token_offset_torch = token_offset.reshape(bs * seqlen, )
    y = torch.empty_like(x)
    y_torch, x_torch, cos_torch, sin_torch = torch.empty_like(x), x.clone(), cos.clone(), sin.clone()
    apply_rotary(y, x, cos, sin, token_offset, interleaved=interleaved)
    apply_rotary_torch(y_torch, x_torch, cos_torch, sin_torch, token_offset_torch, interleaved=interleaved)
    torch.testing.assert_allclose(y, y_torch, rtol=0.001, atol=1e-1)


if __name__ == "__main__":
    bench_rotary_embedding.run(save_path=".", print_data=True)
