"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch
import torch_mlu

import triton
import triton.language as tl
import triton.backends.mlu.driver as driver

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())

TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")


def cfggen1():
    block_m = [1, 2, 4, 6, 8]
    num_stages = [1, 3]
    configs = [triton.Config({
        "BLOCK_M": m,
    }, num_stages=s) for m in block_m for s in num_stages]
    return configs


@triton.autotune(configs=cfggen1(), key=["M", "N"])
@triton.heuristics({
    'BLOCK_N': lambda args: args['N'],
    'num_warps': lambda args: 1,
})
@triton.jit(do_not_specialize=["eps"])
def _layer_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m
    n_offset = tl.arange(0, BLOCK_N)
    gamma = tl.load(W + n_offset)
    beta = tl.load(B + n_offset)
    for m_idx in range(0, split_m, BLOCK_M):
        m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M
        inp = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)

        alpha = 1.0 / N
        sum = tl.sum(inp, 1)
        # X, 1
        mean = sum * alpha
        var = tl.sum(inp * inp, 1) * alpha - mean * mean

        # X, R
        rstd = tl.rsqrt(var + eps)
        # Write mean / rstd
        tl.store(Mean + m_offset, mean, mask=m_offset < M)
        tl.store(Rstd + m_offset, rstd, mask=m_offset < M)

        out = (inp - mean[:, None]) * rstd[:, None]
        opt_out = out * gamma + beta
        # Write output
        tl.store(Y + offset, opt_out, mask=mask)


# %%
# Backward pass
# -------------
#
# The backward pass for the layer normalization operator is a bit more involved than the forward pass.
# Let :math:`\hat{x}` be the normalized inputs :math:`\frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} }` before the linear transformation,
# the Vector-Jacobian Products (VJP) :math:`\nabla_{x}` of :math:`x` are given by:
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# where :math:`\odot` denotes the element-wise multiplication, :math:`\cdot` denotes the dot product, and :math:`\sigma` is the standard deviation.
# :math:`c_1` and :math:`c_2` are intermediate constants that improve the readability of the following implementation.
#
# For the weights :math:`w` and biases :math:`b`, the VJPs :math:`\nabla_{w}` and :math:`\nabla_{b}` are more straightforward:
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}
#
# Since the same weights :math:`w` and biases :math:`b` are used for all rows in the same batch, their gradients need to sum up.
# To perform this step efficiently, we use a parallel reduction strategy: each kernel instance accumulates
# partial :math:`\nabla_{w}` and :math:`\nabla_{b}` across certain rows into the final gredients.


def cfggen2():
    block_m = [1, 2, 3, 4, 5, 6, 8]
    num_stages = [1, 3]
    configs = [triton.Config({
        "BLOCK_M": m,
    }, num_stages=s) for m in block_m for s in num_stages]
    return configs


@triton.autotune(configs=cfggen2(), key=["M", "N"], reset_to_zero=["DW", "DB"])
@triton.heuristics({
    'BLOCK_N': lambda args: args['N'],
    'num_warps': lambda args: 1,
})
@triton.jit
def _layer_norm_bwd(DX,  # pointer to the input gradient
                    DY,  # pointer to the output gradient
                    DW,  # pointer to the partial sum of weights gradient
                    DB,  # pointer to the partial sum of biases gradient
                    X,  # pointer to the input
                    W,  # pointer to the weights
                    Mean,  # pointer to the mean
                    Rstd,  # pointer to the 1/std
                    M,  # number of rows in X
                    N,  # number of columns in X
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)

    row_start = pid * BLOCK_M
    cols = tl.arange(0, BLOCK_N)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_M

    X += cols[None, :]
    DY += cols[None, :]
    W += cols[None, :]
    DX += cols[None, :]
    w = tl.load(W).to(tl.float32)
    alpha = 1 / N
    partial_dw = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    partial_db = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_M)
        mask = row_off[:, None] < M
        # Load data to SRAM
        off = row_off[:, None] * N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row_off, mask=row_off < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask=row_off < M)[:, None].to(tl.float32)
        # Compute dx
        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=1)[:, None]
        c2 = tl.sum(wdy, axis=1)[:, None]
        dx = (wdy - (x_hat * c1 + c2) * alpha) * rstd

        # Accumulate partial sums for dw/db
        partial_dw += (dy * x_hat).to(tl.float32)
        partial_db += (dy).to(tl.float32)
        # Write dx
        tl.store(DX + off, dx.to(x.dtype), mask=mask)

    dw = tl.sum(partial_dw, axis=0)
    db = tl.sum(partial_db, axis=0)
    tl.atomic_add(DW + cols, dw)
    tl.atomic_add(DB + cols, db)


# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have the reduction dim is no bigger than 15872.


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device='mlu')
        rstd = torch.empty((M, ), dtype=torch.float32, device='mlu')
        num_warps = 1
        # enqueue kernel
        _layer_norm_fwd[(TOTAL_CORE_NUM, )](x_arg, y, weight, bias, mean, rstd, M, N, eps, opt_level="Om")
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        # allocate output
        dw = torch.zeros((w.shape[0], ), dtype=w.dtype, device=w.device)
        db = torch.zeros((w.shape[0], ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd[(TOTAL_CORE_NUM, )](dx, dy, dw, db, x, w, m, v, M, N, opt_level="Om")
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device='mlu'):
    # create data
    torch.manual_seed(0)
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='mlu', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='mlu', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='mlu')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    list(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[512 * i for i in range(2, 21)],
            line_arg='provider',
            line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
            line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
            styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
            ylabel='GB/s',
            plot_name=f'layer-norm-{mode}-performance',
            args={'M': 4096, 'dtype': torch.float32, 'mode': mode},
        ) for mode in ['forward', 'backward']))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device='mlu'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='mlu', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='mlu', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='mlu')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':

        def y_fwd():
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

    if provider == 'torch':

        def y_fwd():
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

    if provider == 'apex':
        apex_layer_norm = apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype)

        def y_fwd():
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':

        def gbps(ms):
            return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(48, 8192, torch.float32)
bench_layer_norm.run(save_path='.', print_data=True)

# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
