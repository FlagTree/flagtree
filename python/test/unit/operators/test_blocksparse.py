# import pytest

# import triton
# import triton.ops

# import torch


# def is_hip_mi200():
#     target = triton.runtime.driver.active.get_current_target()
#     return target.backend == 'hip' and target.arch == 'gfx90a'


# def sparsify_tensor(x, mask, block):
#     ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
#     for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
#         ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
#     return ret


# def make_pair(shape, device="cuda", alpha=1e-2, beta=0., trans=False, data=None, dtype=torch.float32):
#     if data is None:
#         data = torch.randn(shape, dtype=torch.float32, requires_grad=True, device=device)
#     ref_ret = data
#     ref_ret = ref_ret * alpha + beta
#     ref_ret = ref_ret.half().to(dtype)
#     if trans:
#         ref_ret = ref_ret.t().requires_grad_()
#     ref_ret = ref_ret.detach().requires_grad_()
#     tri_ret = ref_ret.clone().detach().requires_grad_()
#     return ref_ret, tri_ret


# def mask_tensor(x, mask, block, value=0):
#     ret = x.clone()
#     for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
#         ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
#     return ret


# @pytest.mark.parametrize("MODE", ["sdd", "dds", "dsd"])
# @pytest.mark.parametrize("TRANS_A", [False, True])
# @pytest.mark.parametrize("TRANS_B", [False, True])
# @pytest.mark.parametrize("BLOCK", [16, 32, 64])
# @pytest.mark.parametrize("DTYPE", [torch.float16])
# def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, device, Z=3, H=2, M=512, N=384, K=256):
#     seed = 0
#     torch.manual_seed(seed)
#     is_sdd = MODE == "sdd"
#     is_dsd = MODE == "dsd"
#     is_dds = MODE == "dds"
#     do_sparsify = lambda x: sparsify_tensor(x, layout, BLOCK)
#     do_mask = lambda x: mask_tensor(x, layout, BLOCK)
#     # create inputs
#     # create op
#     a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
#     b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
#     c_shape = (Z, H, M, N)
#     shape = {
#         "sdd": (M, N),
#         "dsd": (a_shape[2], a_shape[3]),
#         "dds": (b_shape[2], b_shape[3]),
#     }[MODE]
#     layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
#     layout[1, 2, :] = 0
#     layout[1, :, 1] = 0
#     # create data
#     a_ref, a_tri = make_pair(a_shape, alpha=.1, dtype=DTYPE)
#     b_ref, b_tri = make_pair(b_shape, alpha=.1, dtype=DTYPE)
#     dc_ref, dc_tri = make_pair(c_shape, dtype=DTYPE)
#     # compute [torch]
#     dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
#     a_ref = do_mask(a_ref) if is_dsd else a_ref
#     b_ref = do_mask(b_ref) if is_dds else b_ref
#     a_ref.retain_grad()
#     b_ref.retain_grad()
#     c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref, b_ref.transpose(2, 3) if TRANS_B else b_ref)
#     c_ref.backward(dc_ref)
#     c_ref = do_sparsify(c_ref) if is_sdd else c_ref
#     da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
#     db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
#     # triton result
#     dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
#     a_tri = do_sparsify(a_tri) if is_dsd else a_tri
#     b_tri = do_sparsify(b_tri) if is_dds else b_tri
#     a_tri.retain_grad()
#     b_tri.retain_grad()
#     op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device=device)
#     c_tri = op(a_tri, b_tri)
#     c_tri.backward(dc_tri)
#     da_tri = a_tri.grad
#     db_tri = b_tri.grad

#     # Bigger tolerance for AMD MI200 devices.
#     # MI200 devices use reduced precision fp16 and bf16 and flush input and
#     # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
#     tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}

#     # compare
#     torch.testing.assert_close(c_ref, c_tri, **tol)
#     torch.testing.assert_close(da_ref, da_tri, **tol)
#     torch.testing.assert_close(db_ref, db_tri, **tol)


# configs = [
#     (16, 256),
#     (32, 576),
#     (64, 1871),
#     (128, 2511),
# ]


# @pytest.mark.parametrize("is_dense", [False, True])
# @pytest.mark.parametrize("BLOCK, WIDTH", configs)
# def test_softmax(BLOCK, WIDTH, is_dense, device, Z=2, H=2, is_causal=True, scale=0.4):
#     # set seed
#     torch.random.manual_seed(0)
#     Z, H, M, N = 2, 3, WIDTH, WIDTH
#     # initialize layout
#     # make sure each row has at least one non-zero element
#     layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
#     if is_dense:
#         layout[:] = 1
#     else:
#         layout[1, 2, :] = 0
#         layout[1, :, 1] = 0
#     # initialize data
#     a_shape = (Z, H, M, N)
#     a_ref, a_tri = make_pair(a_shape)
#     dout_ref, dout_tri = make_pair(a_shape)
#     # compute [torch]
#     a_ref = mask_tensor(a_ref, layout, BLOCK, value=float("-inf"))
#     a_ref.retain_grad()
#     at_mask = torch.ones((M, N), device=device)
#     if is_causal:
#         at_mask = torch.tril(at_mask)
#     M = at_mask[None, None, :, :] + torch.zeros_like(a_ref)
#     a_ref[M == 0] = float("-inf")
#     out_ref = torch.softmax(a_ref * scale, -1)
#     out_ref.backward(dout_ref)
#     out_ref = sparsify_tensor(out_ref, layout, BLOCK)
#     da_ref = sparsify_tensor(a_ref.grad, layout, BLOCK)
#     # compute [triton]
#     a_tri = sparsify_tensor(a_tri, layout, BLOCK)
#     a_tri.retain_grad()
#     dout_tri = sparsify_tensor(dout_tri, layout, BLOCK)
#     op = triton.ops.blocksparse.softmax(layout, BLOCK, device=device, is_dense=is_dense)
#     out_tri = op(a_tri, scale=scale, is_causal=is_causal)
#     out_tri.backward(dout_tri)
#     da_tri = a_tri.grad
#     # compare
#     torch.testing.assert_close(out_tri, out_ref, equal_nan=True)
#     torch.testing.assert_close(da_tri, da_ref, equal_nan=True)


# @pytest.mark.parametrize("block", [16, 32, 64])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
# def test_attention_fwd_bwd(
#     block,
#     dtype,
#     device,
#     input_scale=1.0,
#     scale=1 / 8.0,
#     n_ctx=256,
#     batch_size=2,
#     n_heads=2,
# ):
#     capability = torch.cuda.get_device_capability()
#     if capability[0] < 7:
#         pytest.skip("Only test tl.dot() on devices with sm >= 70")

#     # inputs
#     qkv_shape = (batch_size, n_heads, n_ctx, 64)
#     qkvs = [
#         torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)
#     ]

#     # Triton:
#     n_blocks = n_ctx // block
#     layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
#     query, key, value = [x.clone() for x in qkvs]
#     query.retain_grad()
#     key.retain_grad()
#     value.retain_grad()
#     attn_out = triton_attention(layout, block, query=query, key=key, value=value, scale=scale)
#     # ad hoc loss
#     loss = (attn_out**2).mean()
#     loss.backward()
#     grads = [query.grad, key.grad, value.grad]

#     # Torch version:
#     torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
#     attn_mask = torch.ones([n_ctx, n_ctx], device=device, dtype=dtype)
#     attn_mask = torch.tril(attn_mask, diagonal=0)
#     attn_mask = 1e6 * (-1 + (attn_mask.reshape((1, 1, n_ctx, n_ctx)).cuda()))
#     torch_q.retain_grad()
#     torch_k.retain_grad()
#     torch_v.retain_grad()
#     scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
#     scores = scores + attn_mask
#     probs = torch.softmax(scores, dim=-1)
#     torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)
#     # ad hoc loss
#     torch_loss = (torch_attn_out**2).mean()
#     torch_loss.backward()
#     torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]

#     # comparison
#     # print(f"Triton loss {loss} and torch loss {torch_loss}.  Also checking grads...")
#     torch.testing.assert_close(loss, torch_loss, atol=1e-3, rtol=0)

#     # Bigger tolerance for AMD MI200 devices.
#     # MI200 devices use reduced precision fp16 and bf16 and flush input and
#     # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
#     tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}
#     for g1, g2 in zip(grads, torch_grads):
#         torch.testing.assert_close(g1, g2, **tol)


# @pytest.mark.parametrize("block", [16, 32, 64])
# def triton_attention(
#     layout,
#     block: int,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     scale: float,
# ):
#     sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True,
#                                                       device=value.device)
#     sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False,
#                                                       device=value.device)
#     sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device)

#     w = sparse_dot_sdd_nt(query, key)
#     w = sparse_softmax(w, scale=scale, is_causal=True)
#     a = sparse_dot_dsd_nn(w, value)
#     return a
# 优先使用 PyTorch，失败则使用 PaddlePaddle
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    import paddle
    HAS_PADDLE = True
    print("🚀 using paddle")

import pytest
import triton
import triton.ops


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def sparsify_tensor(x, mask, block):
    if HAS_TORCH:
        ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
        for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
            ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    else:
        ret = paddle.empty((x.shape[0], mask.sum().item(), block, block), dtype=x.dtype)
        nonzero_indices = paddle.nonzero(mask)
        for idx, (h, i, j) in enumerate(nonzero_indices):
            ret[:, idx, :, :] = x[:, h.item(), i.item() * block:(i.item() + 1) * block, j.item() * block:(j.item() + 1) * block]
    return ret


def make_pair(shape, device="cuda", alpha=1e-2, beta=0., trans=False, data=None, dtype=None):
    if HAS_TORCH:
        if dtype is None:
            dtype = torch.float32
        if data is None:
            data = torch.randn(shape, dtype=torch.float32, requires_grad=True, device=device)
        ref_ret = data
        ref_ret = ref_ret * alpha + beta
        ref_ret = ref_ret.half().to(dtype)
        if trans:
            ref_ret = ref_ret.t().requires_grad_()
        ref_ret = ref_ret.detach().requires_grad_()
        tri_ret = ref_ret.clone().detach().requires_grad_()
    else:
        if dtype is None:
            dtype = 'float32'
        if data is None:
            data = paddle.randn(shape, dtype='float32')
            data.stop_gradient = False
        ref_ret = data
        ref_ret = ref_ret * alpha + beta
        ref_ret = ref_ret.astype('float16').astype(dtype)
        if trans:
            ref_ret = paddle.transpose(ref_ret, [1, 0])
            ref_ret.stop_gradient = False
        ref_ret.stop_gradient = False
        tri_ret = ref_ret.clone()
        tri_ret.stop_gradient = False
    return ref_ret, tri_ret


def mask_tensor(x, mask, block, value=0):
    if HAS_TORCH:
        ret = x.clone()
        for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
            ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    else:
        ret = x.clone()
        zero_mask = (mask == 0)
        zero_indices = paddle.nonzero(zero_mask)
        for h, i, j in zero_indices:
            ret[:, h.item(), i.item() * block:(i.item() + 1) * block, j.item() * block:(j.item() + 1) * block] = value
    return ret


@pytest.mark.parametrize("MODE", ["sdd", "dds", "dsd"])
@pytest.mark.parametrize("TRANS_A", [False, True])
@pytest.mark.parametrize("TRANS_B", [False, True])
@pytest.mark.parametrize("BLOCK", [16, 32, 64])
@pytest.mark.parametrize("DTYPE", [torch.float16 if HAS_TORCH else 'float16'])
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, device, Z=3, H=2, M=512, N=384, K=256):
    seed = 0
    if HAS_TORCH:
        torch.manual_seed(seed)
    else:
        paddle.seed(seed)
    
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: mask_tensor(x, layout, BLOCK)
    
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    c_shape = (Z, H, M, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    
    if HAS_TORCH:
        layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
        layout[1, 2, :] = 0
        layout[1, :, 1] = 0
    else:
        layout = paddle.randint(0, 2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
        layout[1, 2, :] = 0
        layout[1, :, 1] = 0
    
    # create data
    a_ref, a_tri = make_pair(a_shape, alpha=.1, dtype=DTYPE)
    b_ref, b_tri = make_pair(b_shape, alpha=.1, dtype=DTYPE)
    dc_ref, dc_tri = make_pair(c_shape, dtype=DTYPE)
    
    # compute [torch/paddle]
    dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    
    if HAS_TORCH:
        a_ref.retain_grad()
        b_ref.retain_grad()
        c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref, b_ref.transpose(2, 3) if TRANS_B else b_ref)
        c_ref.backward(dc_ref)
    else:
        a_ref = a_ref.clone().detach()
        a_ref.stop_gradient = False
        b_ref = b_ref.clone().detach()
        b_ref.stop_gradient = False
        c_ref = paddle.matmul(paddle.transpose(a_ref, [0, 1, 3, 2]) if TRANS_A else a_ref, 
                             paddle.transpose(b_ref, [0, 1, 3, 2]) if TRANS_B else b_ref)
        c_ref.backward(dc_ref)
    
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
    
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    
    if HAS_TORCH:
        a_tri.retain_grad()
        b_tri.retain_grad()
    else:
        a_tri = a_tri.clone().detach()
        a_tri.stop_gradient = False

        b_tri = b_tri.clone().detach()
        b_tri.stop_gradient = False
        
        
    
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device=device)
    c_tri = op(a_tri, b_tri)
    c_tri.backward(dc_tri)
    da_tri = a_tri.grad
    db_tri = b_tri.grad

    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}

    # compare
    if HAS_TORCH:
        torch.testing.assert_close(c_ref, c_tri, **tol)
        torch.testing.assert_close(da_ref, da_tri, **tol)
        torch.testing.assert_close(db_ref, db_tri, **tol)
    else:
        import numpy as np
        np.testing.assert_allclose(c_ref.numpy(), c_tri.numpy(), **tol)
        np.testing.assert_allclose(da_ref.numpy(), da_tri.numpy(), **tol)
        np.testing.assert_allclose(db_ref.numpy(), db_tri.numpy(), **tol)


configs = [
    (64, 1871),
    (16, 256),
    (32, 576),
    (128, 2511),
]


@pytest.mark.parametrize("is_dense", [False, True])
@pytest.mark.parametrize("BLOCK, WIDTH", configs)
def test_softmax(BLOCK, WIDTH, is_dense, device, Z=2, H=2, is_causal=True, scale=0.4):
    # set seed
    if HAS_TORCH:
        torch.random.manual_seed(0)
    else:
        paddle.seed(0)
    
    Z, H, M, N = 2, 3, WIDTH, WIDTH
    # initialize layout
    # make sure each row has at least one non-zero element
    if HAS_TORCH:
        layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
        if is_dense:
            layout[:] = 1
        else:
            layout[1, 2, :] = 0
            layout[1, :, 1] = 0
    else:
        layout = paddle.randint(0, 2, (H, M // BLOCK, N // BLOCK))
        if is_dense:
            layout[:] = 1
        else:
            layout[1, 2, :] = 0
            layout[1, :, 1] = 0
    
    # initialize data
    a_shape = (Z, H, M, N)
    a_ref, a_tri = make_pair(a_shape)
    dout_ref, dout_tri = make_pair(a_shape)
    
    # compute [torch/paddle]
    a_ref = mask_tensor(a_ref, layout, BLOCK, value=float("-inf"))
    
    if HAS_TORCH:
        a_ref.retain_grad()
        at_mask = torch.ones((M, N), device=device)
        if is_causal:
            at_mask = torch.tril(at_mask)
        M = at_mask[None, None, :, :] + torch.zeros_like(a_ref)
        a_ref[M == 0] = float("-inf")
        out_ref = torch.softmax(a_ref * scale, -1)
        out_ref.backward(dout_ref)
    else:
        at_mask = paddle.ones((M, N))
        if is_causal:
            at_mask = paddle.tril(at_mask)
        M = at_mask.unsqueeze(0).unsqueeze(0) + paddle.zeros_like(a_ref)
        a_ref[M == 0] = float("-inf")
        a_ref = a_ref.clone().detach()
        a_ref.stop_gradient = False
        out_ref = paddle.nn.functional.softmax(a_ref * scale, axis=-1)
        out_ref.backward(dout_ref)
    
    out_ref = sparsify_tensor(out_ref, layout, BLOCK)
    da_ref = sparsify_tensor(a_ref.grad, layout, BLOCK)
    
    # compute [triton]
    a_tri = sparsify_tensor(a_tri, layout, BLOCK)
    if HAS_TORCH:
        a_tri.retain_grad()
    else:
        a_tri = a_tri.clone().detach()
        a_tri.stop_gradient = False
    
    dout_tri = sparsify_tensor(dout_tri, layout, BLOCK)
    op = triton.ops.blocksparse.softmax(layout, BLOCK, device=device, is_dense=is_dense)
    out_tri = op(a_tri, scale=scale, is_causal=is_causal)
    out_tri.backward(dout_tri)
    da_tri = a_tri.grad
    
    # compare
    if HAS_TORCH:
        torch.testing.assert_close(out_tri, out_ref, equal_nan=True)
        torch.testing.assert_close(da_tri, da_ref, equal_nan=True)
    else:
        paddle.allclose(out_tri, out_ref, equal_nan=True)
        paddle.allclose(da_tri, da_ref, equal_nan=True)


@pytest.mark.parametrize("block", [64, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16 if HAS_TORCH else 'float32', torch.float32 if HAS_TORCH else 'float16'])
def test_attention_fwd_bwd(
    block,
    dtype,
    device,
    input_scale=1.0,
    scale=1 / 8.0,
    n_ctx=256,
    batch_size=2,
    n_heads=2,
):
    if HAS_TORCH:
        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            pytest.skip("Only test tl.dot() on devices with sm >= 70")
    else:
        # PaddlePaddle 设备能力检查
        try:
            device_props = paddle.device.cuda.get_device_properties()
            if device_props.major < 7:
                pytest.skip("Only test tl.dot() on devices with sm >= 70")
        except:
            pass

    # inputs
    qkv_shape = (batch_size, n_heads, n_ctx, 64)
    if HAS_TORCH:
        qkvs = [
            torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)
        ]
    else:
        def make_leaf(shape, dtype, scale=1.0):
            t = (scale * paddle.randn(shape)).astype(dtype)
            t.stop_gradient = False
            return t
        qkvs = [make_leaf(qkv_shape, dtype, input_scale) for _ in range(3)]        
    # Triton:
    n_blocks = n_ctx // block
    if HAS_TORCH:
        layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
        query, key, value = [x.clone() for x in qkvs]
        query.retain_grad()
        key.retain_grad()
        value.retain_grad()
    else:
        layout = paddle.tril(paddle.ones([n_heads, n_blocks, n_blocks], dtype='int64'))
        query, key, value = [x.clone().detach() for x in qkvs]
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        
        query.clear_gradient()
        key.clear_gradient()
        value.clear_gradient()
    
    attn_out = triton_attention(layout, block, query=query, key=key, value=value, scale=scale)
    # ad hoc loss
    if HAS_TORCH:
        loss = (attn_out**2).mean()
        loss.backward()
        grads = [query.grad, key.grad, value.grad]
    else:
        loss = (attn_out**2).mean()
        loss.backward()
        assert query.is_leaf is True # query.is_leaf is True have gard
        grads = [query.grad, key.grad, value.grad]

    # Torch/Paddle version:
    if HAS_TORCH:
        torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
        attn_mask = torch.ones([n_ctx, n_ctx], device=device, dtype=dtype)
        attn_mask = torch.tril(attn_mask, diagonal=0)
        attn_mask = 1e6 * (-1 + (attn_mask.reshape((1, 1, n_ctx, n_ctx)).cuda()))
        torch_q.retain_grad()
        torch_k.retain_grad()
        torch_v.retain_grad()
        scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)
        # ad hoc loss
        torch_loss = (torch_attn_out**2).mean()
        torch_loss.backward()
        torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]
    else:
        paddle_q, paddle_k, paddle_v = [x.clone().detach() for x in qkvs]
        attn_mask = paddle.ones([n_ctx, n_ctx], dtype=dtype)
        attn_mask = paddle.tril(attn_mask, diagonal=0)
        attn_mask = 1e6 * (-1 + (attn_mask.reshape([1, 1, n_ctx, n_ctx])))
        paddle_q.stop_gradient = False
        paddle_k.stop_gradient = False
        paddle_v.stop_gradient = False
        
        query.clear_gradient()
        key.clear_gradient()
        value.clear_gradient()
        scores = scale * paddle.einsum("bhsd,bhtd->bhst", paddle_q, paddle_k)
        scores = scores + attn_mask
        probs = paddle.nn.functional.softmax(scores, axis=-1)
        paddle_attn_out = paddle.einsum("bhst,bhtd->bhsd", probs, paddle_v)
        # ad hoc loss
        paddle_loss = (paddle_attn_out**2).mean()
        paddle_loss.backward()
        paddle_grads = [paddle_q.grad, paddle_k.grad, paddle_v.grad]

    # comparison
    if HAS_TORCH:
        # print(f"Triton loss {loss} and torch loss {torch_loss}.  Also checking grads...")
        torch.testing.assert_close(loss, torch_loss, atol=1e-3, rtol=0)

        # Bigger tolerance for AMD MI200 devices.
        # MI200 devices use reduced precision fp16 and bf16 and flush input and
        # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
        tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}
        for g1, g2 in zip(grads, torch_grads):
            torch.testing.assert_close(g1, g2, **tol)
    else:
        paddle.allclose(loss, paddle_loss, atol=1e-3, rtol=0.)
        
        # Bigger tolerance for AMD MI200 devices.
        tol = {'atol': 1e-3, 'rtol': 0} if is_hip_mi200() else {}
        for g1, g2 in zip(grads, paddle_grads):
            paddle.allclose(g1, g2, **tol)


@pytest.mark.parametrize("block", [16, 32, 64])
def triton_attention(
    layout,
    block: int,
    query,
    key,
    value,
    scale: float,
):
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True,
                                                      device=value.device if HAS_TORCH else value.place)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False,
                                                      device=value.device if HAS_TORCH else value.place)
    sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device if HAS_TORCH else value.place)

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, is_causal=True)
    a = sparse_dot_dsd_nn(w, value)
    return a
