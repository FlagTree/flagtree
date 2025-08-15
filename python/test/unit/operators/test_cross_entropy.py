import pytest
HAS_TORCH = False
HAS_PADDLE = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    import paddle
    HAS_PADDLE = True

import triton
import triton.ops

@pytest.mark.parametrize("M, N, dtype, mode", [  #
    (M, N, dtype, mode)
    for M in [1024, 821]
    for N in [512, 857, 1871, 2089, 8573, 31000]
    for dtype in ['float16', 'float32']
    for mode in ['forward', 'backward']
])
def test_op(M, N, dtype, mode, device='cuda'):
    if HAS_TORCH:
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8 and dtype == "bfloat16":
            pytest.skip("Only test bfloat16 on devices with sm >= 80")
        dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
        dtype_t = dtype_map[dtype]

        # create inputs
        x = torch.randn(M, N, dtype=dtype_t, device=device, requires_grad=True)
        idx = 4 + torch.ones(M, dtype=torch.int64, device=device)
        # forward pass
        tt_y = triton.ops.cross_entropy(x, idx)
        th_y = torch.nn.CrossEntropyLoss(reduction="none")(x, idx)
        if mode == 'forward':
            torch.testing.assert_close(tt_y, th_y)

        elif mode == 'backward':
            dy = torch.randn_like(tt_y)
            tt_y.backward(dy)
            tt_dx = x.grad.clone()
            x.grad = None
            th_y.backward(dy)
            th_dx = x.grad.clone()
            if dtype == 'float16':
                torch.testing.assert_close(tt_dx, th_dx, rtol=1e-3, atol=1e-3)
            else:
                torch.testing.assert_close(tt_dx, th_dx)

    elif HAS_PADDLE:
        import paddle

        dtype_map = {'float16': 'float16', 'float32': 'float32'}
        dtype_p = dtype_map[dtype]

        # inputs
        x = paddle.randn([M, N], dtype=dtype_p)
        x.stop_gradient = False
        idx = 4 + paddle.ones([M], dtype='int64')

        # forward pass using paddle.nn.functional.cross_entropy
        tt_y = paddle.nn.functional.cross_entropy(
            input=x, label=idx, reduction='none', axis=-1
        )
        th_y = paddle.nn.functional.cross_entropy(
            input=x, label=idx, reduction='none', axis=-1
        )

        if mode == 'forward':
            assert paddle.allclose(tt_y, th_y)

        elif mode == 'backward':
            dy = paddle.randn_like(tt_y)
            tt_y.backward(dy)
            tt_dx = x.grad.clone()
            x.clear_gradient()
            th_y.backward(dy)
            th_dx = x.grad.clone()
            assert paddle.allclose(tt_dx, th_dx, atol=1e-3, rtol=0.)