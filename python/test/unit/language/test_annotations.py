from __future__ import annotations
import pytest
import triton
import triton.language as tl

try:
    import torch
    HAS_TORCH = True
    HAS_PADDLE = False
except :
    import paddle
    HAS_TORCH = False
    HAS_PADDLE = True
def annotated_function(return_type=None, **arg_types):
    """A decorator to add annotations to a function."""
    def decorator(func):
        func.__annotations__ = {**arg_types, 'return': return_type}
        return func
    return decorator


# Test integer annotations
@pytest.mark.parametrize(("signed", "width"), [
    (signed, width) for signed in [False, True]\
                    for width in [8, 16, 32, 64]
] + [(False, 1)]
                         )
def test_int_annotation(signed, width, device):

    if HAS_TORCH:
        @triton.jit
        @annotated_function(X=torch.tensor, v=f"tl.{'' if signed else 'u'}int{width}")
        def _kernel(X, v):
            tl.store(X, v)

        h = _kernel[(1, )](torch.empty(1, device=device), 3)
        pfx = 'si' if signed else 'ui'
        assert f'%arg1: i{width}' in h.asm["ttir"]
        assert f'arith.{pfx}tofp' in h.asm["ttir"]

    elif HAS_PADDLE:
        @triton.jit
        @annotated_function(X=paddle.Tensor, v=f"tl.{'' if signed else 'u'}int{width}")
        def _kernel(X, v):
            tl.store(X, v)

        h = _kernel[(1, )](paddle.empty([1]), 3)
        pfx = 'si' if signed else 'ui'
        assert f'%arg1: i{width}' in h.asm["ttir"]
        assert f'arith.{pfx}tofp' in h.asm["ttir"]


# Test that unknown annotations do not emit an error
def test_unknown_annotation(device):

    if HAS_TORCH:
        @triton.jit
        def _kernel(X: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
            pass

        x = torch.empty(1, device=device)
        _kernel[(1, )](x, x.shape[0], 32)
        try:
            _kernel[(1, )](x.shape[0], x.shape[0], 32)
        except AttributeError:
            pass

    elif HAS_PADDLE:
        @triton.jit
        def _kernel(X: paddle.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
            pass

        x = paddle.empty([1])
        _kernel[(1, )](x, x.shape[0], 32)
        try:
            _kernel[(1, )](x.shape[0], x.shape[0], 32)
        except AttributeError:
            pass