import pytest
import torch

from triton.tools.tensor_descriptor import (
    TENSOR_DESCRIPTOR_STRIDE_ALIGNMENT_VERSION,
    TensorDescriptor,
)


def test_tensor_descriptor_default_stride_alignment_is_unchanged():
    tensor = torch.empty((2, 538), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="strides must be 16-byte aligned"):
        TensorDescriptor.from_tensor(tensor, [1, 2])


def test_tensor_descriptor_accepts_explicit_four_byte_stride_alignment():
    assert TENSOR_DESCRIPTOR_STRIDE_ALIGNMENT_VERSION >= 1
    tensor = torch.empty((2, 538), dtype=torch.bfloat16)

    descriptor = TensorDescriptor.from_tensor(tensor, [1, 2], stride_alignment=4)

    assert descriptor.stride_alignment == 4
    assert tuple(descriptor.strides) == tensor.stride()


def test_tensor_descriptor_four_byte_stride_alignment_rejects_two_byte_stride():
    tensor = torch.empty((2, 537), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="strides must be 4-byte aligned"):
        TensorDescriptor.from_tensor(tensor, [1, 2], stride_alignment=4)


@pytest.mark.parametrize("stride_alignment", [False, 0, -4, 1, 2, 3])
def test_tensor_descriptor_rejects_invalid_stride_alignment(stride_alignment):
    tensor = torch.empty((2, 16), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="stride_alignment"):
        TensorDescriptor.from_tensor(tensor, [1, 2], stride_alignment=stride_alignment)


def test_interpreter_preserves_explicit_stride_alignment():
    from triton.runtime.interpreter import GridExecutor

    tensor = torch.empty((2, 538), dtype=torch.bfloat16)
    descriptor = TensorDescriptor.from_tensor(tensor, [1, 2], stride_alignment=4)
    executor = GridExecutor(lambda: None, [], (1, ))

    (host_descriptor, ), _ = executor._init_args_hst((descriptor, ), {})

    assert host_descriptor.stride_alignment == 4
    assert tuple(host_descriptor.strides) == tensor.stride()
