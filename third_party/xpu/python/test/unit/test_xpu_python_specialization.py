from pathlib import Path

import triton.tools.tensor_descriptor as tensor_descriptor
from triton.tools.tensor_descriptor import TensorDescriptor


def test_tensor_descriptor_uses_xpu_spec_and_accepts_unaligned_layout():
    assert Path(tensor_descriptor.__file__).parts[-7:] == (
        "triton",
        "backends",
        "xpu",
        "spec",
        "triton",
        "tools",
        "tensor_descriptor.py",
    )

    descriptor = TensorDescriptor(object(), [3, 513], [513, 1], [16, 32])
    assert descriptor.shape == [3, 513]
