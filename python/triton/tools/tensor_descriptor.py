# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape

# Increment this when the public stride-alignment override contract changes.
# Backends can use the marker to retain a safe fallback with older wheels.
TENSOR_DESCRIPTOR_STRIDE_ALIGNMENT_VERSION = 1


@dataclass
class TensorDescriptor:
    """Host-side tensor descriptor.

    Descriptor strides remain 16-byte aligned by default.  Callers may relax
    the host-side validation to a power-of-two alignment of at least four
    bytes when the target supports it.  This field controls validation only;
    it does not change a backend's descriptor encoding or hardware contract.
    """
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"
    stride_alignment: int = 16

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        ty = type(self.base)
        if ty.__name__ not in ("FakeTensor", "FunctionalTensor"):
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        assert isinstance(self.stride_alignment, int) and not isinstance(self.stride_alignment, bool), \
            "stride_alignment must be an integer"
        assert self.stride_alignment >= 4 and (self.stride_alignment & (self.stride_alignment - 1)) == 0, \
            "stride_alignment must be a power of two greater than or equal to 4"
        elem_bytes = self.base.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % self.stride_alignment == 0, \
                f"strides must be {self.stride_alignment}-byte aligned"
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero", *, stride_alignment=16):
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, padding, stride_alignment)
