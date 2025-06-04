import os
from typing import List
from triton.language.core import _tensor_member_fn, builtin, _constexpr_to_value, tensor, constexpr
from triton.language.core import dtype as real_dtype
from triton.language import semantic as real_semantic
from triton._C.libtriton import ir
from triton.language.core import float32
# from triton.language.core import _unwrap_if_constexpr, _unwrap_shape
from . import semantic
# from ._utils import validate_block_shape

# class dtype(real_dtype):

#     def to_ir(self, builder: ir.builder) -> ir.type:
#         if self.name in ("uint8", "uint16", "uint32", "uint64"):
#             raise ValueError(f"type {self} not supported in this architecture for now.")

#         if self.name.startswith("fp8"):
#             if self.name not in builder.options.supported_fp8_dtypes:
#                 raise ValueError(f'type {self} not supported in this architecture. '
#                                  f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')
#             if self.name in builder.options.deprecated_fp8_dtypes:
#                 warn(f"{self.name} is deprecated in this architecture and will be removed in a future triton release")

#         if self.name == 'void':
#             return builder.get_void_ty()
#         elif self.name == 'int1':
#             return builder.get_int1_ty()
#         elif self.name in ('int8', 'uint8'):
#             return builder.get_int8_ty()
#         elif self.name in ('int16', 'uint16'):
#             return builder.get_int16_ty()
#         elif self.name in ('int32', 'uint32'):
#             return builder.get_int32_ty()
#         elif self.name in ('int64', 'uint64'):
#             return builder.get_int64_ty()
#         elif self.name == 'fp8e5':
#             return builder.get_fp8e5_ty()
#         elif self.name == 'fp8e5b16':
#             return builder.get_fp8e5b16_ty()
#         elif self.name == 'fp8e4nv':
#             return builder.get_fp8e4nv_ty()
#         elif self.name == 'fp8e4b8':
#             return builder.get_fp8e4b8_ty()
#         elif self.name == 'fp8e4b15':
#             return builder.get_fp8e4b15_ty()
#         elif self.name == 'fp16':
#             return builder.get_half_ty()
#         elif self.name == 'bf16':
#             return builder.get_bf16_ty()
#         elif self.name == 'fp32':
#             return builder.get_float_ty()
#         elif self.name == 'fp64':
#             return builder.get_double_ty()
#         raise ValueError(f'fail to convert {self} to ir type')

# class pointer_type(dtype):

#     def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
#         element_ty = _unwrap_if_constexpr(element_ty)
#         if not isinstance(element_ty, dtype):
#             raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
#         self.element_ty = element_ty
#         self.address_space = address_space
#         self.const = const
#         self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

#     def to_ir(self, builder: ir.builder):
#         return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.__str__()

#     def is_ptr(self):
#         return True

#     def is_const(self):
#         return self.const

#     def __eq__(self, other: pointer_type) -> bool:
#         if not isinstance(other, pointer_type):
#             return False
#         return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

#     def __ne__(self, other: pointer_type) -> bool:
#         return not self.__eq__(other)

#     @property
#     def scalar(self):
#         return self

# class block_type(dtype):

#     def __init__(self, element_ty: dtype, shape: List):
#         self.element_ty = element_ty

#         # Note that block_type's shape is a list of int
#         # while tensor's shape is a list of constexpr.

#         # shape can be empty ([]) when an input is a 0D tensor.
#         self.shape = _unwrap_shape(shape)
#         if not self.shape:
#             raise TypeError('0d block_type is forbidden')

#         self.numel = validate_block_shape(self.shape)
#         self.name = f'<{self.shape}, {self.element_ty}>'

#     def to_ir(self, builder: ir.builder) -> ir.block_type:
#         return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.__str__()

#     def is_block(self):
#         return True

#     def get_block_shapes(self) -> List[int]:
#         return self.shape

#     def __eq__(self, other: block_type) -> bool:
#         if not isinstance(other, block_type):
#             return False
#         return self.element_ty == other.element_ty and self.shape == other.shape

#     def __ne__(self, other: block_type) -> bool:
#         return not self.__eq__(other)

#     @property
#     def scalar(self):
#         return self.element_ty

# class function_type(dtype):

#     def __init__(self, ret_types: List[dtype], param_types: List[dtype]) -> None:
#         self.ret_types = ret_types
#         self.param_types = param_types

#     def __str__(self):
#         return f'fn ({self.param_types}) -> {self.ret_types}'

#     def to_ir(self, builder: ir.builder):
#         ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
#         ret_types = [ret_type.to_ir(builder) for ret_type in self.ret_types]
#         return builder.get_function_ty(ir_param_types, ret_types)


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32,
        _builder=None):
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    assert not allow_tf32, "allow_tf32 is deprecated, please use input_precision='hf32' on Ascend instead."
    if input_precision is None:
        supports_tf32 = _builder and "tf32" in _builder.options.allowed_dot_input_precisions
        default_precision = "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        input_precision = os.getenv("TRITON_F32_DEFAULT", default_precision)
    else:
        assert (input_precision not in [
            "tf32", "tf32x3"
        ]), "input_precision == tf32 or tf32x3 is invalid, please use input_precision='hf32' on Ascend instead."
    input_precision = _constexpr_to_value(input_precision)
    out_dtype = _constexpr_to_value(out_dtype)
    max_num_imprecise_acc = _constexpr_to_value(max_num_imprecise_acc)
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)


@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    """Gather from a tensor along a given dimension.
    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.gather(src, index, axis, _builder)


@_tensor_member_fn
@builtin
def insert(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert (len(ful.shape) > 0)
    assert (len(ful.shape) == len(sub.shape))
    new_offsets = [real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o for o in offsets]
    out = semantic.insert(ful, sub, new_offsets, sizes, strides, _builder)
    return out


@_tensor_member_fn
@builtin
def subview(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert (len(ful.shape) > 0)
    new_offsets = [real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o for o in offsets]
    sub = semantic.subview(ful, new_offsets, sizes, strides, _builder)
    return sub
