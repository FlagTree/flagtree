from functools import wraps
from typing import List
from triton.language import core
from triton.language.math import _add_math_1arg_docstr, _add_math_2arg_docstr, _add_math_3arg_docstr
from triton.language import semantic

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                arg_type = arg.type.scalar.name
                if hasattr(arg, 'was_bool_to_int8') and arg.was_bool_to_int8:
                    # In Triton, int1 maps to the boolean type
                    arg_type = 'int1'
                if arg_type not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg_type}")
            return fn(*args, **kwargs)

        return check

    return wrapper


@core.builtin
@_check_dtype(dtypes=["int32", "uint32"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_umulhi(x.handle, y.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp2(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log2(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_cos(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sin(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
@core._tensor_member_fn
def sqrt_rn(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_precise_sqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_rsqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_precise_divf(x.handle, y.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_erf(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def tanh(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_tanh(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_floor(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_ceil(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    z = semantic.to_tensor(z, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    z, x = core.binary_op_type_legalization(z, x, _builder)
    z, y = core.binary_op_type_legalization(z, y, _builder)
    return core.tensor(_builder.create_fma(x.handle, y.handle, z.handle), x.type)

