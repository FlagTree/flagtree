"""isort:skip_file"""
# Import order is significant here.

from . import math
from . import extra
from .standard import (
    argmax,
    argmin,
    cdiv,
    cumprod,
    cumsum,
    flip,
    interleave,
    max,
    min,
    ravel,
    sigmoid,
    softmax,
    sort,
    sum,
    swizzle2d,
    xor_sum,
    zeros,
    zeros_like,
)
from .core import (
    PropagateNan,
    TRITON_MAX_TENSOR_NUMEL,
    _experimental_descriptor_load,
    _experimental_descriptor_store,
    advance,
    arange,
    associative_scan,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
    atomic_mul,
    bfloat16,
    block_type,
    broadcast,
    broadcast_to,
    cat,
    cast,
    clamp,
    const,
    const_pointer_type,
    constexpr,
    debug_barrier,
    device_assert,
    device_print,
    dot,
    dtype,
    expand_dims,
    float16,
    float32,
    float64,
    float8e4b15,
    float8e4nv,
    float8e4b8,
    float8e5,
    float8e5b16,
    full,
    function_type,
    histogram,
    inline_asm_elementwise,
    int1,
    int16,
    int32,
    int64,
    int8,
    join,
    load,
    make_block_ptr,
    max_constancy,
    max_contiguous,
    maximum,
    minimum,
    multiple_of,
    num_programs,
    permute,
    pi32_t,
    pointer_type,
    program_id,
    range,
    reduce,
    reshape,
    split,
    static_assert,
    static_print,
    static_range,
    store,
    tensor,
    trans,
    uint16,
    uint32,
    uint64,
    uint8,
    view,
    void,
    where,
)
from .math import (umulhi, exp, exp2, fma, log, log2, cos, rsqrt, sin, sqrt, sqrt_rn, abs, fdiv, div_rn, erf, floor,
                   ceil)
from .random import (
    pair_uniform_to_normal,
    philox,
    philox_impl,
    rand,
    rand4x,
    randint,
    randint4x,
    randn,
    randn4x,
    uint_to_uniform_float,
)

__all__ = [
    "PropagateNan",
    "TRITON_MAX_TENSOR_NUMEL",
    "_experimental_descriptor_load",
    "_experimental_descriptor_store",
    "abs",
    "advance",
    "arange",
    "argmax",
    "argmin",
    "associative_scan",
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "atomic_mul",
    "bfloat16",
    "block_type",
    "broadcast",
    "broadcast_to",
    "builtin",
    "cat",
    "cast",
    "cdiv",
    "ceil",
    "clamp",
    "const",
    "const_pointer_type",
    "constexpr",
    "cos",
    "cumprod",
    "cumsum",
    "debug_barrier",
    "device_assert",
    "device_print",
    "div_rn",
    "dot",
    "dtype",
    "erf",
    "exp",
    "exp2",
    "expand_dims",
    "extra",
    "fdiv",
    "flip",
    "float16",
    "float32",
    "float64",
    "float8e4b15",
    "float8e4nv",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
    "floor",
    "fma",
    "full",
    "function_type",
    "histogram",
    "inline_asm_elementwise",
    "interleave",
    "int1",
    "int16",
    "int32",
    "int64",
    "int8",
    "ir",
    "join",
    "load",
    "log",
    "log2",
    "make_block_ptr",
    "math",
    "max",
    "max_constancy",
    "max_contiguous",
    "maximum",
    "min",
    "minimum",
    "multiple_of",
    "num_programs",
    "pair_uniform_to_normal",
    "permute",
    "philox",
    "philox_impl",
    "pi32_t",
    "pointer_type",
    "program_id",
    "rand",
    "rand4x",
    "randint",
    "randint4x",
    "randn",
    "randn4x",
    "range",
    "ravel",
    "reduce",
    "reshape",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sort",
    "split",
    "sqrt",
    "sqrt_rn",
    "static_assert",
    "static_print",
    "static_range",
    "store",
    "sum",
    "swizzle2d",
    "tensor",
    "trans",
    "triton",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "uint_to_uniform_float",
    "umulhi",
    "view",
    "void",
    "where",
    "xor_sum",
    "zeros",
    "zeros_like",
]


def str_to_ty(name):
    if name[0] == "*":
        name = name[1:]
        if name[0] == "k":
            name = name[1:]
            ty = str_to_ty(name)
            return const_pointer_type(ty)
        ty = str_to_ty(name)
        return pointer_type(ty)
    tys = {
        "fp8e4nv": float8e4nv,
        "fp8e4b8": float8e4b8,
        "fp8e5": float8e5,
        "fp8e5b16": float8e5b16,
        "fp8e4b15": float8e4b15,
        "fp16": float16,
        "bf16": bfloat16,
        "fp32": float32,
        "fp64": float64,
        "i1": int1,
        "i8": int8,
        "i16": int16,
        "i32": int32,
        "i64": int64,
        "u1": int1,
        "u8": uint8,
        "u16": uint16,
        "u32": uint32,
        "u64": uint64,
        "B": int1,
    }
    return tys[name]
