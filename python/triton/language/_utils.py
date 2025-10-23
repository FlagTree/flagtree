from __future__ import annotations

from typing import List, TYPE_CHECKING, Any, Union, Dict

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]

TRITON_MAX_TENSOR_NUMEL = 1048576

def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


BITWIDTH_DICT: Dict[str, int] = {
    **{f"u{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"i{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"fp{n}": n
       for n in (16, 32, 64)},
    **{f"fp8{suffix}": 8
       for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")},
    "bf16": 16,
    "void": 0,
}


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]
