from mlir import ir
from mlir.dialects import arith
import numpy as np


def _is_in_range(val, dtype):
    iinfo = np.iinfo(dtype)
    return iinfo.min <= val <= iinfo.max


def _get_cononical_type(val):
    if _is_in_range(val, np.int32):
        return ir.IntegerType.get_signed(32)
    elif _is_in_range(val, np.uint32):
        return ir.IntegerType.get_unsigned(32)
    else:
        return None


def canonical_const_dtype(module, ctx):
    """
    Canonical i64 value in arith.constant to 32-bit value.

    Args:
        module: The mlir module to analyze
        ctx: The mlir ctx

    Returns:
        None
    """

    def walk_callback(op):
        if op.name == "arith.constant":
            op_type = op.result.type
            if isinstance(op_type, ir.IntegerType) and op_type.width == 64:
                with ctx, op.location, ir.InsertionPoint.at_block_begin(module.body):
                    const_value = op.attributes["value"].value
                    type = _get_cononical_type(const_value)
                    if type is None:
                        raise RuntimeError(f"Cannot support i64 value {const_value} in arith.constant.")

                    new_const = arith.constant(type, const_value)
                    op.result.replace_all_uses_with(new_const)
                    op.erase()

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback)
