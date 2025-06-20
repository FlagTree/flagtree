from mlir import ir


def get_linalg_generic_size(module):
    """
    Get the size of linalg generic.

    Args:
        module: The Triton module to analyze.

    Returns:
        int: The size of linalg generic.
    """
    size = 0

    def walk_callback(op):
        nonlocal size
        if op.name == "linalg.generic":
            size += 1

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback, ir.WalkOrder.PRE_ORDER)

    return size
