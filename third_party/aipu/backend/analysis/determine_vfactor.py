from mlir import ir


def determine_vectorization_factor(module, target_bitwidth=256, debug=False):
    """
    Determine the vectorization factor for affine loops in the module,
    preparing for the affine_vectorize pass.

    Args:
        module: The Triton module to analyze
        target_bitwidth: Target vector register bit width (default: 256 for AIPU_X2)

    Returns:
        int: Vectorization factor (1 if no affine.for found,
             otherwise target_bitwidth/min_dtype_width)
    """
    min_width = target_bitwidth
    skip_vectorize = False

    def _get_width(value):
        elem_type = (value.type.element_type if hasattr(value.type, 'element_type') else value.type)
        elem_width = 32 if isinstance(elem_type, ir.IndexType) else elem_type.width
        # Except bool dtype
        if elem_width == 1:
            return None
        return elem_width

    def walk_callback(op):
        nonlocal min_width, skip_vectorize
        if op.name == "affine.for":
            all_ops = (_op for region in op.regions for block in region.blocks for _op in block.operations)
            for _op in all_ops:
                items = _op.results or _op.operands
                for item in items:
                    if width := _get_width(item):
                        min_width = min(min_width, width)

        # Current affine-super-vectorize cannot deal with these op.
        if op.name in ("arith.mulsi_extended", "arith.mului_extended"):
            skip_vectorize = True

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback, ir.WalkOrder.PRE_ORDER)

    # If no affine.for found or no valid types found , vfactor=1
    vfactor = target_bitwidth // min_width
    if debug:
        print(f"[Debug]: Recommended vectorization factor: {vfactor}")
    return vfactor if not skip_vectorize else 1
