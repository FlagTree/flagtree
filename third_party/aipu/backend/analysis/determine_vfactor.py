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

    def walk_callback(op):
        nonlocal min_width
        if op.name == "affine.for":
            all_ops = (_op for region in op.regions for block in region.blocks for _op in block.operations)
            for _op in all_ops:
                for result in _op.results:
                    elem_type = (result.type.element_type if hasattr(result.type, 'element_type') else result.type)
                    elem_width = 32 if isinstance(elem_type, ir.IndexType) else elem_type.width
                    min_width = min(min_width, elem_width)

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback, ir.WalkOrder.PRE_ORDER)

    # If no affine.for found or no valid types found , vfactor=1
    vfactor = target_bitwidth // min_width
    if debug:
        print(f"[Debug]: Recommended vectorization factor: {vfactor}")
    return vfactor
