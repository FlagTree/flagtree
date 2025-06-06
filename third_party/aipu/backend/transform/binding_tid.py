from mlir import ir
from mlir.dialects import arith, func


def binding_tid(module, ctx):
    """
    Binding tid to the third-to-last parameter.

    Args:
        module: The mlir module to analyze
        ctx: The mlir ctx

    Returns:
        None
    """

    def walk_callback(op):
        if op.name == "func.func":
            block = op.regions[0].blocks[0]
            gridx = block.arguments[-3]
            with ctx, op.location, ir.InsertionPoint.at_block_begin(block):
                i32 = ir.IntegerType.get_signless(32)
                local_size = func.call([i32], "local_size", [])
                local_id = func.call([i32], "local_id", [])
                var_a = arith.muli(gridx, local_size)
                var_b = arith.addi(var_a, local_id)

                gridx.replace_all_uses_except(var_b, var_a.owner)

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback)
