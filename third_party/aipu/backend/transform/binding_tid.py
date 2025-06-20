from mlir import ir
from mlir.dialects import arith, func, scf


def binding_tid(module, ctx):
    """
    Binding tid to the third-to-last parameter.
    Triton convention: [gridx_size, gridy_size, gridz_size, gridx_idx, gridy_idx, gridz_idx].
    Aipu_driver set args: [gridx_size, gridy_size, gridz_size, loop_tec_idx, 0, 0].

    Recover gridx_idx, gridy_idx in func_body:

    grid_flat_idx = loop_tec_idx * tec_num + tec_id
    gridx_idx = grid_flat_idx % gridx_size
    gridy_idx = (grid_flat_idx // gridx_size) % gridy_size
    gridz_idx = grid_flat_idx // (gridx_size * gridy_size)

    Args:
        module: The mlir module to analyze
        ctx: The mlir ctx

    Returns:
        None
    """

    def walk_callback(op):
        if op.name == "func.func":
            block = op.regions[0].blocks[0]
            gridx, gridy, gridz = block.arguments[-3:]
            gridx_size, gridy_size, gridz_size = block.arguments[-6:-3]
            with ctx, op.location, ir.InsertionPoint.at_block_begin(block):
                i32 = ir.IntegerType.get_signless(32)
                local_size = func.call([i32], "local_size", [])
                local_id = func.call([i32], "local_id", [])
                mul_tid = arith.muli(gridx, local_size)
                flatten_tid = arith.addi(mul_tid, local_id)

                xy_size = arith.muli(gridx_size, gridy_size)
                total_size = arith.muli(xy_size, gridz_size)
                cond = arith.cmpi(arith.CmpIPredicate.sge, flatten_tid, total_size)
                if_op = scf.IfOp(cond)
                with ir.InsertionPoint(if_op.then_block):
                    func.ReturnOp([])

                new_gridx = arith.remsi(flatten_tid, gridx_size)
                new_gridy = arith.remsi(arith.divsi(flatten_tid, gridx_size), gridy_size)
                new_gridz = arith.divsi(flatten_tid, xy_size)

                gridx.replace_all_uses_except(new_gridx, mul_tid.owner)
                gridy.replace_all_uses_with(new_gridy)
                gridz.replace_all_uses_with(new_gridz)

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback)
