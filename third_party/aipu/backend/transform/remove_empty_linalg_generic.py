from mlir import ir


def remove_empty_linalg_generic(module, ctx):

    def walk_callback(op):
        if op.name == "linalg.generic":
            if list(op.attributes["indexing_maps"]) == [] and list(op.attributes["iterator_types"]) == []:
                num_ops = len(op.regions[0].blocks[0].operations)
                ops = op.regions[0].blocks[0].operations
                msg = "last op of linalg.generic should be linalg.yeild."
                assert ops[num_ops - 1].name == "linalg.yield", msg

                ops_to_move = []
                for i in range(num_ops - 1):
                    ops_to_move.append(ops[i])

                with ctx, op.location:
                    for _op in ops_to_move:
                        _op.move_before(op)
                    op.erase()
        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback)
