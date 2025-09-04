from mlir import ir
from mlir.dialects import arith


def get_i1():
    return ir.IntegerType.get_signless(1)


def get_i8():
    return ir.IntegerType.get_signless(8)


def is_i1(type):
    return (isinstance(type, ir.IntegerType) and type.width == 1)


def is_i8(type):
    return (isinstance(type, ir.IntegerType) and type.width == 8)


def convert_memref_i1_i8(module, ctx):

    def walk_callback(op):
        if op.name == "memref.alloc" and isinstance(op.result.type, ir.MemRefType):
            etype = op.result.type.element_type
            if is_i1(etype):
                with ctx, op.location:
                    shape = op.result.type.shape
                    new_type = ir.MemRefType.get(shape, get_i8())
                    op.result.set_type(new_type)

        if op.name == "builtin.unrealized_conversion_cast":
            if "from_pass_convert_bool_arg" in op.attributes and op.attributes["from_pass_convert_bool_arg"]:
                with ctx, op.location:
                    op.result.replace_all_uses_except(op.operands[0], op.operands[0].owner)
                    op.erase()

        if op.name in ("memref.reinterpret_cast", "memref.collapse_shape", "memref.expand_shape"):
            if is_i8(op.operands[0].type.element_type) and is_i1(op.result.type.element_type):
                with ctx, op.location:
                    shape = op.result.type.shape
                    new_type = ir.MemRefType.get(shape, get_i8())
                    op.result.set_type(new_type)

        if op.name in ("memref.load", "affine.load"):
            base = op.operands[0]
            if is_i1(op.result.type) and is_i8(base.type.element_type):
                with ctx, op.location, ir.InsertionPoint(op):
                    op.result.set_type(get_i8())
                    new_value = arith.trunci(get_i1(), op.result)
                    op.result.replace_all_uses_except(new_value, new_value.owner)
                    op.move_before(new_value.owner)

        if op.name in ("memref.store", "affine.store"):
            value = op.operands[0]
            base = op.operands[1]
            if is_i1(value.type) and is_i8(base.type.element_type):
                with ctx, op.location, ir.InsertionPoint(op):
                    new_value = arith.extui(get_i8(), op.operands[0])
                    op.operands[0] = new_value

        if op.name == "scf.if":
            for result in op.results:
                ty = result.type
                if isinstance(ty, ir.MemRefType) and is_i1(ty.element_type):
                    with ctx, op.location:
                        new_type = ir.MemRefType.get(ty.shape, get_i8())
                        op.result.set_type(new_type)

        return ir.WalkResult.ADVANCE

    module.operation.walk(walk_callback)
