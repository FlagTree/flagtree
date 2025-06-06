import numpy as np
from tvm import tir, ir, aipu
from tvm.script.parser import tir as T
from tvm.aipu import script as S
from mlir import ir as mlir_ir
from mlir.dialects import func

_CMPI_MAPPING = {
    0: T.EQ,
    1: T.NE,
    2: T.LT,
    3: T.LE,
    4: T.GT,
    5: T.GE,
    6: T.LT,
    7: T.LE,
    8: T.GT,
    9: T.GE,
}
_CMPF_MAPPING = {
    1: T.EQ,
    2: T.GT,
    3: T.GE,
    4: T.LT,
    5: T.LE,
    6: T.NE,
    8: T.EQ,
    9: T.GT,
    10: T.GE,
    11: T.LT,
    12: T.LE,
    13: T.NE,
}
_MEMORY_SCOPE_MAPPING = {
    4: "lsram",
    8: "shared",
    11: "alloc_event",
}


class WalkStage:

    def __init__(self, op):
        self.num_regions = len(op.regions)
        self.next_region = 0

    def is_before_all_regions(self):
        return self.next_region == 0

    def is_before_region(self, region):
        return self.next_region == region

    def is_after_region(self, region):
        return self.next_region == region + 1

    def is_after_all_regions(self):
        return self.next_region == self.num_regions

    def advance(self):
        self.next_region += 1

    def get_next_region(self):
        return self.next_region


def _convert_scalar_type(type):
    """convert from mlir_type to tvm_type_str"""
    if isinstance(type, mlir_ir.IndexType):
        return "int32"
    if isinstance(type, mlir_ir.IntegerType):
        sign_str = "u" if type.is_unsigned else ""
        width = min(32, type.width)
        if width == 1:
            return "bool"
        return f"{sign_str}int{width}"
    if isinstance(type, mlir_ir.FloatType):
        return f"float{type.width}"
    raise RuntimeError(f"not scalar type {type}")


def _convert_vector_type(type):
    """convert from mlir_type to tvm_type_str"""
    if isinstance(type, mlir_ir.VectorType):
        assert type.rank == 1
        e_dtype = _convert_scalar_type(type.element_type)
        vtype = f"{e_dtype}x{type.shape[0]}"
        return vtype
    raise RuntimeError(f"not scalar type {type}")


def _get_type(value):
    ty = value.type

    if isinstance(ty, (mlir_ir.IndexType, mlir_ir.IntegerType, mlir_ir.FloatType)):
        return _convert_scalar_type(ty)
    elif isinstance(ty, mlir_ir.VectorType):
        return _convert_vector_type(ty)
    elif isinstance(ty, (mlir_ir.MemRefType, mlir_ir.UnrankedMemRefType)):
        e_dtype = _convert_scalar_type(ty.element_type)
        return ir.PointerType(ir.PrimType(e_dtype))

    raise RuntimeError(f"Cannot parse type {ty}")


def _get_shape(value):
    ty = value.type
    if isinstance(ty, mlir_ir.ShapedType):
        if ty.rank > 0:
            return ty.shape
        return [1]

    raise RuntimeError(f"Cannot parse shape {ty}")


class CodeGenerator():

    def __init__(self, mod) -> None:
        self.mod = mod
        self.ib = tir.ir_builder.create()
        # Dictionary to map MLIR values to corresponding TVM TIR variables or buffers.
        # Keys are MLIR values, and values are TVM TIR variables or buffers.
        self.mlir_to_tir_mapping = {}
        self.name_idx = 0
        self.prim_func = None
        self.scope_stack = []
        self.gridx_var = None
        self.while_cond = None
        self.after_args = None
        self.yeild_args = None

    def create_var_name(self):
        var_name = "var_" + str(self.name_idx)
        self.name_idx += 1
        return var_name

    def emit_let(self, value, related_value):
        var_name = self.create_var_name()
        let_var = self.ib.let(var_name, value)
        self.mlir_to_tir_mapping[related_value] = let_var

    def get_operand(self, op, idx):
        return self.get_or_create_var(op.operands[idx])

    def get_or_create_var(self, value):
        if value in self.mlir_to_tir_mapping:
            return self.mlir_to_tir_mapping[value]

        value_type = _get_type(value)
        var = T.Var(self.create_var_name(), value_type)
        if isinstance(value_type, ir.PointerType):
            var = tir.Pointer(value_type.element_type.dtype, "global", name=self.create_var_name())
        else:
            var = T.Var(self.create_var_name(), value_type)
        self.mlir_to_tir_mapping[value] = var
        return var

    def for_range(self, begin, end, step, kind="serial"):
        self.ib._seq_stack.append([])

        loop_var = T.Var(self.create_var_name(), "int32")
        extent = end if begin == 0 else (end - begin)
        annotations = {"step": step}

        def _exit_cb():
            if kind == "serial":
                kind_id = tir.ForKind.SERIAL
            elif kind == "parallel":
                kind_id = tir.ForKind.PARALLEL
            elif kind == "vectorize":
                kind_id = tir.ForKind.VECTORIZED
            elif kind == "unroll":
                kind_id = tir.ForKind.UNROLLED
            else:
                raise ValueError("Unknown kind")
            self.ib.emit(tir.For(
                loop_var,
                begin,
                extent,
                kind_id,
                self.ib._pop_seq(),
                annotations=annotations,
            ))

        return tir.ir_builder.WithScope(loop_var, _exit_cb)

    def enter_scope(self, scope):
        assert isinstance(scope, tir.ir_builder.WithScope)
        self.scope_stack.append(scope)
        return scope.__enter__()

    def exit_scope(self):
        self.scope_stack.pop().__exit__(None, None, None)

    def dispatch(self, op, stage):
        op_name = "func.func" if isinstance(op, func.FuncOp) else op.name
        # Memref Dialect
        if op_name == "memref.reinterpret_cast":
            self.gen_memref_reinterpret_cast(op)
        elif op_name == "memref.load":
            self.gen_memref_load(op)
        elif op_name == "memref.store":
            self.gen_memref_store(op)
        elif op_name == "memref.alloc":
            self.gen_memref_alloc(op)
        elif op_name == "memref.copy":
            self.gen_memref_copy(op)
        elif op_name == "memref.subview":
            self.gen_memref_subview(op)
        elif op_name == "memref.dma_start":
            self.gen_dma_start(op)
        elif op_name == "memref.dma_wait":
            self.gen_dma_wait(op)
        # Arith Dialect
        elif op_name == "arith.constant":
            self.gen_arith_constant(op)
        elif op_name == "arith.index_cast":
            self.gen_arith_index_cast(op)
        elif op_name in ("arith.addf", "arith.addi"):
            self.gen_binary(op, T.Add)
        elif op_name in ("arith.subf", "arith.subi"):
            self.gen_binary(op, T.Sub)
        elif op_name in ("arith.muli", "arith.mulf"):
            self.gen_binary(op, T.Mul)
        elif op_name in ("arith.minsi", "arith.minnumf"):
            self.gen_binary(op, T.Min)
        elif op_name in ("arith.maxsi", "arith.maxnumf", "arith.maximumf"):
            self.gen_binary(op, T.Max)
        elif op_name in ("arith.divf", "arith.divi", "arith.divsi"):
            self.gen_binary(op, T.Div)
        elif op_name in ("arith.andi", "arith.andf"):
            self.gen_binary(op, T.bitwise_and)
        elif op_name in ("arith.ori", "arith.orf"):
            self.gen_binary(op, T.bitwise_or)
        elif op_name in ("arith.xori", "arith.xorf"):
            self.gen_binary(op, T.bitwise_xor)
        elif op_name in ("arith.remsi", "arith.remui"):
            self.gen_binary(op, T.Mod)
        elif op_name == "arith.cmpi":
            self.gen_binary(op, _CMPI_MAPPING[op.predicate.value])
        elif op_name == "arith.cmpf":
            self.gen_binary(op, _CMPF_MAPPING[op.predicate.value])
        elif op_name in ("arith.sitofp", "arith.extf", "arith.truncf", "arith.extsi", "arith.extui", "arith.trunci",
                         "arith.uitofp"):
            self.gen_arith_cast(op)
        elif op_name == "arith.select":
            self.gen_select(op)
        # Math Dialect
        elif op_name == "math.powf":
            self.gen_binary(op, S.pow)
        elif op_name == "math.tanh":
            self.gen_unary(op, S.tanh)
        elif op_name == "math.exp":
            self.gen_unary(op, S.exp)
        elif op_name == "math.absf":
            self.gen_unary(op, S.abs)
        elif op_name == "math.sin":
            self.gen_unary(op, S.sin)
        elif op_name == "math.cos":
            self.gen_unary(op, S.cos)
        elif op_name == "math.sqrt":
            self.gen_unary(op, S.sqrt)
        elif op_name == "math.erf":
            self.gen_unary(op, S.erf)
        elif op_name == "math.log":
            self.gen_unary(op, S.log)
        # Func Dialect
        elif op_name == "func.return":
            self.gen_func_return(op)
        elif op_name == "func.func":
            self.gen_func_func(op, stage)
        elif op_name == "func.call":
            self.gen_func_call(op)
        # Scf Dialect
        elif op_name == "scf.for":
            self.gen_scf_for(op, stage)
        elif op_name == "scf.if":
            self.gen_scf_if(op, stage)
        elif op_name == "scf.while":
            self.gen_scf_while(op, stage)
        elif op_name == "scf.condition":
            self.while_cond = self.get_operand(op, 0)
            self.after_args = [self.get_or_create_var(arg) for arg in op.args]
        elif op_name == "scf.yield":
            self.yeild_args = [self.get_or_create_var(value) for value in op.operands]
        # Vector Dialect
        elif op_name == "vector.transfer_read":
            self.gen_vload(op)
        elif op_name == "vector.transfer_write":
            self.gen_vstore(op)
        elif op_name == "vector.broadcast":
            self.gen_vbcast(op)
        # Others
        elif op_name == "builtin.module":
            pass
        elif op_name == "builtin.unrealized_conversion_cast":
            self.mlir_to_tir_mapping[op.result] = self.get_operand(op, 0)
        elif op_name == "tt.bitcast":
            self.mlir_to_tir_mapping[op.result] = self.get_operand(op, 0).as_ptr("i8")
        else:
            raise RuntimeError(f"Unsupport op {op_name}.")

    def generate(self):
        self.mod.walk_mod(self.dispatch)
        bm = aipu.tir.BuildManager()
        return bm.build(self.prim_func)

    def gen_memref_reinterpret_cast(self, op):
        result = op.result
        arg = self.get_operand(op, 0)
        dtype = _get_type(result).element_type.dtype
        offset = 0
        if len(op.operands) == 2:
            offset = self.get_operand(op, 1)

        buffer = T.Buffer((-1, ), elem_offset=offset, data=arg.base, dtype=dtype)
        self.mlir_to_tir_mapping[result] = buffer

    def gen_memref_load(self, op):
        result = op.result
        buffer = self.get_operand(op, 0)
        index = [0]
        if len(op.operands) >= 2:
            index = [self.get_operand(op, i) for i in range(1, len(op.operands))]
        self.emit_let(T.BufferLoad(buffer, index), result)

    def gen_memref_store(self, op):
        value = self.get_operand(op, 0)
        buffer = self.get_operand(op, 1)

        index = [0]
        if len(op.operands) >= 3:
            index = [self.get_operand(op, i) for i in range(2, len(op.operands))]
        self.ib.emit(tir.BufferStore(buffer, value, index))

    def gen_memref_alloc(self, op):
        result = op.result
        dtype = _get_type(result).element_type.dtype
        shape = _get_shape(result)
        # set default memory space: lsram
        scope_value = result.type.memory_space.value if result.type.memory_space else 4
        if scope_value == 11:
            event = S.alloc_events(1)
            self.mlir_to_tir_mapping[result] = event
        else:
            buf = self.ib.allocate(dtype, shape, scope=_MEMORY_SCOPE_MAPPING[scope_value])
            self.mlir_to_tir_mapping[result] = buf._buffer

    def gen_dma_start(self, op):
        #  currently, we only support one event, skip stride
        src = self.get_operand(op, 0)
        src = src.buffer if isinstance(src, tir.Pointer) else src
        dst = self.get_operand(op, 2)
        dst = dst.buffer if isinstance(dst, tir.Pointer) else dst
        src_index = self.get_operand(op, 1)
        dst_index = self.get_operand(op, 3)
        num_elements = self.get_operand(op, 4)
        event = self.get_operand(op, 5)
        self.ib.emit(S.async_dma_copy(dst.addr_of(dst_index), src.addr_of(src_index), num_elements, event=event))

    def gen_dma_wait(self, op):
        # currently, we only support one event
        event = self.get_operand(op, 0)
        self.ib.emit(S.wait_events(event))

    def gen_memref_copy(self, op):
        src = self.get_operand(op, 0)
        dst = self.get_operand(op, 1)
        width = src.shape[0]

        dma_copy = S.dma_copy(dst, src, width)
        self.ib.emit(dma_copy)

    def gen_memref_subview(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)
        buffer = arg0.buffer if isinstance(arg0, tir.Pointer) else arg0
        size = self.get_operand(op, 1)

        subview = T.Buffer(size, elem_offset=buffer.elem_offset, data=buffer.data, dtype=buffer.dtype)
        self.mlir_to_tir_mapping[result] = subview

    def gen_arith_constant(self, op):

        def _create_const_expr(op):
            ty = op.result.type
            dtype = _get_type(op.result)
            # scalar
            if isinstance(ty, (mlir_ir.IndexType, mlir_ir.IntegerType, mlir_ir.FloatType)):
                value = bool(op.value) if dtype == "bool" else op.literal_value
                return tir.const(value, dtype)
            # vector
            if isinstance(ty, mlir_ir.VectorType):
                const_value = op.value.maybe_downcast()
                # For FP16, the C++ interface __get_item__ do not have a proper implementation.
                # So here use np.array to directly using its raw data.
                if isinstance(ty.element_type, mlir_ir.F16Type):
                    const_array = np.array(const_value)
                else:
                    const_array = list(const_value)

                return S.cast(const_array, dtype)
            raise RuntimeError(f"Cannot parse constant {op}")

        expr = _create_const_expr(op)
        self.emit_let(expr, op.result)

    def gen_arith_index_cast(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(T.Cast("int32", arg0), result)

    def gen_binary(self, op, method):
        result = op.result
        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)

        self.emit_let(method(arg0, arg1), result)

    def gen_select(self, op):
        #cond, true_value, false_value
        result = op.result

        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)
        arg2 = self.get_operand(op, 2)
        if isinstance(result.type, mlir_ir.VectorType):
            self.emit_let(S.vsel(arg1, arg2, mask=arg0), result)
        else:
            self.emit_let(tir.Select(arg0, arg1, arg2), result)

    def gen_arith_cast(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)

        if arg0.dtype.startswith("bool"):
            # Find the associated_dtype of the bool arg.
            owner = op.operands[0].owner
            associated_dtype = self.get_operand(owner, 0).dtype
            while associated_dtype.startswith("bool"):
                owner = owner.operands[0].owner
                associated_dtype = self.get_operand(owner, 0).dtype

            vsel = S.vsel(S.cast(1, associated_dtype), 0, arg0)
            self.emit_let(S.cast(vsel, _get_type(result)), result)
        else:
            self.emit_let(S.cast(arg0, _get_type(result)), result)

    def gen_unary(self, op, method):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(method(arg0), result)

    def gen_func_return(self, op):
        self.ib.emit(T.ret(None))

    def gen_func_func(self, op, stage):
        if stage.is_after_all_regions():
            func_name = op.name.value
            block = op.regions[0].blocks[0]
            arg_nums = len(block.arguments)

            args = []
            for i in range(arg_nums):
                arg = block.arguments[i]
                var = self.get_or_create_var(arg)
                if isinstance(var, tir.Pointer):
                    args.append(var.base)
                else:
                    args.append(var)

            self.prim_func = tir.PrimFunc(args, self.ib.get()).with_attr("global_symbol", func_name)

    def gen_func_call(self, op):
        result = op.result
        func_name = op.callee.value

        if func_name == "local_size":
            self.emit_let(S.get_local_size(), result)
        elif func_name == "local_id":
            self.emit_let(S.get_local_id(), result)
        else:
            raise RuntimeError(f"Unsupport func call {func_name}.")

    def gen_scf_for(self, op, stage):
        if stage.is_before_all_regions():
            begin = self.get_operand(op, 0)
            end = self.get_operand(op, 1)
            step = self.get_operand(op, 2)

            block = op.regions[0].blocks[0]
            for i, arg in enumerate(block.arguments):
                if i == 0:
                    loop_iter = arg
                else:
                    self.mlir_to_tir_mapping[arg] = self.get_operand(op, i + 2)

            for_range = self.for_range(begin, end, step)
            loop_var = self.enter_scope(for_range)
            self.mlir_to_tir_mapping[loop_iter] = loop_var

        if stage.is_after_all_regions():
            self.exit_scope()
            for i, value in enumerate(op.results):
                self.mlir_to_tir_mapping[value] = self.yeild_args[i]

    def gen_scf_if(self, op, stage):
        # If branch
        if stage.is_before_all_regions():
            cond = self.get_operand(op, 0)

            if_scope = self.ib.if_scope(cond)
            self.enter_scope(if_scope)
        # Else branch
        if stage.is_after_region(0):
            self.exit_scope()
            else_scope = self.ib.else_scope()
            self.enter_scope(else_scope)
        # Finish
        if stage.is_after_all_regions():
            self.exit_scope()

    def gen_scf_while(self, op, stage):
        if stage.is_before_all_regions():
            init_var = self.get_or_create_var(op.inits[0])
            self.mlir_to_tir_mapping[op.before.blocks[0].arguments[0]] = init_var
            self.mlir_to_tir_mapping[op.result] = init_var

        # Before branch
        if stage.is_after_region(0):
            while_scope = self.ib.while_loop(self.while_cond)
            self.enter_scope(while_scope)

            # mapping condition iter_args to after_args
            after_block = op.after.blocks[0]
            for i, arg in enumerate(after_block.arguments):
                self.mlir_to_tir_mapping[arg] = self.after_args[i]

        # After branch
        if stage.is_after_region(1):
            init_var = self.get_or_create_var(op.inits[0])
            self.ib.emit(tir.reassign(init_var, self.yeild_args[0]))

            while_cond = self.while_cond
            self.mod.walk_region(op.before, self.dispatch)
            self.ib.emit(tir.reassign(while_cond, self.while_cond))

        # Finish
        if stage.is_after_all_regions():
            self.exit_scope()

    def gen_vload(self, op):
        result = op.result
        arg0 = self.get_operand(op, 0)
        buffer = arg0.buffer if isinstance(arg0, tir.Pointer) else arg0
        index = 0
        if len(op.operands) >= 2:
            index = self.get_operand(op, 1)

        self.emit_let(S.vload(buffer.addr_of(index), lanes=result.type.shape[0]), result)

    def gen_vstore(self, op):
        value = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)
        buffer = arg1.buffer if isinstance(arg1, tir.Pointer) else arg1
        index = 0
        if len(op.operands) >= 3:
            index = self.get_operand(op, 2)

        self.ib.emit(S.vstore(value, buffer.addr_of(index)))

    def gen_vbcast(self, op):
        result = op.result
        value = self.get_operand(op, 0)
        dtype = result.type
        self.emit_let(S.vbcast(S.cast(value, value.dtype), lanes=dtype.shape[0]), result)


class AIPUModule:

    def __init__(self, mod):
        # wrap triton module to mlir module
        self.mod = mod

    def walk_region(self, region, callback):
        for block in region.blocks:
            self.walk_block(block, callback)

    def walk_block(self, block, callback):
        for nested_op in block.operations:
            self.walk_op(nested_op, callback)

    def walk_op(self, op, callback):
        # operation walk
        stage = WalkStage(op)
        regions = op.regions
        for region in regions:
            callback(op, stage)
            stage.advance()
            self.walk_region(region, callback)
        callback(op, stage)

    def walk_mod(self, dispatch):
        # module walk entry
        self.walk_op(self.mod.operation, dispatch)


def codegenAIPU(mod):
    mod = AIPUModule(mod)
    generator = CodeGenerator(mod)
    return generator.generate()
