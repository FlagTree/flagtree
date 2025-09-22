import numpy as np
#import tvm
#from tvm import tir, ir
#from tvm.script.parser import tir as T
#from tvm.compass.dsl import BuildManager, script as S
from mlir import ir as mlir_ir
from mlir.dialects import func


def gen_cumsum(dtype, axis, shape):
    assert axis == 0, f"Only support axis == 0 but got {axis}."
    assert len(shape) == 1, f"Only support 1d cumsum but got {shape}."
    length = shape[0]

    @S.prim_func
    def cumsum(inp: S.ptr(dtype), out: S.ptr(dtype)):
        out[0] = inp[0]
        for i in range(1, length):
            out[i] = out[i - 1] + inp[i]

    return T.prim_func(cumsum.py_func, check_well_formed=False).with_attr("tir.is_entry_func", False)


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
    -1: "local",
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


def _get_vload_vstore_mask(op, buffer, lanes_value):
    mask = op.mask
    assert mask is None, "Currently only support op.mask is None."
    if len(buffer.shape) == 1:
        lanes_buffer = buffer.shape[0]
        if lanes_buffer != -1 and lanes_buffer < lanes_value:
            mask = f"{lanes_buffer}T"
    return mask


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


def _compute_strides(shape):
    """compute tensor strides with given shape"""
    strides = [1]
    for dim_size in reversed(shape[1:]):
        strides.append(strides[-1] * dim_size)
    return list(reversed(strides))


def _compute_dynamic_strides(shape, strides):
    """compute tensor dynamic strides with given shape"""
    assert len(shape) == len(strides)
    assert not mlir_ir.ShapedType.is_dynamic_size(strides[-1])

    for i in range(len(strides) - 2, -1, -1):
        if mlir_ir.ShapedType.is_dynamic_size(strides[i]):
            strides[i] = shape[i + 1] * strides[i + 1]

    return strides


def _is_scalar_type(ty):
    return isinstance(ty, (mlir_ir.IndexType, mlir_ir.IntegerType, mlir_ir.FloatType))


def _is_vector_type(ty):
    return isinstance(ty, mlir_ir.VectorType)


def _is_pointer_type(ty):
    return isinstance(ty, (mlir_ir.MemRefType, mlir_ir.UnrankedMemRefType))


def _get_type(value):
    ty = value.type

    if _is_scalar_type(ty):
        return _convert_scalar_type(ty)
    elif _is_vector_type(ty):
        return _convert_vector_type(ty)
    elif _is_pointer_type(ty):
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


def _get_buffer(ptr):
    if isinstance(ptr, tir.Buffer):
        return ptr
    return _get_buffer(ptr.buffer)


def _eliminate_one_in_shape(src_shape, src_strides, dst_strides):
    if len(src_shape) <= 2:
        return src_shape, src_strides, dst_strides

    new_src_shape = []
    new_src_strides = []
    new_dst_strides = []

    for i, size in enumerate(src_shape):
        if size == 1:
            continue
        new_src_shape.append(size)
        new_src_strides.append(src_strides[i])
        new_dst_strides.append(dst_strides[i])

    return new_src_shape, new_src_strides, new_dst_strides


class CodeGenerator():

    def __init__(self, mod) -> None:
        self.mod = mod
        self.ib = tir.ir_builder.create()
        # Dictionary to map MLIR values to corresponding TVM TIR variables or buffers.
        # Keys are MLIR values, and values are TVM TIR variables or buffers.
        self.mlir_to_tir_mapping = {}
        self.name_idx = 0
        self.ir_mod = tvm.IRModule()
        self.scope_stack = []
        self.gridx_var = None
        self.while_cond = None
        self.after_args = None
        self.yeild_args = None

    def _get_associated_dtype(self, op):
        # Find the associated_dtype of the bool arg.
        try:
            owner = op.operands[0].owner
            associated_dtype = self.get_operand(owner, 0).dtype
            while associated_dtype.startswith("bool"):
                owner = owner.operands[0].owner
                associated_dtype = self.get_operand(owner, 0).dtype
        except IndexError:
            associated_dtype = "int32"
        finally:
            return associated_dtype

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

    def get_list_value(self, list_value):
        return [self.get_or_create_var(i) for i in list_value]

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

    def get_static_or_dynamic_value(self, static_v, dynamic_v):
        dynamic_v = list(dynamic_v)
        return [
            self.get_or_create_var(dynamic_v.pop(0)) if mlir_ir.ShapedType.is_dynamic_size(v) else v for v in static_v
        ]

    def get_offsets_sizes_strides(self, op):
        offsets = self.get_static_or_dynamic_value(op.static_offsets, op.offsets)
        sizes = self.get_static_or_dynamic_value(op.static_sizes, op.sizes)
        strides = self.get_static_or_dynamic_value(op.static_strides, op.strides)

        return offsets, sizes, strides

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
        elif op_name == "memref.cast":
            self.gen_memref_cast(op)
        elif op_name == "memref.expand_shape":
            self.gen_memref_expand_shape(op)
        elif op_name == "memref.collapse_shape":
            self.gen_memref_collapse_shape(op)
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
            if _is_vector_type(op.operands[0].type):
                self.gen_binary(op, S.vmod)
            else:
                self.gen_binary(op, T.Mod)
        elif op_name in ("arith.remf", ):
            remainder = lambda x, y: T.call_extern(_get_type(op.result), "remainder", x, y)
            self.gen_binary(op, remainder)
        elif op_name == "arith.cmpi":
            self.gen_binary(op, _CMPI_MAPPING[op.predicate.value])
        elif op_name == "arith.cmpf":
            if op.predicate.value == 13:
                result = op.result
                if _is_vector_type(result.type):
                    self.gen_binary(op, S.vcneq)
                else:
                    not_equal = lambda x, y: T.call_extern(_get_type(result), "isnotequal", x, y)
                    self.gen_binary(op, not_equal)
            else:
                self.gen_binary(op, _CMPF_MAPPING[op.predicate.value])
        elif op_name in ("arith.sitofp", "arith.extf", "arith.truncf", "arith.extsi", "arith.extui", "arith.trunci",
                         "arith.uitofp", "arith.fptosi", "arith.bitcast"):
            self.gen_arith_cast(op)
        elif op_name == "arith.select":
            self.gen_select(op)
        elif op_name in ("arith.shrsi", "arith.shrui"):
            self.gen_binary(op, T.shift_right)
        elif op_name == "arith.shli":
            self.gen_binary(op, T.shift_left)
        elif op_name in ("arith.mulsi_extended", "arith.mului_extended"):
            self.gen_arith_mul_extended(op)
        # Math Dialect
        elif op_name == "math.atan2":
            self.gen_binary(op, S.atan2)
        elif op_name == "math.powf":
            self.gen_binary(op, S.pow)
        elif op_name == "math.rsqrt":
            self.gen_unary(op, S.rsqrt)
        elif op_name == "math.tanh":
            self.gen_unary(op, S.tanh)
        elif op_name == "math.exp2":
            self.gen_unary(op, S.exp2)
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
        elif op_name == "math.floor":
            self.gen_unary(op, S.floor)
        elif op_name == "math.trunc":
            trunc = lambda x: T.call_extern(_get_type(op.result), "trunc", x)
            self.gen_unary(op, trunc)
        # MathExt Dialect
        elif op_name == "mathext.fmod":
            fmod = lambda x, y: T.call_extern(_get_type(op.result), "fmod", x, y)
            self.gen_binary(op, fmod)
        elif op_name == "mathext.div_rz":
            self.ib.emit(tir.call_extern("void", "__vset_rounding_mode_rtz"))
            self.gen_binary(op, T.Div)
            self.ib.emit(tir.call_extern("void", "__vset_rounding_mode_rtn"))
        elif op_name == "math.isinf":
            self.gen_is_like(op, S.isinf)
        elif op_name == "math.isfinite":
            self.gen_is_like(op, S.isfinite)
        elif op_name == "math.isnan":
            self.gen_is_like(op, S.isnan)
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
            pf_args = [self.get_or_create_var(value) for value in op.operands]
            self.yeild_args = [arg.addr_of(0) if isinstance(arg, tir.Buffer) else arg for arg in pf_args]
        # Vector Dialect
        elif op_name == "vector.transfer_read":
            self.gen_vload(op)
        elif op_name == "vector.transfer_write":
            self.gen_vstore(op)
        elif op_name == "vector.broadcast":
            self.gen_vbcast(op)
        # TritonTilingExt Dialect
        elif op_name == "ttx.cumsum":
            inp = self.get_operand(op, 0)
            out = self.get_operand(op, 1)
            dtype = inp.dtype
            axis = op.attributes["axis"].value
            shape = _get_shape(op.operands[0])

            inp_type = ir.PrimType(dtype)
            ret_type = ir.PrimType("void")
            cumsum_gv = ir.GlobalVar("cumsum", ir.FuncType([inp_type, inp_type], ret_type))
            self.ir_mod[cumsum_gv] = gen_cumsum(dtype, axis, shape)
            self.ib.emit(tir.call_tir(cumsum_gv, inp.addr_of(0), out.addr_of(0)))
        # Others
        elif op_name == "builtin.module":
            pass
        elif op_name == "builtin.unrealized_conversion_cast":
            result = op.result
            arg0 = self.get_operand(op, 0)
            if "from_pass_convert_bool_arg" in op.attributes and op.attributes["from_pass_convert_bool_arg"]:
                self.mlir_to_tir_mapping[result] = arg0.as_ptr(_convert_scalar_type(result.type.element_type))
            else:
                self.mlir_to_tir_mapping[result] = arg0
        elif op_name == "tt.bitcast":
            self.mlir_to_tir_mapping[op.result] = self.get_operand(op, 0).as_ptr("i8")
        else:
            raise RuntimeError(f"Unsupport op {op_name}.")

    def generate(self):
        self.mod.walk_mod(self.dispatch)
        return BuildManager().build(self.ir_mod)

    def gen_memref_reinterpret_cast(self, op):
        result = op.result
        arg = self.get_operand(op, 0)
        data = arg.base if isinstance(arg, tir.Pointer) else arg.data
        dtype = _get_type(result).element_type.dtype
        offsets, sizes, strides = self.get_offsets_sizes_strides(op)

        buffer = T.Buffer(sizes, elem_offset=offsets[0], data=data, dtype=dtype, strides=strides)
        self.mlir_to_tir_mapping[result] = buffer

    def gen_memref_load(self, op):
        result = op.result
        buffer = _get_buffer(self.get_operand(op, 0))
        indices = self.get_list_value(op.indices) or [0]

        self.emit_let(T.BufferLoad(buffer, indices), result)

    def gen_memref_store(self, op):
        value = self.get_operand(op, 0)
        buffer = _get_buffer(self.get_operand(op, 1))
        indices = self.get_list_value(op.indices) or [0]

        self.ib.emit(tir.BufferStore(buffer, value, indices))

    def gen_memref_alloc(self, op):
        result = op.result
        dtype = _get_type(result).element_type.dtype
        shape = _get_shape(result)
        scope_value = result.type.memory_space.value if result.type.memory_space else -1

        if scope_value == 11:
            self.emit_let(S.alloc_events(1), op.result)
        else:
            buf = self.ib.allocate(dtype, shape, scope=_MEMORY_SCOPE_MAPPING[scope_value])
            buf._buffer.strides = _compute_strides(shape)
            self.mlir_to_tir_mapping[result] = buf._buffer

    def gen_dma_start(self, op):
        #  currently, we only support one event, skip stride
        src = _get_buffer(self.get_operand(op, 0))
        dst = _get_buffer(self.get_operand(op, 2))
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
        src_shape, src_strides, dst_strides = _eliminate_one_in_shape(src.shape, src.strides, dst.strides)
        src = src.addr_of(0) if isinstance(src, tir.Buffer) else src
        dst = dst.addr_of(0) if isinstance(dst, tir.Buffer) else dst

        if len(src_shape) == 1:
            dma_copy = S.dma_copy(dst, src, src_shape[0])
            self.ib.emit(dma_copy)
        elif len(src_shape) == 2:
            src_ptr = tir.Pointer(src.dtype, src.scope, name=self.create_var_name())
            self.ib.emit(lambda x: tir.LetStmt(src_ptr.base, src, x))
            dst_ptr = tir.Pointer(dst.dtype, dst.scope, name=self.create_var_name())
            self.ib.emit(lambda x: tir.LetStmt(dst_ptr.base, dst, x))

            with self.ib.if_scope(src_strides[1] < 0):
                with self.ib.for_range(0, src_shape[0], name=self.create_var_name()) as i:
                    with self.ib.for_range(0, src_shape[1], name=self.create_var_name()) as j:
                        src_offset = i * src_strides[0] + j * src_strides[1]
                        dst_offset = i * dst_strides[0] + j * dst_strides[1]
                        self.ib.emit(tir.BufferStore(dst_ptr.buffer, src_ptr.buffer[src_offset], [dst_offset]))
            with self.ib.else_scope():
                with self.ib.if_scope(src_strides[1] == 0):
                    with self.ib.for_range(0, src_shape[0]):
                        dma_memset = S.dma_memset(dst_ptr, src_ptr[0], src_shape[1])
                        self.ib.emit(dma_memset)
                        self.ib.emit(tir.reassign(src_ptr.base, src_ptr + src_strides[0]))
                        self.ib.emit(tir.reassign(dst_ptr.base, dst_ptr + dst_strides[0]))
                with self.ib.else_scope():
                    with self.ib.if_scope(src_strides[0] == 0):
                        with self.ib.for_range(0, src_shape[0]):
                            dma_copy = S.dma_copy(dst_ptr, src_ptr, src_shape[1])
                            self.ib.emit(dma_copy)
                            self.ib.emit(tir.reassign(src_ptr.base, src_ptr + src_strides[0]))
                            self.ib.emit(tir.reassign(dst_ptr.base, dst_ptr + dst_strides[0]))
                    with self.ib.else_scope():
                        dma_copy = S.dma_copy(dst, src, width=src_shape[1], src_stride=src_strides[0],
                                              times=src_shape[0], dst_stride=dst_strides[0])
                        self.ib.emit(dma_copy)
        elif len(src_shape) == 3:
            src_ptr = tir.Pointer(src.dtype, src.scope, name=self.create_var_name())
            self.ib.emit(lambda x: tir.LetStmt(src_ptr.base, src, x))
            dst_ptr = tir.Pointer(dst.dtype, dst.scope, name=self.create_var_name())
            self.ib.emit(lambda x: tir.LetStmt(dst_ptr.base, dst, x))

            with self.ib.if_scope(src_strides[2] < 0):
                with self.ib.for_range(0, src_shape[0], name=self.create_var_name()) as i:
                    with self.ib.for_range(0, src_shape[1], name=self.create_var_name()) as j:
                        with self.ib.for_range(0, src_shape[2], name=self.create_var_name()) as k:
                            src_offset = i * src_strides[0] + j * src_strides[1] + k * src_strides[2]
                            dst_offset = i * dst_strides[0] + j * dst_strides[1] + k * dst_strides[2]
                            self.ib.emit(tir.BufferStore(dst_ptr.buffer, src_ptr.buffer[src_offset], [dst_offset]))
            with self.ib.else_scope():
                # Scalar broadcast scenario.
                with self.ib.if_scope(src_strides[2] == 0):
                    with self.ib.for_range(0, src_shape[0]):
                        temp_dst_ptr = tir.Pointer(dst.dtype, dst.scope, name=self.create_var_name())
                        self.ib.emit(lambda x: tir.LetStmt(temp_dst_ptr.base, dst_ptr, x))
                        with self.ib.for_range(0, src_shape[1]):
                            # Here need to use the origin src pointer base.
                            dma_memset = S.dma_memset(temp_dst_ptr, src[-src.offset], src_shape[2])
                            self.ib.emit(dma_memset)
                            self.ib.emit(tir.reassign(temp_dst_ptr.base, temp_dst_ptr + dst_strides[1]))
                        self.ib.emit(tir.reassign(dst_ptr.base, dst_ptr + dst_strides[0]))
                with self.ib.else_scope():
                    with self.ib.for_range(0, src_shape[0]):
                        dma_copy = S.dma_copy(dst_ptr, src_ptr, width=src_shape[2], src_stride=src_strides[1],
                                              times=src_shape[1], dst_stride=dst_strides[1])
                        self.ib.emit(dma_copy)
                        self.ib.emit(tir.reassign(src_ptr.base, src_ptr + src_strides[0]))
                        self.ib.emit(tir.reassign(dst_ptr.base, dst_ptr + dst_strides[0]))
        else:
            raise RuntimeError(f"Only suport 1d/2d/3d DMA copy, but got shape={src.shape}.")

    def gen_memref_subview(self, op):
        result = op.result
        buffer = _get_buffer(self.get_operand(op, 0))
        offsets, sizes, strides = self.get_offsets_sizes_strides(op)

        buf_strides = buffer.strides
        if len(strides) != len(buf_strides):
            buf_strides = _compute_strides(buffer.shape)
        update_strides = [buf_strides[i] * strides[i] for i in range(len(strides))]
        update_offset = buffer.elem_offset + sum(off * stride for off, stride in zip(offsets, buf_strides))

        subview = T.Buffer(sizes, elem_offset=update_offset, data=buffer.data, dtype=buffer.dtype,
                           strides=update_strides)
        self.mlir_to_tir_mapping[result] = subview

    def gen_memref_expand_shape(self, op):
        result = op.result
        inp_buf = _get_buffer(self.get_operand(op, 0))
        out_shape = self.get_static_or_dynamic_value(op.static_output_shape, op.output_shape)
        strides = result.type.get_strides_and_offset()[0] or [1]

        expanded_buffer = T.Buffer(out_shape, elem_offset=inp_buf.elem_offset, data=inp_buf.data, dtype=inp_buf.dtype,
                                   strides=_compute_dynamic_strides(out_shape, strides))
        self.mlir_to_tir_mapping[result] = expanded_buffer

    def gen_memref_collapse_shape(self, op):
        result = op.result
        inp_buf = _get_buffer(self.get_operand(op, 0))
        out_shape = _get_shape(result)
        strides = result.type.get_strides_and_offset()[0] or [1]

        out_buffer = T.Buffer(out_shape, elem_offset=inp_buf.elem_offset, data=inp_buf.data, dtype=inp_buf.dtype,
                              strides=_compute_dynamic_strides(out_shape, strides))
        self.mlir_to_tir_mapping[result] = out_buffer

    def gen_memref_cast(self, op):
        result = op.result
        arg = self.get_operand(op, 0)
        dtype = _get_type(result).element_type.dtype

        self.mlir_to_tir_mapping[result] = arg.as_ptr(dtype)

    def gen_arith_constant(self, op):

        def _create_const_expr(op):
            ty = op.result.type
            dtype = _get_type(op.result)

            if _is_scalar_type(ty):
                value = bool(op.value) if dtype == "bool" else op.literal_value
                return tir.const(value, dtype)
            if _is_vector_type(ty):
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

        self.emit_let(S.cast(arg0, "int32"), result)

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

        self.emit_let(tir.Select(arg0, arg1, arg2), result)

    def gen_arith_cast(self, op):
        result = op.result
        dtype = _get_type(op.result)
        arg0 = self.get_operand(op, 0)

        if dtype.startswith("bool") and _is_vector_type(op.result.type):
            # arg_type >bool
            arg0 = S.vcneq(arg0, S.cast(0, arg0.dtype))
        elif arg0.dtype.startswith("bool") and _is_vector_type(op.operands[0].type):
            # bool->associate_type->ret_type
            associated_dtype = self._get_associated_dtype(op)
            arg0 = T.Select(arg0, S.cast(1, associated_dtype), S.cast(0, associated_dtype))
            # if associate_type is float16 -> float32
            if associated_dtype.startswith("float16"):
                arg0 = S.cast(arg0, "float32")
            arg0 = S.cast(arg0, _get_type(result))
        else:
            arg0 = S.cast(arg0, _get_type(result))

        self.emit_let(arg0, result)

    def gen_arith_mul_extended(self, op):
        mull, mulh = op.results
        arg0 = self.get_operand(op, 0)
        arg1 = self.get_operand(op, 1)

        self.emit_let(T.Mul(arg0, arg1), mull)
        mulh_func = "mul_hi"
        if isinstance(mulh.type, mlir_ir.VectorType):
            mulh_func = "vmul_hi"
        self.emit_let(T.call_extern(_get_type(mulh), mulh_func, arg0, arg1), mulh)

    def gen_unary(self, op, method):
        result = op.result
        arg0 = self.get_operand(op, 0)

        self.emit_let(method(arg0), result)

    def gen_is_like(self, op, method):
        result = op.result
        arg0 = self.get_operand(op, 0)

        value = method(arg0)
        var_name = self.create_var_name()
        let_var = tir.Var(var_name, dtype=f"boolx{arg0.dtype.lanes}")
        self.ib.emit(lambda x: tir.LetStmt(let_var, value, x))
        self.mlir_to_tir_mapping[result] = let_var

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

            gv = ir.GlobalVar(func_name)
            prim_func = tir.PrimFunc(args, self.ib.get())
            self.ir_mod[gv] = prim_func.with_attr("global_symbol", func_name).with_attr("tir.is_entry_func", True)

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
            for result in op.results:
                res_type = _get_type(result)
                if isinstance(res_type, ir.PointerType):
                    dtype = res_type.element_type.dtype
                    handle_var = tir.Var(self.create_var_name(), ir.PointerType(ir.PrimType(dtype), "local"))
                    empty_var = self.ib.let(self.create_var_name(), S.i32(0))
                    self.ib.emit(lambda x: tir.LetStmt(handle_var, empty_var.addr, x))
                    self.mlir_to_tir_mapping[result] = handle_var
                else:
                    self.emit_let(S.cast(0, res_type), result)

            if_scope = self.ib.if_scope(cond)
            self.enter_scope(if_scope)

        # Else branch
        if stage.is_after_region(0):
            for i, result in enumerate(op.results):
                reassign_var = self.get_or_create_var(result)
                self.ib.emit(tir.reassign(reassign_var, self.yeild_args[i]))

            self.exit_scope()
            else_scope = self.ib.else_scope()
            self.enter_scope(else_scope)

        # Finish
        if stage.is_after_all_regions():
            for i, result in enumerate(op.results):
                reassign_var = self.get_or_create_var(result)
                self.ib.emit(tir.reassign(reassign_var, self.yeild_args[i]))

            self.exit_scope()
            for result in op.results:
                res_type = _get_type(result)
                if isinstance(res_type, ir.PointerType):
                    handle_var = self.get_or_create_var(result)
                    init_ptr = tir.Pointer(res_type.element_type.dtype, "local", handle_var)
                    self.mlir_to_tir_mapping[result] = init_ptr
                else:
                    self.mlir_to_tir_mapping[result] = self.get_or_create_var(result)

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
        # TODO(CP-22941): support other permutation_map cases

        result = op.result
        lanes = result.type.shape[0]
        buffer = _get_buffer(self.get_operand(op, 0))
        indices = self.get_list_value(op.indices)
        mask = _get_vload_vstore_mask(op, buffer, lanes)

        def _is_broadcast(permutation_map):
            """ permutation_map:  (d0, d1, ...) ->(0) is for broadcast """
            affinemap = permutation_map.value
            return (len(affinemap.results) == 1 and str(affinemap.results[0]) == '0')

        if _is_broadcast(op.permutation_map):
            indices = indices or [0]
            value = T.BufferLoad(buffer, indices)
            self.emit_let(S.vbcast(S.cast(value, value.dtype), lanes=lanes), result)
        else:
            indices = indices or 0
            self.emit_let(S.vload(buffer.addr_of(indices), lanes=lanes, mask=mask), result)

    def gen_vstore(self, op):
        value = self.get_operand(op, 0)
        lanes_value = op.operands[0].type.shape[0]
        buffer = _get_buffer(self.get_operand(op, 1))
        indices = self.get_list_value(op.indices) or 0

        mask = _get_vload_vstore_mask(op, buffer, lanes_value)
        self.ib.emit(S.vstore(value, buffer.addr_of(indices), mask=mask))

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
