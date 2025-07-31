import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os
import clang.cindex

# The `need_workspace_func` map records the number of workspaces required by different functions.
# The keys of the map are function names, and the values are dictionaries that map parameter types to the corresponding workspace sizes.
# For each function, the "default" key specifies the default workspace size, which applies when no specific parameter type is explicitly marked.
# If a parameter type (e.g., "f32") has a separate workspace size specified, it overrides the "default" value.
# If no specific parameter type is marked, the workspace size for that function defaults to the "default" value.
# For example, the map for 'fast_erf' indicates that 'fast_erf' requires 3 workspaces for the 'f32' type, while for 'fp16' or 'bf16', it requires 4 workspaces (the default value).
need_workspace_func = {
    'fast_erf': {
        "f32": 3,
        "default": 4,
    },
    'fast_tanh': {
        "default": 1,
        "f32": 2,
    },
    'fast_powi': {"default": 2},
    'ultra_silu': {
        "f32": 2,
        "default": 0,
    },
    'ultra_pow': {"default": 4},
    'ctz': {"default": 1},
    'ultra_gelu': {"default": 2},
    'fast_silu': {"default": 1},
}

special_funcs = [
    "philox", "ultra_silu_out", "ultra_silubp_out", "broadcast_ultra_silu_mul_out", "broadcast_ultra_silubp_mul_out",
    "sub_exp_mul", "philox_v2", "philox_v3", "fast_dividef", "ultra_pow"
]


def get_worspace_num(op_name, dtype):
    if op_name not in need_workspace_func:
        return 0
    num_map = need_workspace_func[op_name]
    if dtype not in num_map:
        return num_map['default']
    else:
        return num_map[dtype]


def parse_export_symbol(cn_hdr_path):

    def visit(node):
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            functions.append(node.spelling)
        else:
            for cursor in node.get_children():
                visit(cursor)

    index = clang.cindex.Index.create()

    args = [
        '-x',
        'c++',
        '-D__BANG__',
        '-D__mlu_device__=',
        '-Dbfloat16_t=float',
        '-Dhalf=float',
        '-Duint32_t=unsigned int',
        '-Duint64_t=unsigned long long',
    ]
    translation_unit = index.parse(cn_hdr_path, args)

    for diag in translation_unit.diagnostics:
        location = diag.location
        print(f"Diagnostic: {diag.spelling} (File: {location.file}, Line: {location.line}, Column: {location.column})")

    functions = []
    for cursor in translation_unit.cursor.get_children():
        visit(cursor)
    return functions


def create_special_funcs():
    ret = ""

    # Create philox.
    ret += "# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo, subsequenceHi, innerRounds\n"
    ret += "@core.extern\n"
    ret += "def philox(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, _builder=None):\n"
    ret += "\tret_type = core.block_type(core.dtype(\"uint32\"), [arg0.value, 4])\n"
    ret += "\tassert (not is_block_arg(arg0) and not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3) and not is_block_arg(arg4) and not is_block_arg(arg5) and not is_block_arg(arg6) and not is_block_arg(arg7)), 'philox: all args must be scalars'\n"
    ret += "\targs = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_philox_u32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n\n"

    # Create philox_v2.
    ret += "# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo\n"
    ret += "@core.extern\n"
    ret += "def philox_v2(arg0, arg1, arg2, arg3, arg4, arg5, _builder=None):\n"
    ret += "\tassert (not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3) and not is_block_arg(arg4) and not is_block_arg(arg5)), 'philox_v2: all args must be scalars'\n"
    ret += "\tassert (isinstance(arg0, core.constexpr)), 'philox_v2: outGroups(arg0) must be constexpr'\n"
    ret += "\tassert(arg0.value % 128==0), 'philox_v2: outGroups(arg0) must be divided by 128'\n"
    ret += "\tret_type = core.block_type(core.dtype(\"uint32\"), [arg0.value, 4])\n"
    ret += "\targs = [arg0, arg1, arg2, arg3, arg4, arg5]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_philox_v2_u32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n\n"

    # Create philox_v3.
    ret += "# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo, innerRounds, subsequenceLimit\n"
    ret += "@core.extern\n"
    ret += "def philox_v3(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, _builder=None):\n"
    ret += "\tassert (not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3) and not is_block_arg(arg4)), 'philox_v3: all args must be scalars'\n"
    ret += "\tassert (isinstance(arg0, core.constexpr)), 'philox_v3: outGroups(arg0) must be constexpr'\n"
    ret += "\tassert (isinstance(arg5, core.constexpr)), 'philox_v3: subsequenceLo(arg5) must be constexpr'\n"
    ret += "\tassert (isinstance(arg6, core.constexpr)), 'philox_v3: innerRounds(arg6) must be constexpr'\n"
    ret += "\tassert (isinstance(arg7, core.constexpr)), 'philox_v3: subsequenceLimit(arg7) must be constexpr'\n"
    ret += "\tassert (arg0.value % 128==0), 'philox_v3: outGroups(arg0) must be divided by 128'\n"
    ret += "\tassert (arg5.value % 128==0), 'philox_v3: subsequenceLo(arg5) must be divided by 128'\n"
    ret += "\tassert (arg6.value in [2, 4, 6, 8, 10]), 'philox_v3: innerRounds(arg0) must be divided by 128'\n"
    ret += "\tassert (arg7.value % 128==0), 'philox_v3: subsequenceLimit(arg7) must be divided by 128'\n"
    ret += "\tret_type = core.block_type(core.dtype(\"uint32\"), [arg0.value, 4])\n"
    ret += "\targs = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_philox_v3_u32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n\n"

    # Create ultra_silu_float2half.
    ret += "@core.extern\n"
    ret += "def ultra_silu_float2half(arg0,_builder=None):\n"
    ret += "\tret_type = core.block_type(core.float16, arg0.shape)\n"
    ret += "\targs = [arg0]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silu_f32_outf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silu_float2bfloat16.
    ret += "@core.extern\n"
    ret += "def ultra_silu_float2bfloat16(arg0,_builder=None):\n"
    ret += "\tret_type = core.block_type(core.bfloat16, arg0.shape)\n"
    ret += "\targs = [arg0]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silu_f32_outbf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silubp_float2half.
    ret += "@core.extern\n"
    ret += "def ultra_silubp_float2half(arg0,_builder=None):\n"
    ret += "\tret_type = core.block_type(core.float16, arg0.shape)\n"
    ret += "\targs = [arg0]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silubp_f32_outf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silubp_float2bfloat16.
    ret += "@core.extern\n"
    ret += "def ultra_silubp_float2bfloat16(arg0,_builder=None):\n"
    ret += "\tret_type = core.block_type(core.bfloat16, arg0.shape)\n"
    ret += "\targs = [arg0]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silubp_f32_outbf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silu_mul_float2half.
    ret += "@core.extern\n"
    ret += "def ultra_silu_mul_float2half(arg0, arg1, _builder=None):\n"
    ret += "\tret_type = core.block_type(core.float16, arg0.shape)\n"
    ret += "\tassert (not is_block_arg(arg1)), 'ultra_silu_mul_float2half: arg1 must be a scalar'\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silu_mul_scalar_f32_outf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silu_mul_float2bfloat16.
    ret += "@core.extern\n"
    ret += "def ultra_silu_mul_float2bfloat16(arg0, arg1, _builder=None):\n"
    ret += "\tret_type = core.block_type(core.bfloat16, arg0.shape)\n"
    ret += "\tassert (not is_block_arg(arg1)), 'ultra_silu_mul_float2bfloat16: arg1 must be a scalar'\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silu_mul_scalar_f32_outbf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silubp_mul_float2half.
    ret += "@core.extern\n"
    ret += "def ultra_silubp_mul_float2half(arg0, arg1, _builder=None):\n"
    ret += "\tret_type = core.block_type(core.float16, arg0.shape)\n"
    ret += "\tassert (not is_block_arg(arg1)), 'ultra_silubp_mul_float2half: arg1 must be a scalar'\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silubp_mul_scalar_f32_outf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create ultra_silubp_mul_float2bfloat16.
    ret += "@core.extern\n"
    ret += "def ultra_silubp_mul_float2bfloat16(arg0, arg1, _builder=None):\n"
    ret += "\tret_type = core.block_type(core.bfloat16, arg0.shape)\n"
    ret += "\tassert (not is_block_arg(arg1)), 'ultra_silubp_mul_float2bfloat16: arg1 must be a scalar'\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_silubp_mul_scalar_f32_outbf16\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\treturn core.tensor(ret, ret_type)\n"

    # Create cycle_sub_mul_exp.
    ret += "# result = exp2((arg0 - arg1) * arg2)\n"
    ret += "@core.extern\n"
    ret += "def cycle_sub_mul_exp(arg0, arg1, arg2, _builder=None):\n"
    ret += "\tret_shape = arg0.shape\n"
    ret += "\tsub_val = arg1\n"
    ret += "\tn = arg0.type.numel\n"
    ret += "\tif not is_block_arg(arg1):\n"
    ret += "\t\tassert n // 64 >= 16, \"arg0 element num must be more than 64 * 16\"\n"
    ret += "\t\tsub_val = core.full([64], arg1, dtype=arg0.dtype, _builder=_builder)\n"
    ret += "\telse:\n"
    ret += "\t\tassert is_cycle_args(arg0, arg1)\n"
    ret += "\t\tassert arg0.shape[1:] == arg1.shape[1:], 'arg0 and arg1 shape must equal except the hightest dim'\n"
    ret += "\t\tn_short = sub_val.type.numel\n"
    ret += "\t\tsub_val = core.reshape(sub_val, (n_short), can_reorder=True, _builder=_builder)\n"
    ret += "\t\tassert ((n % n_short == 0) and (n // n_short >= 16)), \"arg0 element num must be divisible by arg1 element num, and the ratio between the two must be greater than 16.\"\n"
    ret += "\targ0 = core.reshape(arg0, (n), can_reorder=True, _builder=_builder)\n"
    ret += "\tret_type = core.block_type(arg0.dtype, arg0.shape)\n"
    ret += "\targs = [arg0, sub_val, arg2]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_cycle_sub_exp_f32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\tret_tensor = core.tensor(ret, ret_type)\n"
    ret += "\treturn core.reshape(ret_tensor, ret_shape, can_reorder=True, _builder=_builder)\n"

    # Create fast_dividef.
    ret += "@core.extern\n"
    ret += "def fast_dividef(arg0, arg1, _builder=None):\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = None\n"
    ret += "\tif is_block_arg(arg0) and is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg0.dtype, arg0.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_fast_div_f16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\t\tif arg0.dtype.is_bf16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_fast_div_bf16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\t\tif arg0.dtype.is_fp32():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_fast_div_f32_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\telif is_block_arg(arg0) and not is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg0.dtype, arg0.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_fast_div_scalar_f16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\t\tif arg0.dtype.is_bf16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_fast_div_scalar_bf16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\telif not is_block_arg(arg0) and is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg1.dtype, arg1.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_scalar_fast_div_vector_f16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\t\tif arg0.dtype.is_bf16():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_scalar_fast_div_vector_bf16_rn\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\tif ret is not None:\n"
    ret += "\t\treturn core.tensor(ret, ret_type)\n"
    ret += "\telse:\n"
    ret += "\t\treturn None\n"

    # Create ultra_pow.
    ret += "@core.extern\n"
    ret += "def ultra_pow(arg0, arg1, _builder=None):\n"
    ret += "\targs = [arg0, arg1]\n"
    ret += "\targs = [semantic.to_tensor(arg, _builder) for arg in args]\n"
    ret += "\targs = [arg.handle for arg in args]\n"
    ret += "\tret = None\n"
    ret += "\tif is_block_arg(arg0) and is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg0.dtype, arg0.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp32():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_pow_f32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\telif is_block_arg(arg0) and not is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg0.dtype, arg0.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp32():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_vector_ultra_pow_scalar_f32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\telif not is_block_arg(arg0) and is_block_arg(arg1):\n"
    ret += "\t\tret_type = core.block_type(arg1.dtype, arg1.shape)\n"
    ret += "\t\tif arg0.dtype.is_fp32():\n"
    ret += "\t\t\tret = _builder.create_extern_elementwise(\"\", \"\", \"__cn_scalar_ultra_pow_vector_f32\", args, ret_type.to_ir(_builder), True)\n"
    ret += "\tif ret is not None:\n"
    ret += "\t\treturn core.tensor(ret, ret_type)\n"
    ret += "\telse:\n"
    ret += "\t\treturn None\n"

    return ret


def compat_func():
    name_map = {
        'popc': 'popcnt',
        'fast_dividef': 'fast_div_rn',
        'isfinited': 'finitef',
    }
    func_str = ""
    for key, value in name_map.items():
        func_str += f"{value} = {key}\n"
    return func_str


class Symbol:
    _name: str
    _op_name: str
    _compute_mode: str
    _ret_type: str
    _arg_names: List[str]
    _arg_types: List[str]
    _workspace_names: List[str]
    _workspace_types: List[str]

    def __init__(
        self,
        name: str,
        op_name: str,
        compute_mode: str,
        ret_type: str,
        arg_names: List[str],
        arg_types: List[str],
        workspace_names: List[str],
        workspace_types: List[str],
    ) -> None:
        '''
        A symbol is a function declaration.
        :param name: name of the symbol
        :param op_name: name of the operation
        :param compute_mode: supported input types(vector, scalar, broadcast or all)
        :param ret_type: return type of the operation
        :param arg_names: names of the arguments
        :param arg_types: types of the arguments
        '''
        self._name = name
        self._op_name = op_name
        self._compute_mode = compute_mode
        self._ret_type = ret_type
        self._arg_names = list(arg_names)
        self._arg_types = list(arg_types)
        self._workspace_names = list(workspace_names)
        self._workspace_types = list(workspace_types)

    @property
    def name(self) -> str:
        return self._name

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def compute_mode(self) -> str:
        return self._compute_mode

    @property
    def ret_type(self) -> str:
        return self._ret_type

    @property
    def arg_names(self) -> List[str]:
        return self._arg_names

    @property
    def arg_types(self) -> List[str]:
        return self._arg_types

    @property
    def workspace_names(self) -> List[str]:
        return self._workspace_names

    @property
    def workspace_types(self) -> List[str]:
        return self._workspace_types


def convert_type(type_str) -> Optional[str]:
    if type_str == "i1" or type_str == "bool":
        return "int1"
    elif type_str == "i8" or type_str == "s8":
        return "int8"
    elif type_str == "u8":
        return "uint8"
    elif type_str == "i16" or type_str == "s16":
        return "int16"
    elif type_str == "u16":
        return "uint16"
    elif type_str == "i32" or type_str == "s32":
        return "int32"
    elif type_str == "u32":
        return "uint32"
    elif type_str == "i64" or type_str == "s64":
        return "int64"
    elif type_str == "u64":
        return "uint64"
    elif type_str == "half" or type_str == "f16":
        return "fp16"
    elif type_str == "bf16" or type_str == "%struct.__bang_bfloat16":
        return "bf16"
    elif type_str == "float" or type_str == "f32":
        return "fp32"
    elif type_str == "double" or type_str == "f64":
        return "fp64"
    elif type_str == "void":
        return "void"
    else:
        # ignore other types, such as pointer types
        return None


def to_unsigned(type_str) -> str:
    if type_str == "int32":
        return "uint32"
    elif type_str == "int64":
        return "uint64"
    else:
        return type_str


class ExternLibrary(ABC):
    _name: str
    _cn_path: str
    _cn_symbols: Dict[str, Symbol]
    _format: bool
    _grouping: bool

    def __init__(
        self,
        name: str,
        cn_path: str,
        format: bool = True,
        grouping: bool = True,
    ) -> None:
        '''
        Abstract class for extern library.
        :param name: name of the library
        :param cn_path: path of the library of cmbricon
        :param format: whether to format the generated stub file
        '''
        self._name = name
        self._cn_path = cn_path
        self._cn_symbols = {}
        self._format = format
        self._grouping = grouping

    @property
    def name(self) -> str:
        return self._name

    @property
    def cn_path(self) -> str:
        return self._cn_path

    @property
    def cn_symbols(self) -> Dict[str, Symbol]:
        return self._cn_symbols

    @property
    def grouping(self) -> bool:
        return self._grouping

    @abstractmethod
    def parse_cn_symbols(self, input_file, export_symbols) -> None:
        pass

    @abstractmethod
    def _output_stubs(self) -> str:
        pass

    def generate_stub_file(self, output_path) -> None:
        file_str = self._output_stubs()
        if file_str is None or len(file_str) == 0:
            raise Exception("file_str is empty")

        output_file = f"{output_path}"
        with open(output_file, "w") as f:
            f.write(file_str)
            f.close()
            if self._format:
                subprocess.Popen(["yapf", "-i", output_file], stdout=subprocess.PIPE).communicate()
                subprocess.Popen(["isort", output_file], stdout=subprocess.PIPE).communicate()


class Libdevice(ExternLibrary):
    _symbol_groups: Dict[str, List[Symbol]]

    def __init__(self, cn_path) -> None:
        '''
        Constructor for Libdevice.
        :param cn_path: path of the libdevice library of cambricon
        '''
        super().__init__("libdevice", cn_path)
        self._symbol_groups = {}
        self.is_pure = True

    @staticmethod
    def _skipped_arg_num(compute_mode):
        if compute_mode == 'cycle':
            return 3
        else:
            return 2

    @staticmethod
    def _extract_cn_symbol(line) -> Optional[Symbol]:
        # Extract symbols from line in the following format:
        # "define <dso_local> [zeroext/signext] [ret_type/void] @<name>(<arg_types>,)"
        entries = line.split("@")
        ret_str = entries[0]
        func_str = entries[1]
        # Get ret_type
        ret_strs = ret_str.split()
        ret_type = convert_type(ret_strs[-1])

        # The function header follows the format "func_name(args...) attr {",
        # so to separate the function name from the parameters,
        # you should locate the first opening parenthesis "(" to divide them.
        splitor_pos = func_str.find('(')
        func_strs = [func_str[:splitor_pos], func_str[splitor_pos + 1:]]
        # Get function name
        func_name = func_strs[0].replace("@", "")
        extra_strs = ["__cn", "_vector_", "_scalar_"]

        op_name = func_name
        compute_mode = ""
        if "scalar" in op_name and "vector" in op_name:
            compute_mode = "broadcast"
        elif "scalar" in op_name:
            compute_mode = "scalar"
        elif "cycle" in op_name:
            compute_mode = "cycle"
        elif "vector" in op_name:
            compute_mode = "vector"
        else:
            return None

        for string in extra_strs:
            op_name = op_name.replace(string, "")
        # Special case for cast
        op_name_str = op_name.split("_")
        dtype_list = ["s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64", "f16", "f32", "bool", "bf16"]
        is_unsigned = False
        if "cast" not in op_name:
            for dtype_str in dtype_list:
                if dtype_str in op_name:
                    op_name = op_name.replace("_" + dtype_str, "")
                    if dtype_str in ["u8", "u16", "u32", "u64"]:
                        is_unsigned = True
        if compute_mode == "broadcast":
            for dtype_str in dtype_list:
                op_name = op_name.replace(dtype_str, "")
            op_name = "broadcast_" + op_name
        if op_name.startswith("_"):
            op_name = op_name[1:]
        if "cast" not in op_name:
            for dtype_str in dtype_list:
                if op_name.endswith(dtype_str):
                    op_name = op_name.rstrip(dtype_str)
        arg_strs = func_strs[1].split(",")
        arg_types = []
        arg_names = []
        workspace_arg_strs = []
        workspace_types = []
        workspace_names = []

        if op_name in need_workspace_func:
            workspace_num = get_worspace_num(op_name, op_name_str[-1])
            if workspace_num != 0:
                workspace_arg_strs = arg_strs[-workspace_num:]
                if compute_mode == "vector":
                    arg_strs = arg_strs[:-workspace_num]
        # Get arg_types
        for i, arg_str in enumerate(arg_strs):
            arg_type = convert_type(arg_str.split()[0].replace("*", ""))
            if arg_type is None:
                return None
            if ret_type == "void" and i < Libdevice._skipped_arg_num(compute_mode):
                continue
            if is_unsigned == True:
                arg_type = "u" + arg_type
            if op_name_str[0] == "cast":
                arg_type = convert_type(op_name_str[1])
            arg_name = 'arg'
            if ret_type == "void":
                arg_name = arg_name + str(i - 2)
            else:
                arg_name = arg_name + str(i)
            arg_types.append(arg_type)
            arg_names.append(arg_name)

        for i, arg_str in enumerate(workspace_arg_strs):
            arg_type = convert_type(arg_str.split()[0].replace("*", ""))
            if arg_type is None:
                return None
            if is_unsigned == True:
                arg_type = "u" + arg_type
            if op_name_str[0] == "cast":
                arg_type = convert_type(op_name_str[1])
            arg_name = 'arg'
            if ret_type == "void":
                arg_name = arg_name + str(i - 2 + len(arg_strs))
            else:
                arg_name = arg_name + str(i)
            workspace_types.append(arg_type)
            workspace_names.append(arg_name)

        # Update ret_type.
        if ret_type == "void":
            ret_type = arg_strs[1].split()[0].replace("*", "")
            ret_type = convert_type(ret_type)
        if is_unsigned == True:
            ret_type = "u" + ret_type
        if op_name_str[0] == "cast":
            ret_type = convert_type(op_name_str[3])
        # Now we can not support cycle libdevice function.
        if "atomic" in op_name or "reduce" in op_name or "broadcast" in op_name or compute_mode == 'cycle':
            return None
        # Rename round_mode from ['tz', 'dn', 'up'] to ['rz', 'rd', 'ru'].
        cn_round_mode = ['_tz', '_dn', '_up']
        nv_round_mode = ['_rz', '_rd', '_ru']
        for i in range(len(cn_round_mode)):
            if op_name.endswith(cn_round_mode[i]):
                op_name = op_name.replace(cn_round_mode[i], nv_round_mode[i])
        return Symbol(func_name, op_name, compute_mode, ret_type, arg_names, arg_types, workspace_names,
                      workspace_types)

    def _group_cn_symbols(self) -> None:
        # Group functions together by renaming.
        renaming = {
            'isfinite': 'isfinited',
            'mulh': 'mulhi',
            'and': 'bitwise_and',
            'or': 'bitwise_or',
            'not': 'bitwise_not',
            'fast_pow': 'fast_powf',
            'popcnt': 'popc',
            'fast_exp': 'fast_expf',
            'fast_exp10': 'fast_exp10f',
            'fast_log2': 'fast_log2f',
            'fast_log10': 'fast_log10',
            'fast_div_rn': 'fast_dividef',
            'ultra_silu_outb': 'ultra_silu_out',
            'ultra_silubp_outb': 'ultra_silubp_out',
            'broadcast_ultra_silu_mul_outb': 'broadcast_ultra_silu_mul_out',
            'broadcast_ultra_silubp_mul_outb': 'broadcast_ultra_silubp_mul_out',
            'cast_bf16_to_f16': 'bfloat162half',
            'cast_bf16_to_f32': 'bfloat162float',
            'cast_bf16_to_s16': 'bfloat162short',
            'cast_bf16_to_s32': 'bfloat162int',
            'cast_bf16_to_s64': 'bfloat162ll',
            'cast_bf16_to_s8': 'bfloat162byte',
            'cast_bf16_to_u16': 'bfloat162ushort',
            'cast_bf16_to_u32': 'bfloat162uint',
            'cast_bf16_to_u64': 'bfloat162ull',
            'cast_bf16_to_u8': 'bfloat162ubyte',
            'cast_f16_to_bf16': 'half2bfloat16',
            'cast_f16_to_f32': 'half2float',
            'cast_f16_to_s16': 'half2short',
            'cast_f16_to_s16_rz': 'half2short_rz',
            'cast_f16_to_s32': 'half2int',
            'cast_f16_to_s32_rz': 'half2int_rz',
            'cast_f16_to_s64': 'half2ll',
            'cast_f16_to_s64_rz': 'half2ll_rz',
            'cast_f16_to_s8': 'half2byte',
            'cast_f16_to_s8_rz': 'half2byte_rz',
            'cast_f16_to_u16': 'half2ushort',
            'cast_f16_to_u16_rz': 'half2ushort_rz',
            'cast_f16_to_u32': 'half2uint',
            'cast_f16_to_u32_rz': 'half2uint_rz',
            'cast_f16_to_u64': 'half2ull',
            'cast_f16_to_u64_rz': 'half2ull_rz',
            'cast_f16_to_u8': 'half2ubyte',
            'cast_f16_to_u8_rz': 'half2ubyte_rz',
            'cast_f32_to_bf16': 'float2bfloat16',
            'cast_f32_to_f16': 'float2half',
            'cast_f32_to_f16_rn': 'float2half_rn',
            'cast_f32_to_f16_rz': 'float2half_rz',
            'cast_f32_to_f64': 'float2double',
            'cast_f32_to_f64_rz': 'float2double_rz',
            'cast_f32_to_s16': 'float2short',
            'cast_f32_to_s16_rz': 'float2short_rz',
            'cast_f32_to_s32': 'float2int',
            'cast_f32_to_s32_rd': 'float2int_rd',
            'cast_f32_to_s32_rn': 'float2int_rn',
            'cast_f32_to_s32_ru': 'float2int_ru',
            'cast_f32_to_s32_rz': 'float2int_rz',
            'cast_f32_to_s64': 'float2ll',
            'cast_f32_to_s64_rn': 'float2ll_rn',
            'cast_f32_to_s64_rz': 'float2ll_rz',
            'cast_f32_to_s8': 'float2byte',
            'cast_f32_to_s8_rz': 'float2byte_rz',
            'cast_f32_to_u16': 'float2ushort',
            'cast_f32_to_u16_rz': 'float2ushort_rz',
            'cast_f32_to_u32': 'float2uint',
            'cast_f32_to_u32_rn': 'float2uint_rn',
            'cast_f32_to_u32_rz': 'float2uint_rz',
            'cast_f32_to_u64': 'float2ull',
            'cast_f32_to_u64_rz': 'float2ull_rz',
            'cast_f32_to_u8': 'float2ubyte',
            'cast_f32_to_u8_rz': 'float2ubyte_rz',
            'cast_f64_to_f32': 'double2float',
            'cast_f64_to_f32_rn': 'double2float_rn',
            'cast_f64_to_s64': 'double2ll',
            'cast_f64_to_s64_rz': 'double2ll_rz',
            'cast_s16_to_bf16': 'short2bfloat16',
            'cast_s16_to_f16': 'short2half',
            'cast_s16_to_f16_rn': 'short2half_rn',
            'cast_s16_to_f32': 'short2float',
            'cast_s16_to_s32': 'short2int',
            'cast_s16_to_s64': 'short2ll',
            'cast_s16_to_s8': 'short2byte',
            'cast_s16_to_u16': 'short2ushort',
            'cast_s16_to_u32': 'short2uint',
            'cast_s16_to_u64': 'short2ull',
            'cast_s16_to_u8': 'short2ubyte',
            'cast_s32_to_bf16': 'int2bfloat16',
            'cast_s32_to_f16': 'int2half',
            'cast_s32_to_f16_rn': 'int2half_rn',
            'cast_s32_to_f32': 'int2float',
            'cast_s32_to_f32_rn': 'int2float_rn',
            'cast_s32_to_f32_rz': 'int2float_rz',
            'cast_s32_to_s16': 'int2short',
            'cast_s32_to_s64': 'int2ll',
            'cast_s32_to_s8': 'int2byte',
            'cast_s32_to_u16': 'int2ushort',
            'cast_s32_to_u32': 'int2uint',
            'cast_s32_to_u64': 'int2ull',
            'cast_s32_to_u8': 'int2ubyte',
            'cast_s64_to_bf16': 'll2bfloat16',
            'cast_s64_to_f16': 'll2half',
            'cast_s64_to_f16_rn': 'll2half_rn',
            'cast_s64_to_f32': 'll2float',
            'cast_s64_to_f32_rn': 'll2float_rn',
            'cast_s64_to_f32_rz': 'll2float_rz',
            'cast_s64_to_f64': 'll2double',
            'cast_s64_to_f64_rn': 'll2double_rn',
            'cast_s64_to_s16': 'll2short',
            'cast_s64_to_s32': 'll2int',
            'cast_s64_to_s8': 'll2byte',
            'cast_s64_to_u16': 'll2ushort',
            'cast_s64_to_u32': 'll2uint',
            'cast_s64_to_u64': 'll2ull',
            'cast_s64_to_u8': 'll2ubyte',
            'cast_s8_to_bf16': 'byte2bfloat16',
            'cast_s8_to_f16': 'byte2half',
            'cast_s8_to_f32': 'byte2float',
            'cast_s8_to_s16': 'byte2short',
            'cast_s8_to_s32': 'byte2int',
            'cast_s8_to_s64': 'byte2ll',
            'cast_s8_to_u16': 'byte2ushort',
            'cast_s8_to_u32': 'byte2uint',
            'cast_s8_to_u64': 'byte2ull',
            'cast_s8_to_u8': 'byte2ubyte',
            'cast_u16_to_bf16': 'ushort2bfloat16',
            'cast_u16_to_f16': 'ushort2half',
            'cast_u16_to_f16_rn': 'ushort2half_rn',
            'cast_u16_to_f32': 'ushort2float',
            'cast_u16_to_s16': 'ushort2short',
            'cast_u16_to_s32': 'ushort2int',
            'cast_u16_to_s64': 'ushort2ll',
            'cast_u16_to_s8': 'ushort2byte',
            'cast_u16_to_u32': 'ushort2uint',
            'cast_u16_to_u64': 'ushort2ull',
            'cast_u16_to_u8': 'ushort2ubyte',
            'cast_u32_to_bf16': 'uint2bfloat16',
            'cast_u32_to_f16': 'uint2half',
            'cast_u32_to_f16_rn': 'uint2half_rn',
            'cast_u32_to_f32': 'uint2float',
            'cast_u32_to_f32_rn': 'uint2float_rn',
            'cast_u32_to_s16': 'uint2short',
            'cast_u32_to_s32': 'uint2int',
            'cast_u32_to_s64': 'uint2ll',
            'cast_u32_to_s8': 'uint2byte',
            'cast_u32_to_u16': 'uint2ushort',
            'cast_u32_to_u64': 'uint2ull',
            'cast_u32_to_u8': 'uint2ubyte',
            'cast_u64_to_bf16': 'ull2bfloat16',
            'cast_u64_to_f16': 'ull2half',
            'cast_u64_to_f16_rn': 'ull2half_rn',
            'cast_u64_to_f32': 'ull2float',
            'cast_u64_to_f32_rn': 'ull2float_rn',
            'cast_u64_to_s16': 'ull2short',
            'cast_u64_to_s32': 'ull2int',
            'cast_u64_to_s64': 'ull2ll',
            'cast_u64_to_s8': 'ull2byte',
            'cast_u64_to_u16': 'ull2ushort',
            'cast_u64_to_u32': 'ull2uint',
            'cast_u64_to_u8': 'ull2ubyte',
            'cast_u8_to_bf16': 'ubyte2bfloat16',
            'cast_u8_to_f16': 'ubyte2half',
            'cast_u8_to_f32': 'ubyte2float',
            'cast_u8_to_s16': 'ubyte2short',
            'cast_u8_to_s32': 'ubyte2int',
            'cast_u8_to_s64': 'ubyte2ll',
            'cast_u8_to_s8': 'ubyte2byte',
            'cast_u8_to_u16': 'ubyte2ushort',
            'cast_u8_to_u32': 'ubyte2uint',
            'cast_u8_to_u64': 'ubyte2ull',
        }
        for symbol in self._cn_symbols.values():
            op_name = symbol.op_name
            if op_name in renaming:
                op_name = renaming[op_name]
                symbol._op_name = op_name
            if op_name in self._symbol_groups:
                self._symbol_groups[op_name].append(symbol)
            else:
                self._symbol_groups[op_name] = [symbol]

    def parse_cn_symbols(self, input_file, export_symbols) -> None:
        op_name_list = []
        if len(self.cn_symbols) > 0:
            return
        output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
        extra_nv_name = {'mod': {'fmod': [['fp32', 'fp32'], ['fp16', 'fp16']]}}
        for line in output:
            symbol = self._extract_cn_symbol(line)
            if (symbol is None) or (symbol.name not in export_symbols):
                continue
            self._cn_symbols[symbol.name] = symbol
            op_name_list.append(symbol.op_name)
            # Add extern_nv_name symbol to _symbol_groups.
            if symbol.op_name in extra_nv_name:
                ret_type = symbol.ret_type
                arg_names = symbol.arg_names
                arg_types = symbol.arg_types
                for extra_name in extra_nv_name[symbol.op_name]:
                    extra_arg_types = extra_nv_name[symbol.op_name][extra_name]
                    if symbol.arg_types in extra_arg_types:
                        extra_symbol = Symbol(symbol.name, extra_name, symbol.compute_mode, symbol.ret_type,
                                              symbol.arg_names, symbol.arg_types, symbol.workspace_names,
                                              symbol.workspace_types)
                        self._cn_symbols[symbol.name] = extra_symbol
                        op_name_list.append(extra_name)
        op_name_list = set(op_name_list)
        op_name_dict = {}
        for op_name in op_name_list:
            op_name_dict[op_name] = op_name
        self._group_cn_symbols()

    def _output_stubs(self) -> str:
        # Generate python functions in the following format:
        # @extern.extern
        # def <op_name>(<args>, _builder=None):
        #   arg_type_symbol_dict = {[arg_type]: {(symbol, ret_type)}}
        #   return extern.dispatch("libdevice", <path>, <args>, <arg_type_symbol_dict>, _builder)
        import_str = "from triton.language import core, semantic\n"

        header_str = ""

        is_arg_block_type = r"""
def is_block_arg(arg):
    arg_is_block = True
    if isinstance(arg, core.constexpr):
        arg_is_block = False
    else:
        if not arg.type.is_block():
            arg_is_block = False
    return arg_is_block


def is_cycle_args(arg0, arg1):
    arg0_shape = arg0.shape
    arg1_shape = arg1.shape
    need_broadcast = False
    for i in reversed(range(0, len(arg1_shape))):
        if not need_broadcast:
            if arg1_shape[i] == arg0_shape[i]:
                continue
            need_broadcast = True
        if arg0_shape[i] % arg1_shape[i] != 0:
            return False
    return True
        """
        header_str += is_arg_block_type
        func_str = "\n"

        for symbols in self._symbol_groups.values():
            symbols_op_name = symbols[0].op_name
            if symbols_op_name in special_funcs:
                continue

            func_str += "@core.extern\n"
            func_name_str = f"def {symbols_op_name}("
            for arg_name in symbols[0].arg_names:
                func_name_str += f"{arg_name}, "
            func_name_str += "_builder=None):\n"

            return_str = f"\t\treturn core.extern_elementwise(\"\", \"\", ["
            for arg_name in symbols[0].arg_names:
                return_str += f"{arg_name}, "
            return_str += "], \n"

            cn_vector_return_str = ""
            cn_scalar_arg_type_symbol_dict_str = "{"
            cn_vector_arg_type_symbol_dict_str = "{"
            for symbol in symbols:
                if symbol.compute_mode == "scalar":
                    cn_scalar_arg_type_symbol_dict_str += "("
                else:
                    cn_vector_arg_type_symbol_dict_str += "("
                for arg_type in symbol.arg_types:
                    if symbol.compute_mode == "scalar":
                        cn_scalar_arg_type_symbol_dict_str += f'core.dtype("{arg_type}"),'
                    else:
                        cn_vector_arg_type_symbol_dict_str += f'core.dtype("{arg_type}"),'
                ret_type = f'core.dtype("{symbol.ret_type}")'
                if symbol.compute_mode == "scalar":
                    cn_scalar_arg_type_symbol_dict_str += "): (\"" + symbol.name + "\", " + ret_type + "),\n"
                else:
                    cn_vector_arg_type_symbol_dict_str += "): (\"" + symbol.name + "\", " + ret_type + "),\n"
            cn_scalar_arg_type_symbol_dict_str += "}"
            cn_vector_arg_type_symbol_dict_str += "}"

            cn_scalar_return_str = return_str + cn_scalar_arg_type_symbol_dict_str
            cn_scalar_return_str += f", is_pure={self.is_pure}"
            cn_scalar_return_str += ", _builder=_builder)\n"
            cn_vector_return_str += return_str + cn_vector_arg_type_symbol_dict_str
            cn_vector_return_str += f", is_pure={self.is_pure}"
            cn_vector_return_str += ", _builder=_builder)\n"

            if cn_scalar_arg_type_symbol_dict_str == "{}":
                cn_scalar_return_str = "\t\t\treturn None\n"
            if cn_vector_arg_type_symbol_dict_str == "{}":
                cn_vector_return_str = "\t\t\treturn None\n"
            cn_vector_if_str = ""
            for arg_name in symbol.arg_names:
                cn_vector_if_str = cn_vector_if_str + f'is_block_arg({arg_name}) or '
            cn_vector_if_str = cn_vector_if_str.rstrip(" or ")
            func_str += func_name_str + "\tif " + cn_vector_if_str + " :\n" + cn_vector_return_str + "\telse:\n\t" + cn_scalar_return_str + "\n"
        func_str += create_special_funcs()
        func_str += compat_func()
        file_str = import_str + header_str + func_str
        return file_str


class LLVMDisassembler:
    _path: str
    _ll_file: str

    def __init__(self, path) -> None:
        '''
        Invoke llvm-dis to disassemble the given file.
        :param path: path to llvm-dis
        '''
        self._path = path
        self._ll_file = "/tmp/extern_lib.ll"

    def disasm(self, lib_path: str) -> None:
        subprocess.Popen([self._path, lib_path, "-o", self.ll_file], stdout=subprocess.PIPE).communicate()

    @property
    def ll_file(self) -> str:
        return self._ll_file

    @property
    def path(self) -> str:
        return self._path


extern_libs = ["libdevice"]


def build(
    llvm_dis_path: str,
    cn_lib_path: str,
    cn_hdr_path: str,
    lib_name: str,
    output_path: str,
) -> None:
    '''
      Interface function to build the library file.
      :param llvm_dis_path: path to the llvm-dis binary
      :param lib_path: path to the external library file
      :param lib_name: name of the library
      :param output_path: path to the output file
    '''
    if lib_name == "libdevice":
        extern_lib = Libdevice(cn_lib_path)
    else:
        raise Exception(f"Unknown extern library: {lib_name}")

    export_symbols = parse_export_symbol(cn_hdr_path)
    llvm_disassembler = LLVMDisassembler(llvm_dis_path)
    llvm_disassembler.disasm(cn_lib_path)
    extern_lib.parse_cn_symbols(llvm_disassembler.ll_file, export_symbols)
    extern_lib.generate_stub_file(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-dis", dest="llvm_dis_path", help="Path to llvm-dis", default="llvm-dis")
    parser.add_argument("--cn-lib-path", dest="cn_lib_path", help="Path to the extern library of cambricon")
    parser.add_argument("--cn-hdr-path", dest="cn_hdr_path", help="Path to the extern library header of cambricon")
    parser.add_argument("--lib-name", dest="lib_name", help="Name of the extern library")
    parser.add_argument("--output", dest="output_path", help="Output file path", default="/tmp/")
    args = parser.parse_args()

    build(args.llvm_dis_path, args.cn_lib_path, args.cn_hdr_path, args.lib_name, args.output_path)
