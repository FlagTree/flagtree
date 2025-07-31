from __future__ import annotations
import hashlib
import json
from .._C.libtriton import get_cache_invalidating_env_vars, ir
from ..backends import backends
from ..backends.compiler import GPUTarget
from .. import __version__
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..tools.disasm import get_sass
from dataclasses import dataclass
# TODO: this shouldn't be here
from .code_generator import ast_to_ttir
from pathlib import Path
from .._utils import find_paths_if, get_iterable_path
import re
import functools
import os

# Table that associates strings to AttrsDescriptor (sub)classes.
# In this way we can dynamically select the correct class
# constructor
_descriptor_table = {}


def register_descriptor(cls):
    """
    Register a descriptor into the descriptor table
    """
    _descriptor_table[cls.__name__] = cls
    return cls


@register_descriptor
@dataclass
class AttrsDescriptor:
    """
    This class handles compile-time properties for specific function parameters.

    Different backends can add more properties to the common ones. The class
    contains two fields:

    `arg_properties`: a dictionary containing the different compile-time properties for different
        parameters. I.e., the dictionary is a map from property names to parameter indices
        {
        "prop0": (0, 2, 3)
        "prop1": (0, 4, 5)
        }
        Different backends might need different properties on those paraemters to enable
        specific optimizations. The common compile time properties contained in this class
        are :
        - "tt.divisibility", i.e., is the given parameter divisible by 16
        - "tt.equal_to_1", i.e., is the given parameter an integer constant 1

    `property_values`: a dictionary containing the value of the different compile-time properties, like:
        {
            "prop0": val0
            "prop1": val1
        }

    `constant_properties`: a set containing the properties that can be used to determine if a parameter is constant

    """
    __slots__ = ('divisibility_16', 'equal_to_1', 'equal_to_none', 'arg_properties', 'property_values',
                 'constant_properties')

    def __init__(self, params=None, values=None):
        """
        Initialize the compile-time properties

        We can initialize the AttrsDescriptor class by passing the list of params
        of the function and their `values`. The function will try to apply the properties
        to the values and save the parameters in the `arg_properties` list. If we don't pass
        either the `params` or the `values` we should initialize the class via an alternative method
        (see `from_dict` or `from_hints`)
        """
        # Default initialization
        self.arg_properties = {}
        self.property_values = {}
        self.equal_to_none = {}
        self.constant_properties = set()

        self._add_common_properties(params, values)
        self._add_backend_properties(params, values)
        self._init_slots()

    def _add_common_properties(self, params, values):
        """ Add common compile-time properties """
        self.property_values["tt.divisibility"] = 16
        self.property_values["tt.equal_to"] = 1
        self.constant_properties.add("tt.equal_to")

        if (params is None) or (values is None):
            return

        # Compile properties deduction
        assert (len(params) == len(values))

        # Divisibility property
        divisibility_16 = []
        for param, arg in zip(params, values):
            if param.do_not_specialize or \
               param.do_not_specialize_on_alignment:
                continue
            paths = find_paths_if(arg, lambda path, val: AttrsDescriptor.is_divisible_by_16(val))
            divisibility_16 += [(param.num, ) + x for x in paths]
        self.arg_properties["tt.divisibility"] = divisibility_16

        # Equal to 1 property
        equal_to_1 = []
        for param, arg in zip(params, values):
            if param.do_not_specialize:
                continue
            paths = find_paths_if(arg, lambda path, val: AttrsDescriptor.is_equal_to_1(val))
            equal_to_1 += [(param.num, ) + x for x in paths]
        self.arg_properties["tt.equal_to"] = equal_to_1

        # Equal to None property
        equal_to_none = []
        for param, arg in zip(params, values):
            paths = find_paths_if(arg, lambda path, val: val is None)
            equal_to_none += [(param.num, ) + x for x in paths]
        self.equal_to_none = equal_to_none

    def _add_backend_properties(self, params=None, values=None):
        """ This method is for different subclasses to implement their own compile-time properties """
        pass

    def _init_slots(self):
        """ Initialize the slots of this class """
        for name, val in self.arg_properties.items():
            setattr(self, name.removeprefix('tt.') + '_' + str(self.property_values[name]), val)

    def get_fn_attrs(self) -> Dict:
        """
        Get the function attributes as a dictionary.

        The returned dictionary will look like :
            {
            "arg0" : [(prop_name00, val00), (prop_name01, val01), ...)]}
            "arg1" : [(prop_name10, val10), (prop_name11, val11), ...)]}
            }
        """
        attrs = {}
        for prop_name, arg_set in self.arg_properties.items():
            prop_val = self.property_values[prop_name]
            for arg in arg_set:
                attrs[arg] = attrs.get(arg, []) + [(prop_name, prop_val)]
        return attrs

    def get_constants(self) -> Dict:
        """ Return a mapping of constant parameters to their values """
        constants = {}
        for prop_name in self.constant_properties:
            for p in self.arg_properties.get(prop_name, []):
                constants[p] = self.property_values[prop_name]
        for v in self.equal_to_none:
            constants[v] = None
        return constants

    def filter_out_constants(self):
        """ Return the same object, without properties marked as constants"""
        import copy
        c = copy.deepcopy(self)
        for prop_name in c.constant_properties:
            c.arg_properties.pop(prop_name, None)
            c.property_values.pop(prop_name, None)
        c.constant_properties = {}
        return c

    def hash(self):
        values = [sorted(self.arg_properties.values())]
        values += [sorted(self.property_values.values())]
        values += [sorted(self.constant_properties)]
        key = str(values)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        """
        Store the fields of this class in a serializable dictionary
        """
        # We need to only store the `arg_properties` field. To initialize the
        # other fields we relay on the class type. We store it as a string in
        # the dictionary so that we can use it to invoke the appropriate
        # (sub)class constructor in the `from_dict` method.
        return {"arg_properties": self.arg_properties, "cls": type(self).__name__}

    @staticmethod
    def from_dict(data):
        """
        Create the object from a serializable dictionary
        """
        attrs_descriptor = _descriptor_table[data["cls"]]()
        for prop_name, param_ids in data["arg_properties"].items():
            from collections.abc import Iterable
            attrs_descriptor.arg_properties[prop_name] = [
                tuple(i) if isinstance(i, Iterable) else tuple((i, )) for i in param_ids
            ]
        attrs_descriptor._init_slots()
        return attrs_descriptor

    @classmethod
    def from_hints(cls, hints: List[Tuple[int, int]]):
        """
        Create the class from a set of hints that are passed in.

        Instead of deducing the properties from a list of paramaters and values,
        the user can pass in a list of `hints=[(param_index, val)]` and if `val`
        matches one of the values of the properties (e.g., `prop_val[prop0]`),
        then we insert `param_index` into the correct list (e.g., in
        `arg_properties[prop0]`)
        """
        attrs_descriptor = cls()
        for prop_name, prop_val in attrs_descriptor.property_values.items():
            attrs_descriptor.arg_properties[prop_name] = [i for i, h in hints.items() if h == prop_val]
        attrs_descriptor._init_slots()
        return attrs_descriptor

    @staticmethod
    def is_divisible_by_16(x):
        """ Return if the argument is a multiple of 16"""
        if hasattr(x, "data_ptr"):
            return x.data_ptr() % 16 == 0
        elif isinstance(x, int):
            return x % 16 == 0
        if x is None:
            return True
        return False

    @staticmethod
    def is_equal_to_1(x):
        """ Return if the argument is a constant 1"""
        return True if isinstance(x, int) and not isinstance(x, bool) and x == 1 else False

    @staticmethod
    def get_property_key(val, align):
        if align and AttrsDescriptor.is_divisible_by_16(val):
            return "D"
        if AttrsDescriptor.is_equal_to_1(val):
            return "1"
        return "N"

    def __repr__(self):
        return f"AttrsDescriptor.from_dict({self.to_dict()!r})"


# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
mlir_prototype_pattern = r"^\s*tt\.func\s+(?:public\s+)?(@\w+)(\((?:%\w+: [\S\s]+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*(attributes \{[\S\s]+\})?\s+\{\s*$"
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

mlir_arg_type_pattern = r'%\w+: ((?:[^,\s<)]+|<[^>]+>)+(?: {[^}]+})?),?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    tma = re.search(r'tt.nv_tma_desc = 1', x)
    if tma is not None:
        return 'nvTmaDesc'
    x = re.sub(r' {[^}]+}', '', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def _get_num_warps_from_ir_str(src: str):
    ttgir_num_warps_pattern = r'"triton_gpu.num-warps"\s?=\s?(\d+)\s?:'
    # TODO(jlebar): Using a regex to get num-warps is a hack, and will break if
    # e.g. someone has an instruction (not module) attribute named "num-warps".
    num_warps_matches = re.findall(ttgir_num_warps_pattern, src)
    assert len(num_warps_matches) == 1, "Expected exactly one match for num_warps"
    num_warps = int(num_warps_matches[0])
    return num_warps


class ASTSource:

    def __init__(self, fn, signature, constants=None, attrs=None) -> None:
        # Walk around inductor compat.
        def compat(constants, signature):
            if all(not isinstance(key, str) for key in constants):
                return constants, signature
            signature |= {key: 'constexpr' for key, _ in constants.items() if key not in signature}
            idx_map = list()
            for k, v in signature.items():
                idx_map.append((fn.arg_names.index(k), k))
            idx_map.sort()
            new_signature = dict()
            for _, key in idx_map:
                new_signature[key] = signature[key]
            signature = new_signature
            constexprs = find_paths_if(list(signature.values()), lambda _, val: val == "constexpr")
            constexprs = {path: constants[get_iterable_path(list(signature.keys()), path)] for path in constexprs}
            return constexprs, signature

        constants, signature = compat(constants, signature)
        self.fn = fn
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.constexprs = constants
        self.attrs = attrs
        if isinstance(self.signature, str):
            self.signature = {k: v.strip() for k, v in enumerate(self.signature.split(","))}
        else:
            for k in self.signature.keys():
                if not isinstance(k, str):
                    raise TypeError("Signature keys must be string")
        if self.constexprs is None:
            self.constexprs = {}
        if self.attrs is None:
            self.attrs = AttrsDescriptor()
        # this is the constexprs plus the specialized constants
        spec_constants = {self.fn.arg_names[k[0]]: v for k, v in self.attrs.get_constants().items() if len(k) == 1}
        self.constants = dict()
        if constants is not None:
            for k, v in constants.items():
                k = (fn.arg_names.index(k), ) if isinstance(k, str) else k
                assert isinstance(k, tuple)
                self.constants[k] = v

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        # Note - we stringify the keys here to allow sorting to work for cases
        # where constants have mixed int/str keys.
        sorted_constants = sorted((str(k), v) for k, v in self.constexprs.items())
        key = f"{self.fn.cache_key}-{self.attrs.hash()}-{sorted_sig}-{sorted_constants}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, options, codegen_fns, module_map, context):
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.src = path.read_text()
        match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
        self.name = match.group(1)
        signature = match.group(2)
        types = re.findall(arg_type_pattern[self.ext], signature)
        self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, options, codegen_fns, module_map, context):
        module = ir.parse_mlir_module(self.path, context)
        module.context = context
        return module

    def parse_options(self):
        if self.ext == "ttgir":
            return {'num_warps': _get_num_warps_from_ir_str(self.src)}
        return dict()


@functools.lru_cache()
def triton_key():
    import pkgutil
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]
    # compiler
    path_prefixes = [
        (os.path.join(TRITON_PATH, "compiler"), "triton.compiler."),
        (os.path.join(TRITON_PATH, "backends"), "triton.backends."),
    ]
    for path, prefix in path_prefixes:
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.sha256(f.read()).hexdigest()]

    # backend
    libtriton_hash = hashlib.sha256()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.walk_packages([language_path], prefix="triton.language."):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    return f'{__version__}' + '-'.join(contents)


def parse(full_name, ext, context):
    if ext in ["ttir", "ttgir", "linalg", "linalgopt", "mluir", "mluiropt"]:
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext in ["llir", "ptx", "mlisa"]:
        return Path(full_name).read_text()
    if ext in ["cubin", "cnbin"]:
        return Path(full_name).read_bytes()


def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if os.getenv("TRITON_FRONT_END_DEBUGGING", "0") == "1":
        return

    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


def compile(src, target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        src = IRSource(src)
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    env_vars = get_cache_invalidating_env_vars()
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
    if not always_compile and metadata_path is not None:
        # cache hit!
        metadata = json.loads(Path(metadata_path).read_text())
        return CompiledKernel(src, metadata_group, hash)
    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()
    # try:
    module = src.make_ir(options, codegen_fns, module_map, context)
    # except Exception as e:
    #     filter_traceback(e)
    #     raise
    use_ir_loc = os.environ.get("USE_IR_LOC", None)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if (fn_override_manager is not None and (full_name := fn_override_manager.get_file(ir_filename)) is not None):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    context.disable_multithreading()
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target):
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)


class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self) -> None:
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))


class AsmDict(dict):

    def __missing__(self, key):

        if key == "sass":
            value = get_sass(self["cubin"])
        else:
            raise KeyError("Unknown key: '%s'" % key)

        self[key] = value
        return value


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    # TODO: move out of this namespace since it's a runtime thing
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        if self.metadata.onchip_mem_analysis:
            self.onchip_mem_cal = backend.get_estimate_onchip_memory_usage_fn(self.asm["onchip_mem_analysis"],
                                                                              self.metadata.name)
            return

        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None

    def _init_handles(self):
        if self.module is not None:
            return
        device = driver.active.get_current_device()
        # create launcher
        self.run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        if self.metadata.shared > max_shared:
            raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)

    def __getattribute__(self, name):
        if name == 'run':
            self._init_handles()
        return super().__getattribute__(name)

    def launch_metadata(self, grid, stream, *args):
        if CompiledKernel.launch_enter_hook is None:
            return None
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {}
        arg_idx = 0
        for i, arg_name in enumerate(self.src.fn.arg_names):
            if i in self.src.fn.constexprs:
                arg_dict[arg_name] = self.src.constexprs[arg_name]
            else:
                arg_dict[arg_name] = args[arg_idx]
                arg_idx += 1
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            filtered_args = tuple(arg for arg in args if arg is not None)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *filtered_args)

        return runner
