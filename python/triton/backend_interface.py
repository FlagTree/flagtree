# TODO: 0625
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass


class GeneralBackend:
    @staticmethod
    def kernel_suffix(signature, specialization):
        # suffix format:
        # <argid><'c' if equal to 1><'d' if divisible by 16><'e' if divisible by 8>
        suffix = ''
        for i, _ in enumerate(signature):
            suffix += str(i)
            if i in specialization.equal_to_1:
                suffix += 'c'
            if i in specialization.divisible_by_16:
                suffix += 'd'
        return suffix

    @staticmethod
    def ast_to_ttir(fn, specialization, context, options, codegen_fns):
        from triton.compiler.code_generator import _get_fn_file_line, CodeGenerator
        from triton.language import str_to_ty
        from triton import language

        attrs = specialization.attrs
        # create kernel prototype
        cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in specialization.constants.items()}
        # visit kernel AST
        gscope = fn.__globals__.copy()
        function_name = fn.repr(specialization)
        tys = list(specialization.signature.values())
        new_constants = {k: True if k in tys and tys[k] == "i1" else 1 for k in attrs.equal_to_1}
        new_attrs = {k: [("tt.divisibility", 16)] for k in attrs.divisible_by_16}

        all_constants = constants.copy()
        all_constants.update(new_constants)
        arg_types = [str_to_ty(v) for k, v in specialization.signature.items() if k not in specialization.constants]
        file_name, begin_line = _get_fn_file_line(fn)

        prototype = language.function_type([], arg_types)
        generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants, function_name=function_name,
                                jit_fn=fn, attributes=new_attrs, is_kernel=True, file_name=file_name,
                                begin_line=begin_line, options=options, codegen_fns=codegen_fns)
        generator.visit(fn.parse())

        ret = generator.module
        # module takes ownership of the context
        ret.context = context
        return ret

    @dataclass
    class AttrsDescriptor:
        divisible_by_16: set = None
        equal_to_1: set = None

        def __post_init__(self):
            if self.divisible_by_16 is None:
                self.divisible_by_16 = set()
            if self.equal_to_1 is None:
                self.equal_to_1 = set()

        def to_dict(self):
            return {'divisible_by_16': list(self.divisible_by_16), 'equal_to_1': list(self.equal_to_1)}

        @staticmethod
        def from_dict(data):
            return GeneralBackend().AttrsDescriptor(divisible_by_16=set(data.get('divisible_by_16', [])),
                                equal_to_1=set(data.get('equal_to_1', [])))

        def hash(self):
            key = str([sorted(x) for x in self.__dict__.values()])
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def compile(src, target=None, options=None):
        from triton.compiler.compiler import (
            make_backend, ASTSource, IRSource, 
            triton_key, CompiledKernel,
            filter_traceback, parse
        )
        from triton.backends.compiler import GPUTarget
        from triton.runtime.driver import driver
        from triton._C.libtriton import get_cache_invalidating_env_vars, ir
        from triton.runtime.cache import get_cache_manager, get_dump_manager, get_override_manager

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
        metadata_filename = f"{src.name}.json"
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
        try:
            module = src.make_ir(options, codegen_fns, context)
        except Exception as e:
            filter_traceback(e)
            raise
        use_ttgir_loc = os.environ.get("USE_TTGIR_LOC", "0") == "1"
        for ext, compile_ir in list(stages.items())[first_stage:]:
            next_module = compile_ir(module, metadata)
            ir_filename = f"{src.name}.{ext}"
            metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
            if fn_dump_manager is not None:
                fn_dump_manager.put(next_module, ir_filename)
            if (fn_override_manager is not None and fn_override_manager.has_file(ir_filename)):
                print(f"\nOverriding kernel with file {ir_filename}")
                full_name = fn_override_manager.get_file(ir_filename)
                next_module = parse(full_name, ext, context)
            # use an env variable to parse ttgir from file
            if use_ttgir_loc and ext == "ttgir":
                ttgir_full_name = fn_cache_manager.get_file(ir_filename)
                next_module.create_location_snapshot(ttgir_full_name)
                print(f"Create new locations for {ttgir_full_name}")
            module = next_module
        # write-back metadata
        metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                                binary=False)
        fn_cache_manager.put_group(metadata_filename, metadata_group)
        # return handle to compiled kernel
        return CompiledKernel(src, metadata_group, hash)

    @staticmethod
    def get_language__all__():
        __all__ = [
            "libdevice", "globaltimer", "num_threads", "num_warps", "smid", "convert_custom_float8_sm70",
            "convert_custom_float8_sm80"
        ]
        return __all__

    @staticmethod
    def get_ops__all__():
        __all__ = ["blocksparse", "_cross_entropy", "cross_entropy", "_matmul", "matmul", "attention", "get_higher_dtype"]
        return __all__
