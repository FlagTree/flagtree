from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from triton.runtime import driver
from triton.runtime.cache import get_dump_manager
from dataclasses import dataclass
import functools
from typing import Any, Union, Tuple, Dict
from types import ModuleType
from pathlib import Path
import tempfile
import os
import subprocess
import hashlib
import ctypes
from typing import Optional

from triton.backends.ascend.utils import downgrade_llir, _get_llvm_path, _get_mlir_path, _get_triton_adapter_opt_path, \
    _get_kernel_target, _get_npucompiler_path, _is_ascend_sanitizer_enabled


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    # return lambda lhsType, rhsType: (16, 16, 16)
    return lambda lhsType, rhsType: (1, 1, 1)


def make_ttir(mod, metadata, opt):
    if 'hash' not in metadata:
        metadata['hash'] = hashlib.md5(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    if opt.debug:
        dump_manager = get_dump_manager(metadata['hash'])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)

    return mod


def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        cmd_list = [
            triton_adapter_opt_path, src_path, f'--triton-to-linalg=global-kernel=false named-ops={named_ops}', "-o",
            dst_path
        ]
        if _is_ascend_sanitizer_enabled():
            cmd_list += ["--mlir-print-debuginfo"]  # pass debug info

        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(dst_path).read_text(), "kernel.ttadapter.mlir", binary=False)

        return Path(dst_path).read_text()


def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(ttadapter_path).write_text(linalg)
        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")
        # TritonAdapter-MLIR to LLVM-MLIR
        subprocess.check_call([
            mlir_opt_path, ttadapter_path, "--convert-linalg-to-affine-loops", "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor", "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine", "--convert-linalg-to-loops", "--convert-scf-to-cf", "--convert-cf-to-llvm",
            "--convert-arith-to-llvm", "--convert-math-to-llvm", "--convert-complex-to-llvm",
            "--convert-vector-to-llvm", "--convert-index-to-llvm", "--memref-expand", "--expand-strided-metadata",
            "--finalize-memref-to-llvm", "--convert-func-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine", "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts", "-o", llmlir_path
        ])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(llmlir_path).read_text(), "kernel.llir.mlir", binary=False)

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(llir_path).read_text(), "kernel.ll", binary=False)

        return Path(llir_path).read_text()


def llir_to_cpuasm(llir: str, metadata, opt):
    # add metadata at final stage
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, CPU backend requires no limit on shared memory size
    metadata['shared'] = 1
    # We can get a function name (C naming) from
    # LLVM-IR by getting the first "define void @".
    fn_name = llir.split("define void @")[1].split("(")[0].strip()
    metadata['name'] = fn_name + " cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        linked_path = os.path.join(tmpdir, "kernel_linked.ll")
        dst_path = os.path.join(tmpdir, "kernel.s")

        llir = downgrade_llir(llir)
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(llir, "kernel_downgrade.ll", binary=False)

        Path(src_path).write_text(llir)

        linker_path = _get_llvm_path("bin", "llvm-link")
        libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
        subprocess.check_call([linker_path, src_path, libclc_path, "--only-needed", "-S", "-o", linked_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(linked_path).read_text(), "kernel_linked.ll", binary=False)

        llc_path = _get_llvm_path("bin", "llc")
        subprocess.check_call([llc_path, linked_path, "-o", dst_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata['hash'])
            dump_manager.put(Path(dst_path).read_text(), "kernel.s", binary=False)

        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


def linalg_to_bin_enable_npu_compile(linalg: str, metadata, opt):
    import re
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata['shared'] = 1
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata['mix_mode'] = re.search(r'mix_mode\s*=\s*"([^"]+)"', linalg).group(1)
    metadata['kernel_name'] = re.search(r'func\.func\s+@(\w+)', linalg).group(1)
    # Use while space to split kernel_name and mix_mode.
    # Check the function load_binary in npu_driver.py.
    metadata['name'] = metadata['kernel_name'] + " " + metadata['mix_mode']
    # remove the mix_mode attribute
    linalg = re.sub(r', mix_mode\s*=\s*"[^"]*"', '', linalg)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel_reloc.o")
        callback_path = os.path.join(tmpdir, "libkernel.so")
        multibuffer = metadata['multibuffer']
        _compile_option_list = [
            f"--enable-auto-multi-buffer={multibuffer}",
        ]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        npu_compiler_path = _get_npucompiler_path()
        if (npu_compiler_path.endswith("bishengir-compile")):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-hivm-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
        cmd_list = [npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file]
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            callback_func = getattr(lib, metadata['kernel_name'] + "_infer_workspace_shape_function")
            callback_func.restype = ctypes.c_int64
            callback_func.argtypes = []
            metadata['workspace_size'] = callback_func()

        return Path(bin_path).read_bytes()


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = 2
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    enable_npu_compile: bool = True
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    multibuffer: bool = True

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = -1

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class HuaweiBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu' or target.backend == 'npu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if (target.backend == "cpu"):
            self.binary_ext = "cpuasm"
        elif (target.backend == "npu"):
            self.binary_ext = "npubin"

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == 'npu':
            args = {k: opts[k] for k in NPUOptions.__dataclass_fields__.keys() if k in opts}
            options = NPUOptions(**args)
        else:
            args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
            options = CPUOptions(**args)
        return options

    def pack_metadata(self, metadata):
        from triton.backends.ascend.utils import TRITON_PROFILER_REGISTERED
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Huawei backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig, mix_mode = metadata.name.split()
        if (len(kernel_name_orig) > KERNEL_NAME_MAX_LEN):
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
            # import warnings
            # # red = "\x1b[31;20m"
            # # reset = "\x1b[0m"
            # warnings.warn(kernel_name_orig + " is truncated to " + kernel_name)
            # warnings.warn("because '" + kernel_name_orig + "' exceeds torchnpu profiler's length limit < 50")
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "profiler_registered": TRITON_PROFILER_REGISTERED,
        }

    def get_codegen_implementation(self):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def load_dialects(self, ctx):
        pass

    def add_stages(self, stages, options):
        if self.target.backend == 'npu':
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.enable_npu_compile:
                stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
                stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options)
            else:
                pass
        else:
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
            stages["llir"] = lambda src, metadata: linalg_to_llir(src, metadata, options)
            stages["cpuasm"] = lambda src, metadata: llir_to_cpuasm(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
