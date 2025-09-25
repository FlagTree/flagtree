import pickle
import ctypes
import functools
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
#20250923ph
from triton.backends.aipu import transform, analysis
#from triton.backends.aipu.codegen import codegenAIPU
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, aipu, passes
#import triton._C.libaipu_interface as aipu_interface
from mlir.passmanager import PassManager
from mlir.ir import Context, Module

from triton.backends.aipu.utils import (
    _check_bishengir_api_change,
    _check_bishengir_is_regbased,
    _enable_unpublished_feature,
    _get_npucompiler_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
)
from triton.backends.aipu.driver import NPUUtils

from triton.backends.compiler import (
    AttrsDescriptor,
    BaseBackend,
    GPUTarget,
    register_descriptor,
)


def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


@dataclass(frozen=True)
class AIPUOptions:
    vector_register_bits: int = 256
    num_tecs: int = 4
    num_stages: int = 2
    num_cores: int = 3
    cluster_dims: tuple = (1, 1, 1)
    arch: str = "x2"
    backend_name: str = "aipu"
    debug: bool = False
    sanitize_overflow: bool = True
    num_warps: int = 4
    num_ctas: int = -1
    num_buffers_warp_spec: int = -1
    num_consumer_groups: int = -1
    reg_dec_producer: int = -1
    reg_inc_consumer: int = -1
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )

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
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    multibuffer: bool = None
    enable_hivm_auto_cv_balance: bool = None
    unit_flag: bool = None
    inject_barrier_all: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None

    stream: int = None

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class AscendAttrsDescriptor(AttrsDescriptor):

    # For now we collect shapes of tensor at runtime.
    # We comment out the following func but keep it for future reference.
    def _add_backend_properties(self, params=None, values=None):
        pass


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'
    REMOVE_MIX_MODE_REGEX = r', mix_mode\s*=\s*"[^"]*"'

    metadata["shared"] = 1
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    metadata["name"] = metadata["kernel_name"] + " " + metadata["mix_mode"]
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    linalg = re.sub(REMOVE_MIX_MODE_REGEX, "", linalg)
    return linalg, metadata


def linalg_to_bin_enable_npu_compile(linalg: str, metadata, opt):

    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        if _check_bishengir_is_regbased():
            bishengir_hivm_opt = "--reg-based=true"
        else:
            bishengir_hivm_opt = "--enable-hivm-compile=true"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = []
        if _enable_unpublished_feature():
            _compile_option_list += [
                f"--target={NPUUtils().get_arch()}",
            ]
        multibuffer = opt.multibuffer
        if multibuffer is not None:
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multibuffer}",
            ]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        enable_hivm_auto_cv_balance = opt.enable_hivm_auto_cv_balance
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        unit_flag = opt.unit_flag
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        inject_barrier_all = opt.inject_barrier_all
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = opt.limit_auto_multi_buffer_only_for_local_buffer
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = opt.set_workspace_multibuffer
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        tile_mix_vector_loop = opt.tile_mix_vector_loop
        if tile_mix_vector_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-vector-loop={tile_mix_vector_loop}"]

        tile_mix_cube_loop = opt.tile_mix_cube_loop
        if tile_mix_cube_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-cube-loop={tile_mix_cube_loop}"]

        auto_multi_buffer = opt.limit_auto_multi_buffer_of_local_buffer
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        if _is_auto_map_parallel_blocks_enabled():
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                "--enable-triton-kernel-compile=true",
            ]
        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'npu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        self.binary_ext = "npubin"
        #aipu_interface.passes.register_all_passes()

    def parse_options(self, opts) -> Any:
        return AIPUOptions(**{k: opts[k] for k in AIPUOptions.__dataclass_fields__.keys() if k in opts})

    def pack_metadata(self, metadata):
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig, mix_mode = metadata.name.split()
        if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "tensor_kinds": metadata.tensor_kinds,
        }

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.aipu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        aipu.load_dialects(ctx)

    def get_arg_specialization(*arg, **kwargs):
        return None

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_linalg(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        aipu.passes.convert.add_triton_to_linalg_pipeline(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_npubin(mod, metadata, opt):

        linalg_str = str(mod)
        metadata.update({
            "enable_nd2nz_on_vector": opt.enable_nd2nz_on_vector,
            "multibuffer": opt.multibuffer,
            "enable_hivm_auto_cv_balance": opt.enable_hivm_auto_cv_balance,
            "unit_flag": opt.unit_flag,
            "inject_barrier_all": opt.inject_barrier_all,
            "limit_auto_multi_buffer_only_for_local_buffer": opt.limit_auto_multi_buffer_only_for_local_buffer,
            "limit_auto_multi_buffer_of_local_buffer": opt.limit_auto_multi_buffer_of_local_buffer,
            "set_workspace_multibuffer": opt.set_workspace_multibuffer,
            "tile_mix_vector_loop": opt.tile_mix_vector_loop,
            "tile_mix_cube_loop": opt.tile_mix_cube_loop,
        })
        return linalg_to_bin_enable_npu_compile(linalg_str, metadata, opt)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["npubin"] = (lambda src, metadata: linalg_to_bin_enable_npu_compile(src, metadata, options))

    @functools.lru_cache()
    def hash(self):
        return "aipu_builder"
