from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, mlu
from triton.backends.mlu.driver import default_neuware_dir
from triton.runtime.errors import OutOfResources

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional, Dict
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import warnings
import subprocess
from pathlib import Path


def path_to_binary(binary: str):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [os.environ.get(f"TRITON_{binary.upper()}_PATH", ""), os.path.join(default_neuware_dir(), "bin", binary)]

    for p in paths:
        bin = p.split(" ")[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*" + binary + "( version:)? (\d+\.\d+\.\d+).*", result.decode("utf-8"),
                                    flags=re.MULTILINE)
                if version is not None:
                    return p, version.group(2)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_cnas_version():
    _, version = path_to_binary("cnas")
    return version


MIN_REQUIRED_CNTOOLKIT_VERSION = "4.1.0"


def check_cntoolkit_version():
    from packaging import version
    version_file = os.path.join(default_neuware_dir(), "version.txt")
    if not os.path.exists(version_file):
        return warnings.warn(f"{version_file} is not found, please install cntoolkit-cloud")

    with open(version_file, 'r') as f:
        line = f.readline().strip()
    match = re.search(r"Version\s+([\d.]+)", line)
    if match:
        cntoolkit_version = match.group(1)
    else:
        raise RuntimeError(f"Cannot find cntoolkit version")

    if version.parse(cntoolkit_version) < version.parse(MIN_REQUIRED_CNTOOLKIT_VERSION):
        raise RuntimeError(
            f"cntoolkit version {cntoolkit_version} is lower than required {MIN_REQUIRED_CNTOOLKIT_VERSION}")


check_cntoolkit_version()


def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


def _extract_memory_info(log: str) -> dict:
    pattern = r'(NRAM|WRAM|SRAM)\s+(\d+)\s+([-]?\d+)\s+(\d+)\s+'
    return re.findall(pattern, log)


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@functools.lru_cache()
def default_libdevice_dir():
    return str(Path(__file__).parent / "lib")


@dataclass(frozen=True)
class MLUOptions:
    num_warps: int = 1
    num_stages: int = 0
    cnas_version: int = None
    enable_soft_i64: bool = False
    is_linear: bool = False
    kernel_name: str = None

    # These options only used in the GPU, here, we are just setting default values.
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None

    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dtypes: Tuple[str] = ()
    allow_half_div: bool = True
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    precision_mode: str = "precision"
    backend_name: str = "mlu"
    sanitize_overflow: bool = False
    opt_level: str = "O3"
    restrict_ptr: bool = None
    restrict_ptr_hint: bool = False
    can_promote_shared: bool = False
    force_use_shared_memory: bool = False
    # Default bottleneck set to I/O, default behavior for software pipeline.
    bottleneck: str = None
    pipeline_strategies: Tuple[str] = ()
    onchip_mem_analysis: str = False
    # Eanble internal mlu instruction bound check, it will slow down the running
    # speed, only used for debug.
    enable_mlu_bound_check: bool = False

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        assert self.bottleneck in [None, "io", "mv", "simd"]
        if self.num_warps not in [1, 4, 8, 16, 32]:
            warnings.warn("num_warps should in 1/4/8/16/32 for mlu backend")
        assert self.opt_level in ["O0", "O1", "O2", "O3", "Om", "Os"]

        # Only block and u1 task are supported.
        if self.num_warps > 4:
            warnings.warn(
                f"num_warps is currently set to {self.num_warps}; values greater "
                f"than 4 are not supported, falling back to 4", UserWarning)
            object.__setattr__(self, 'num_warps', 4)

        # Fallback to 1 if num_warps set to 2.
        if self.num_warps == 2:
            warnings.warn("num_warps equals to 2 is not supported currently, "
                          "fallback to 1 if encountered.", UserWarning)
            object.__setattr__(self, 'num_warps', 1)

        if self.debug is None:
            object.__setattr__(self, 'debug', False)

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MLUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'mlu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "cnbin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in MLUOptions.__dataclass_fields__.keys() if k in opts}
        # When arch is less than mtp_5xx, tf32 is not supported, use fp32 for calculation.
        if "allowed_dot_input_precisions" not in args:
            if self.capability < 500:
                args["allowed_dot_input_precisions"] = ("ieee")

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(MLUOptions.supported_fp8_dtypes)
            if self.capability >= 600:
                supported_fp8_dtypes = supported_fp8_dtypes.union(("fp8e5", "fp8e4nv"))
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        args["max_num_imprecise_acc_default"] = 0

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

        if "enable_mlu_bound_check" not in args:
            args["enable_mlu_bound_check"] = os.getenv("TRITON_ENABLE_MLU_BOUND_CHECK", "0") == "1"
        return MLUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.promote_shared,
        )

    def get_codegen_implementation(self):
        codegen_fns = {"convert_custom_types": lambda arg, dst_ty: arg, "min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.mlu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def get_attrs_descriptor(self, params, args):
        """
        Return an attribute descriptor: given a set of parameters and arguments
        the descriptor stores a set of compile time properties that can improve code
        generation. Different backends might benefit from different properties
        """
        from triton.compiler import AttrsDescriptor
        return AttrsDescriptor(params, args)

    def compute_spec_key(self, arg, align):
        """
        Return the ascii key for a given argument with a given set of properties
        """
        from triton.compiler import AttrsDescriptor
        return AttrsDescriptor.get_property_key(arg, align)

    def load_dialects(self, ctx):
        mlu.load_dialects(ctx)

    @staticmethod
    def set_num_warps(mod: Any, num_warps: int, builder):
        '''
        Set num warps on triton module.
        :param: mod: tt ir module.
        :num_warps: num warps set by user, it will attach attributes
                    triton.xpe = num_warps > 1 ? 4 : 1 and
                    triton.xtask = num_warps / 4.
        '''
        if num_warps >= 4:
            mod.set_attr("triton.xpe", builder.get_int32_attr(4))
            mod.set_attr("triton.xtask", builder.get_int32_attr(num_warps // 4))
        else:
            mod.set_attr("triton.xpe", builder.get_int32_attr(1))

    @staticmethod
    def ttir_get_kernel_info(ttir: str) -> dict:
        '''
        Get kernel info from ttir.
        '''
        info = dict(kernel_name='', contain_readperf=False)
        for line in ttir.split('\n'):
            line = line.strip()
            if line.startswith('tt.func public'):
                info['kernel_name'] = line.split('@')[1].split("(")[0]
            if line.startswith('mlu.readperf'):
                info['contain_readperf'] = True
        return info

    @staticmethod
    def stringify_arch(capability):
        return f'mtp_{capability}'

    @staticmethod
    def onchip_mem_analysis(mod, opt):
        return mlu.analysis_onchip_mem_usage(mod, opt)

    @staticmethod
    def get_estimate_onchip_memory_usage_fn(code: str, func_name: str):
        namespace = {}
        exec(code, namespace, namespace)
        if func_name in namespace:
            return namespace[func_name]
        else:
            raise ValueError(f"not found function '{func_name}'")

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        mlu.passes.add_arith_canonicalizer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)

        kernel_info = MLUBackend.ttir_get_kernel_info(str(mod))
        if not kernel_info['kernel_name']:
            raise RuntimeError("can not get kernel name from ttir")

        if kernel_info['contain_readperf'] and (opt.num_warps > 1 or opt.num_stages > 1):
            raise RuntimeError("the readperf op is not allowed when num_warps or num_stages is greater than 1")

        # Set the required fields for metadata.
        metadata["name"] = kernel_info['kernel_name']
        metadata["shared"] = 0

        builder = ir.builder(mod.context)
        MLUBackend.set_num_warps(mod, opt.num_warps, builder)

        mod.set_attr("tt.num_stages", builder.get_int32_attr(opt.num_stages))
        if opt.bottleneck:
            mod.set_attr("tt.bottleneck_stream", builder.get_str_attr(opt.bottleneck))
        if opt.pipeline_strategies is not None:
            mod.set_attr("tt.pipeline_strategies", builder.get_str_array_attr(opt.pipeline_strategies))
        if capability < 600:
            mod.set_attr("triton.enable_soft_i64", builder.get_bool_attr(opt.enable_soft_i64))
        else:
            if opt.enable_soft_i64:
                warnings.warn("Ignore enable_soft_i64 for capability {capability}")

        mod.set_attr("triton.is_linear", builder.get_bool_attr(opt.is_linear))
        if opt.kernel_name is not None:
            mod.set_attr("triton.kernel_name", builder.get_str_attr(opt.kernel_name))
        if opt.restrict_ptr is not None:
            mod.set_attr("genesis.restrict_ptr", builder.get_bool_attr(opt.restrict_ptr))
        elif opt.restrict_ptr_hint:
            mod.set_attr("genesis.restrict_ptr_hint", builder.get_bool_attr(True))
        promote_shared = opt.can_promote_shared and opt.force_use_shared_memory
        metadata["promote_shared"] = promote_shared
        mod.set_attr("genesis.promote_shared", builder.get_bool_attr(promote_shared))

        assert opt.precision_mode in ["fast", "precision"]

        mod.set_attr("genesis.assert", builder.get_bool_attr(opt.enable_mlu_bound_check))
        mod.set_attr("genesis.precision_mode", builder.get_str_attr(opt.precision_mode))
        arch = MLUBackend.stringify_arch(capability)
        mod.set_attr("genesis.arch", builder.get_str_attr(arch))

        return mod

    @staticmethod
    def make_linalg(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_canonicalizer(pm)
        mlu.passes.add_hot_cold_splitting(pm)
        mlu.passes.add_wrap_func_body_with_single_block(pm)
        mlu.passes.add_inliner(pm)
        mlu.passes.add_conservate_pointer_mode_set(pm)
        passes.common.add_canonicalizer(pm)
        mlu.passes.add_canonicalize_triton(pm)
        mlu.passes.add_pointer_strength_reduction(pm)
        mlu.passes.add_pointer_contiguity_enhancement(pm)
        mlu.passes.add_pointer_constancy_degeneration(pm)
        mlu.passes.add_refine_elementwise_symbol_attr(pm)
        mlu.passes.add_canonicalize_triton(pm)
        mlu.passes.add_optimize_triangle_mask(pm)
        mlu.passes.add_triton_to_arith(pm)
        passes.common.add_canonicalizer(pm)
        mlu.passes.add_arith_canonicalizer(pm)
        mlu.passes.add_tensor_canonicalizer(pm)
        mlu.passes.add_arithext_to_linalg(pm)
        mlu.passes.add_triton_to_linalg(pm)
        passes.common.add_cse(pm)
        mlu.passes.add_extract_like_move_backward(pm)
        passes.common.add_canonicalizer(pm)
        mlu.passes.add_convert_scalar_i64_to_tensor(pm)
        passes.common.add_canonicalizer(pm)
        mlu.passes.add_arith_to_linalg(pm)
        mlu.passes.add_math_to_linalg(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        mlu.passes.add_wrap_func_body_with_single_block(pm)
        mlu.passes.add_convert_triton_to_scf(pm)
        mlu.passes.add_generate_triton_executalbe(pm)
        mlu.passes.add_set_attr_to_forop(pm)
        mlu.passes.add_bubble_up_load(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_optimized_linalg(mod, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        mlu.passes.add_auto_tile_pipeline(pm, opt.opt_level)
        pm.run(mod)
        return mod

    @staticmethod
    def make_mluir(mod, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        mlu.passes.add_post_process_pipeline(pm, opt.opt_level)
        pm.run(mod)
        return mod

    @staticmethod
    def make_optimize_mluir(mod, opt, capability):
        mlu.optimize_mluir(mod, opt, capability, default_libdevice_dir())
        return mod

    @staticmethod
    def make_mlisa(mod):
        pm = ir.pass_manager(mod.context, "builtin.module", ir.NESTING.IMPLICIT)
        pm.enable_debug()
        mlu.passes.serialize_to_mlisa(pm)
        pm.run(mod)

        return mlu.get_mlisa_from_module(mod).decode('utf-8')

    @staticmethod
    def make_cnbin(src, opt, capability):
        cnas, _ = path_to_binary("cnas")
        cnlink, _ = path_to_binary("cnlink")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.mlisa') as fsrc, \
                tempfile.NamedTemporaryFile(delete=False, mode='a+', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.cnbin'
            ffatbin = fsrc.name + '.cnfatbin'

            opt_level = opt.opt_level
            if opt_level == "Om":
                opt_level = "O3"
            debug = []
            # We only enable -g debugging when opt_level == O0.
            if opt_level == "O0" and opt.debug:
                debug = ['-g']
            arch = MLUBackend.stringify_arch(capability)
            line_info = [] if os.environ.get('TRITON_DISABLE_LINE_INFO') else ['-lineinfo']
            cnas_cmd = [cnas, *line_info, *debug, f'-{opt_level}', '--verbose', '-a', arch, '-i', fsrc.name, '-o', fbin]
            # FIXME: remove cnlink when cnModuleLoadData support cnbin.
            cnlink_cmd = [cnlink, '--fatbin', '-i', fbin, '-o', ffatbin]
            try:
                subprocess.run(cnas_cmd, check=True, close_fds=False, stdout=flog, stderr=flog)
                subprocess.run(cnlink_cmd, check=True, close_fds=False, stdout=flog, stderr=flog)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(fbin):
                    os.remove(fbin)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                meminfo = _extract_memory_info(log)
                for info in meminfo:
                    memory_type, used, avail, total = info
                    if int(avail) < 0:
                        raise OutOfResources(int(used), int(total), memory_type)

                raise RuntimeError(f'`cnas+cnlink` failed with error code {e.returncode}: \n{log}\n'
                                   f'Repro cnas command: {" ".join(cnas_cmd)}\n'
                                   f'Repro cnlink command: {" ".join(cnlink_cmd)}\n')

            with open(ffatbin, 'rb') as f:
                cnfatbin = f.read()
            if os.path.exists(ffatbin):
                os.remove(ffatbin)

        return cnfatbin

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, self.capability)
        if options.onchip_mem_analysis:
            stages["onchip_mem_analysis"] = lambda src, metadata: self.onchip_mem_analysis(src, options)
            return
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["linalgopt"] = lambda src, metadata: self.make_optimized_linalg(src, options)
        stages["mluir"] = lambda src, metadata: self.make_mluir(src, options)
        stages["mluiropt"] = lambda src, metadata: self.make_optimize_mluir(src, options, self.capability)
        stages["mlisa"] = lambda src, metadata: self.make_mlisa(src)
        stages["cnbin"] = lambda src, metadata: self.make_cnbin(src, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = get_cnas_version()
        return f'{version}-{self.capability}'
