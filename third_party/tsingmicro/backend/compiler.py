from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
from pathlib import Path


def _get_ztc_opt_path() -> str:
    path = os.getenv("ZTC_OPT_PATH", "")
    if path == "":
        raise Exception("ZTC_OPT_PATH is not set.")
    return path


def _get_vendor_runtime_path() -> str:
    path = os.getenv("LIB_VENDOR_RUNTIME_PATH", "")
    if path == "":
        raise Exception("LIB_VENDOR_RUNTIME_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


# The riscv c header files and libraries path.
def _get_libc_root() -> str:
    path = os.getenv("LIB_C_ROOT", "")
    if path == "":
        raise Exception("LIB_C_ROOT is not set.")
    return path


def _dump_ir_if_needed(files):
    path = os.getenv("ZTC_DUMP_PATH", "")
    if not path:
        return

    os.makedirs(path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _ttir_to_coreir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "core.mlir")
        Path(src_path).write_text(ttir_code)
        ztc_opt_path = _get_ztc_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            ztc_opt_path, src_path, "--triton-to-core-dialects", "--one-shot-bufferize",
            #"--mlir-print-debuginfo",
            "-o", dst_path
        ])
        return Path(dst_path).read_text()


def _optimize_coreir(coreir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return coreir


def _coreir_to_mkir(mod):
    # Get core dialects as string
    coreir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "core.mlir")
        dst_path = os.path.join(tmpdir, "mk.mlir")
        Path(src_path).write_text(coreir_code)
        ztc_opt_path = _get_ztc_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            ztc_opt_path, src_path, "--core-dialects-to-mk",
            #"--mlir-print-debuginfo",
            "-o", dst_path
        ])
        return Path(dst_path).read_text()


def _optimize_mkir(mkir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return mkir


def _coreir_to_txir(mod):
    # Get core dialects as string
    coreir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "core.mlir")
        dst_path = os.path.join(tmpdir, "tx.mlir")
        Path(src_path).write_text(coreir_code)
        ztc_opt_path = _get_ztc_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            ztc_opt_path, src_path, "--expand-strided-metadata", "--mk-to-tx81",
            "--lower-affine",  # convert affine.load to memref.load, need exec before tx81-to-llvm since we will support spm offset to memref.load
            "--cse",  # unused memref.subview/memref.reinterpret
            #"--mlir-print-debuginfo",
            "-o", dst_path
        ])
        return Path(dst_path).read_text()


def _optimize_txir(txir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return txir


def _txir_to_llir(mod):
    txir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tx.mlir")
        llvmir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(src_path).write_text(txir_code)
        ztc_opt_path = _get_ztc_opt_path()
        _dump_ir_if_needed([src_path])
        # Tx81 and core dialects to LLVM-MLIR
        subprocess.check_call([
            ztc_opt_path, src_path, "--tx81-memref-to-llvm", "--tx81-to-llvm", "--convert-scf-to-cf",
            "--convert-math-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm",
            # Use tx81-memref-to-llvm custom pass for now.
            # "--finalize-memref-to-llvm",
            "--convert-arith-to-llvm",  # need exec last since arith.const conversion
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts", "--canonicalize",
            #"--mlir-print-debuginfo",
            "-o", llvmir_path
        ])
        _dump_ir_if_needed([llvmir_path])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llvmir_path, "--mlir-to-llvmir", "-o", llir_path])

        _dump_ir_if_needed([llir_path])
        return Path(llir_path).read_text()


def _mkir_to_llir(mkir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        mkir_path = os.path.join(tmpdir, "mk.mlir")
        llvmir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(mkir_path).write_text(mkir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        # MagicKernel-MLIR to LLVM-MLIR
        subprocess.check_call([
            mlir_opt_path, mkir_path, "--convert-linalg-to-affine-loops",
            # Note: eliminate-empty-tensors fails when there are multiple func.return ops
            # in a single kernel which are the results of early returns.
            # See python/examples/test_early_return.py for examples.
            # We disable this pass for now since performance on CPU isn't the main
            # focus at the moment.
            # "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor", "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine", "--convert-linalg-to-loops", "--expand-strided-metadata", "--convert-scf-to-cf",
            "--convert-arith-to-llvm", "--convert-math-to-llvm", "--convert-complex-to-llvm",
            "--convert-vector-to-llvm", "--convert-index-to-llvm", "--memref-expand", "--finalize-memref-to-llvm",
            "--convert-func-to-llvm", "--convert-cf-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine", "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--canonicalize", "--reconcile-unrealized-casts", "--mlir-print-debuginfo", "-o", llvmir_path
        ])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llvmir_path, "--mlir-to-llvmir", "-o", llir_path])
        _dump_ir_if_needed([mkir_path, llvmir_path, llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        # FIXME: Hardcoded path
        #dst_path = os.path.join(tmpdir, "kernel.so")
        dst_path = "/tmp/kernel.o"
        Path(src_path).write_text(llir)
        clang_path = _get_llvm_bin_path("clang++")
        subprocess.check_call([
            clang_path, src_path, "-O2", "-c", "-fPIC", "--target=riscv64-unknown-elf", "-march=rv64imafdc", "-o",
            dst_path
        ])

        _dump_ir_if_needed([dst_path])
        with open(dst_path, 'rb') as f:
            so = f.read()
        return so


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (metadata.num_warps, metadata.num_ctas, metadata.shared, metadata.cluster_dims[0],
                metadata.cluster_dims[1], metadata.cluster_dims[2], metadata.name)

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `ztc.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, opt):
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
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["coreir"] = lambda src, metadata: _optimize_coreir(_ttir_to_coreir(src))
        # stages["mkir"] = lambda src, metadata: _optimize_mkir(_coreir_to_mkir(src))
        stages["txir"] = lambda src, metadata: _optimize_txir(_coreir_to_txir(src))
        stages["llir"] = lambda src, metadata: _optimize_llir(_txir_to_llir(src))
        stages["so"] = lambda src, metadata: _llir_to_bin(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
