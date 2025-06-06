from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from triton.runtime.cache import get_cache_manager
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


def _get_tsm_opt_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "tsingmicro-opt")


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_SYSPATH", "")
    if path == "":
        raise Exception("LLVM_SYSPATH is not set.")
    return os.path.join(path, "bin", bin_name)


def _get_tx8_path(sub_name: str) -> str:
    path = os.getenv("TX8_HOME", "")
    if path == "":
        raise Exception("TX8_HOME is not set.")
    return os.path.join(path, sub_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_DUMP_PATH", "")
    if not path:
        return

    os.makedirs(path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


# Build a accelerator controller ELF
def compile_accelerator():
    # TODO : cache mechanism
    # name = "npu_" + name
    # key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    # cache = get_cache_manager(key)
    # cache_path = cache.get_file(f"{name}.so")

    # if cache_path is None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # FIXME: Hardcoded path
        #dst_path = os.path.join(tmpdir, f"{name}.so")
        dst_path = "/tmp/kernel.so"
        libc_lib = os.path.join(_get_tx8_path("Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"), "riscv64-unknown-elf",
                                "lib", "rv64imafdc", "lp64d")
        libvr_path = os.path.join(os.path.dirname(__file__), "lib")
        clang_path = _get_llvm_bin_path("clang")
        lld_path = _get_llvm_bin_path("ld.lld")
        tx8_lib = _get_tx8_path("lib")
        subprocess.check_call([
            clang_path, "-shared", "--target=riscv64-unknown-linux-gnu", "-march=rv64imafdc", "-O2",
            f"-fuse-ld={lld_path}", "-nostdlib", "-nostartfiles", "-Wl,--allow-shlib-undefined", "-mabi=lp64d",
            "-Wl,--no-dynamic-linker",
            # FIXME: Hardcoded path
            "/tmp/kernel.o", f"-L{libvr_path}", f"-L{libc_lib}", f"-L{tx8_lib}", "-Wl,--whole-archive",
            "-linstr_tx81",  # Tx81 intrinsic API
            "-lvr",  # Wrapper API of Tx81 intrinsic
            "-Wl,--no-whole-archive", "-lm", "-o", dst_path
        ])

        _dump_ir_if_needed([dst_path])
        with open(dst_path, 'rb') as f:
            so = f.read()
        return so


def _ttir_to_coreir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "core.mlir")
        Path(src_path).write_text(ttir_code)
        tsm_opt_path = _get_tsm_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            tsm_opt_path, src_path, "--triton-to-core-dialects", "--one-shot-bufferize=allow-return-allocs-from-loops",
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
        tsm_opt_path = _get_tsm_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            tsm_opt_path, src_path, "--core-dialects-to-mk",
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
        tsm_opt_path = _get_tsm_opt_path()
        _dump_ir_if_needed([src_path])
        subprocess.check_call([
            tsm_opt_path, src_path, "--expand-strided-metadata",
            "--lower-affine",  # convert affine.load to memref.load, need exec before tx81-to-llvm since we will support spm offset to memref.load
            "--mk-to-tx81", "--cse",  # unused memref.subview/memref.reinterpret
            #"--mlir-print-debuginfo",
            "-o", dst_path
        ])
        return Path(dst_path).read_text()


def _optimize_txir(txir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return txir


def _txir_to_llir(mod, metadata):
    txir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tx.mlir")
        llvmir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(src_path).write_text(txir_code)
        tsm_opt_path = _get_tsm_opt_path()
        _dump_ir_if_needed([src_path])
        # Tx81 and core dialects to LLVM-MLIR
        args = [
            tsm_opt_path, src_path,
            # Use tx81-memref-to-llvm to replace "--finalize-memref-to-llvm".
            "--tx81-memref-to-llvm", "--convert-scf-to-cf", "--convert-math-to-llvm",
            "--convert-cf-to-llvm",  # need exec before "convert-func-to-llvm"
            "--convert-func-to-llvm",  # need exec before "kernel-arg-buffer", otherwise un-rank memref will translate to int(rank) + ptr
        ]

        args.append(
            "--kernel-arg-buffer"
        )  # need exec before "tx81-to-llvm" which will declare other func. We want only replace the triton kernel

        # other pass
        args += [
            "--tx81-to-llvm", "--convert-arith-to-llvm",  # need exec last since arith.const conversion
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts", "--canonicalize",
            #"--mlir-print-debuginfo",
            "-o", llvmir_path
        ]

        subprocess.check_call(args)

        _dump_ir_if_needed([llvmir_path])

        llvm_file = os.getenv("CUSTOMIZED_IR", "")
        if (llvm_file != ""):
            llvmir_path = os.getenv("TRITON_DUMP_PATH", "")

            if not llvmir_path:
                return

            llvmir_path = os.path.join(llvmir_path, llvm_file)

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
            clang_path, src_path, "-O2", "-c", "-fPIC", "--target=riscv64-unknown-linux-gnu", "-march=rv64imafdc", "-o",
            dst_path
        ])

        _dump_ir_if_needed([dst_path])

        # compile kernel and intrinsic wrapper to shared library
        return compile_accelerator()


@dataclass(frozen=True)
class TXDAOptions:
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
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class TXDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'txda'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in TXDAOptions.__dataclass_fields__.keys() if k in opts})
        return TXDAOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (metadata.num_warps, metadata.num_ctas, metadata.shared, metadata.cluster_dims[0],
                metadata.cluster_dims[1], metadata.cluster_dims[2], metadata.name)

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
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
        stages["llir"] = lambda src, metadata: _optimize_llir(_txir_to_llir(src, metadata))
        stages["so"] = lambda src, metadata: _llir_to_bin(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
