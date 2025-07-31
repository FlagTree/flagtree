import hashlib
import importlib.util
import os
import sys
import subprocess
import re
import tempfile
import hashlib
import torch

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
import triton.backends
import triton.language as tl
from triton.runtime.cache import get_cache_manager
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.runtime.driver import driver
from triton.backends.mlu.driver import default_neuware_dir, ty_to_cpp
from triton.backends.mlu.compiler import path_to_binary

desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cnbin`
data along with utilities of cnrt api.

signature is provided as a list of types or constexpr values, e.g.

`mlu_compile.py --kernel-name kernel --signature "*fp32, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that pointer of argument 0 and 1 are assumed to be multiple of 16, argument 2
is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

Note the hint value 16 of the scalar value means that the i32 value of argument 1 is multiples of 16.

The resulting entry point will have signature

cnrtRet_t kernel_{specialization_suffix}(cnrtQueue_t queue, cnrtDim3_t* dim, float* arg0, int32_t arg1, int32_t arg2);\n"

Different such specialized entry points can be combined using the `mlu_linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `mlu_compile.py` script
"""

ARCH_PREFIX = 'mtp_'


def get_mlu_backend(target):
    device_mlu = 'mlu'
    if device_mlu not in backends.keys():
        raise RuntimeError("Backend mlu is not supported!")
    return backends[device_mlu].compiler(target)


def get_target(isa_version=None):
    if isa_version is not None:
        return GPUTarget('mlu', isa_version, 0)
    return driver.active.get_current_target()


def get_arch(isa_version=None):
    """ Get mlu arch. """
    target = driver.active.get_current_target()
    arch = target.arch if isa_version is None else isa_version
    return f'{ARCH_PREFIX}{arch}'


def generate_arch_specific_filename(base_filename: Path, new_suffix: str, arch: str = None):
    if arch == None:
        return base_filename.with_suffix(f".{new_suffix}")
    return base_filename.with_name(f"{base_filename.stem}_{arch}.{new_suffix}")


def generate_file_path(tmp_dir_path, suffix, arch):
    base_filename = tmp_dir_path / tmp_dir_path.name
    filename = generate_arch_specific_filename(base_filename, suffix, arch)
    return filename


def get_mlisa(src, arch, args, device_kernel_name, mlisa_path):
    if args.is_linear == None:
        is_linear = True if arch in ["mtp_592"] else False
    else:
        is_linear = bool(args.is_linear)
    options = {
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "is_linear": is_linear,
        "kernel_name": device_kernel_name,
        "precision_mode": args.precision_mode,
        "restrict_ptr": args.restrict_ptr,
    }
    isa_version = convert_arch_to_isa_version(arch)
    target = get_target(isa_version)
    backend = get_mlu_backend(target)
    options = backend.parse_options(options)
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    kernel_mlisa = 'mlisa'
    with mlisa_path.open("wb") as fp:
        fp.write(ccinfo.asm[kernel_mlisa].encode())
    return mlisa_path


def get_cnbin(arch, mlisa_path, cnbin_path):
    cnas, _ = path_to_binary("cnas")
    cnas_cmd = [cnas, '-a', arch, '-i', mlisa_path, '-o', cnbin_path]
    subprocess.run(cnas_cmd, check=True, close_fds=False)
    return cnbin_path


def get_cnfatbin(src, args, device_kernel_name, out_path):
    cnbin_files = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for arch in args.archs:
            mlisa_path = generate_file_path(tmp_dir_path, 'mlisa', arch)
            mlisa_file = get_mlisa(src, arch, args, device_kernel_name, mlisa_path)
            cnbin_path = generate_file_path(tmp_dir_path, 'cnbin', arch)
            cnbin_file = get_cnbin(arch, mlisa_file, cnbin_path)
            cnbin_files.append(cnbin_file)

        cnlink, _ = path_to_binary("cnlink")
        fatbin_path = generate_arch_specific_filename(out_path, "cnfatbin")
        cnlink_cmd = [cnlink, '--fatbin'] + [arg for cnbin_file in cnbin_files
                                             for arg in ('-i', cnbin_file)] + ['-o', fatbin_path]
        subprocess.run(cnlink_cmd, check=True, close_fds=False)
    return fatbin_path


def compile_obj(fatbin_path, source_path, obj_path):
    """ Compile mlu obj with mlu source and fatbin. """
    neuware_home = os.getenv("NEUWARE_HOME", "/usr/local/neuware")
    os.environ["NEUWARE_HOME"] = neuware_home
    if not os.path.exists(neuware_home):
        raise RuntimeError(f"NEUWARE_HOME: '{neuware_home}' is not exists!")
    CNCC = os.path.join(neuware_home, "bin/cncc")
    COMPILE_CMD_TPL = f"{CNCC} {source_path.absolute()} \
            -fPIC -Werror -Wall -std=c++11 \
            --bang-host-only -Xclang -fbang-include-mlubinary -Xclang {fatbin_path.absolute()} \
            -c -o {obj_path.absolute()}"

    ret = os.system(COMPILE_CMD_TPL)
    if ret != 0:
        raise RuntimeError(f"Run cmd: \n {COMPILE_CMD_TPL} failed!")


def convert_arch_to_isa_version(arch: str):
    if arch == None:
        return None
    if arch.startswith(ARCH_PREFIX):
        return int(arch[len(ARCH_PREFIX):])
    else:
        raise RuntimeError(f"mlu arch: {arch} is not supported!")


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--device-kernel-name", "-cn", type=str, default="", help="Name of the device kernel name")
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", type=int, default=0, help="Number of stages to launch the kernel")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out path")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--type", "-t", type=str, default='obj', choices=['obj', 'fatbin'],
                        help="Compile output type of AOT")
    parser.add_argument(
        "--precision_mode", "-p", type=str, default='precision', choices=['precision', 'fast'],
        help="Compile mode type of AOT.(Select 'precision' for precision mode and 'fast' for performance mode.)")
    parser.add_argument(
        "--restrict_ptr", "-r", type=bool, default=False, choices=[True, False],
        help="Trigger genesis's more radical optimization analysis."
        "If there is no overlap in function parameters, it can be set to True without affecting accuracy.")
    parser.add_argument(
        "--archs", "-a", type=str, nargs='+', default=[get_arch()],
        help="Specify the AOT compilation target MLU architecture(s). "
        "Supported values: mtp_372, mtp_592. Use spaces to separate multiple architectures.\n"
        "Examples:\n"
        "  --archs mtp_592\n"
        "  --archs mtp_372 mtp_592\n"
        "  -a mtp_592\n")
    parser.add_argument(
        "--is-linear", "-l", type=int, default=None, choices=[0, 1], help=
        "Optional parameter, whether to enable linear memory. If not specified, it will automatically determine whether to enable it based on isa version."
    )
    args = parser.parse_args()

    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)
    device_kernel_name = args.device_kernel_name if args.device_kernel_name else args.kernel_name

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        # for const dtype
        if 'dtype' in s:
            # dtype[fp32] for dtype of fp32
            return tl.dtype(s[6:-1].strip(" "))
        return None

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    # set divisible_by_16 by default if arg is pointer
    hints.update({i: constexpr("16") for i, s in enumerate(signature) if s[0] == '*' and ':' not in s})
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {
        kernel.arg_names[i]: s.split(":")[0] if kernel.arg_names[i] not in constants else 'constexpr'
        for i, s in enumerate(signature)
    }
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cnbin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = triton.compiler.AttrsDescriptor.from_hints(hints)
    for p, v in attrs.get_constants().items():
        constants.update({kernel.arg_names[p]: v})

    src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
    signature = {key: value for key, value in signature.items() if value != 'constexpr'}
    fatbin_path = get_cnfatbin(src, args, device_kernel_name, out_path)

    arg_names = signature.keys()
    func_name = '_'.join([out_name])
    params = {
        "device_kernel_name": device_kernel_name,
        "func_name": [func_name, func_name.upper()],
        "arg_names": ", ".join([f"{arg}" for arg in arg_names]),
        "signature": ", ".join([f"{ty_to_cpp(ty, 'void*')} {name}" for name, ty in zip(arg_names, signature.values())]),
        "func_docstring": doc_string,
        "num_warps": args.num_warps,
        "tt_jit_name": args.kernel_name,
    }
    # generate header
    header_path = generate_arch_specific_filename(out_path, "h")
    template_path = Path(__file__).parent / "mlu_compile.h"
    with header_path.open("w") as fp:
        fp.write(Path(template_path).read_text().format(**params))

    # generate source
    source_path = generate_arch_specific_filename(out_path, "mlu")
    template_path = Path(__file__).parent / "mlu_compile.mlu"
    with source_path.open("w") as fp:
        fp.write(Path(template_path).read_text().format(**params))

    # generate obj
    if args.type == "obj":
        obj_path = generate_arch_specific_filename(out_path, "o")
        compile_obj(fatbin_path, source_path, obj_path)
        fatbin_path.unlink()
        source_path.unlink()
