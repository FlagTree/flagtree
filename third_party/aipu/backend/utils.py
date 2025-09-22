# -*- coding: utf-8 -*-
import functools
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
import pybind11


def _get_npucompiler_path() -> str:
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH."
            )
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return npu_compiler_path


def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path


def _check_bishengir_api_change() -> bool:
    bishengir_path = _get_npucompiler_path()
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and 'limit-auto-multi-buffer-buffer' in result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def _check_bishengir_is_regbased() -> bool:
    bishengir_path = _get_npucompiler_path()
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and 'reg-based' in result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


@functools.lru_cache(None)
def _get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return Path(path)


def _is_ascend_sanitizer_enabled() -> bool:
    return os.getenv("TRITON_ENABLE_SANITIZER", "false").lower() in ("true", "1")


def _is_debug_line_info_disabled() -> bool:
    return os.getenv("TRITON_DISABLE_LINE_INFO", "true").lower() in ("true", "1")


def _is_auto_map_parallel_blocks_enabled() -> bool:
    if not _enable_unpublished_feature():
        return False
    return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")


def _enable_unpublished_feature() -> bool:
    return os.getenv("ENABLE_UNPUBLISHED_FEATURE", "false").lower() in ("true", "1")


def _build_npu_ext(obj_name: str, src_path, src_dir, *, kernel_launcher=None) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = clangxx if clangxx is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_path]
    cc_cmd += [f"-w"]

    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]

    asc_path = _get_ascend_path()
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]

    if kernel_launcher == "torch":
        import torch
        import torch_npu
        torch_path = os.path.dirname(os.path.realpath(torch.__file__))
        torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
        use_cxx11_abi = _check_cxx11_abi()
        cc_cmd += [
            f"-I{os.path.join(torch_path, 'include')}",
            f"-I{os.path.join(torch_npu_path, 'include')}",
            f"-L{os.path.join(torch_npu_path, 'lib')}",
            "-ltorch_npu",
            f"-D_GLIBCXX_USE_CXX11_ABI={use_cxx11_abi}",
        ]

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_path)


def _check_cxx11_abi():
    import torch
    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        "i1": 12,
        "i8": 2,
        "i16": 6,
        "i32": 3,
        "i64": 9,
        "u32": 8,
        "u64": 10,
        "fp16": 1,
        "bf16": 27,
        "fp32": 0,
        "fp64": 11,
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")
    return MAP_SIGTYPE_TO_INT[sigty]
