import os
import shutil


def insert_at_file_start(filepath, import_lines):
    import tempfile
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if import_lines in content:
            return False
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write(import_lines + '\n\n')
            with open(filepath, 'r') as original_file:
                tmp_file.write(original_file.read())
        backup_path = filepath + '.bak'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(filepath, backup_path)
        shutil.move(tmp_file.name, filepath)
        print(f"[INFO]: {filepath} is patched")
        return True
    except PermissionError:
        print(f"[ERROR]: No permission to write to {filepath}!")
    except FileNotFoundError:
        print(f"[ERROR]: {filepath} does not exist!")
    except Exception as e:
        print(f"[ERROR]: Unknown error: {str(e)}")
    return False


def append_at_file_end(filepath, import_lines):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if import_lines in content:
            return False
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write('\n' + import_lines)
        return True
    except PermissionError:
        print(f"[ERROR]: No permission to write to {filepath}!")
    except FileNotFoundError:
        print(f"[ERROR]: {filepath} does not exist!")
    except Exception as e:
        print(f"[ERROR]: Unknown error: {str(e)}")
    return False


def post_install():
    import site
    install_dir = site.getsitepackages()
    install_dir = os.path.join(install_dir, "triton")
    init_path = os.path.join(install_dir, "__init__.py")
    patched_content = f"""
import sys
from .triton_patch.language import _utils as ascend_utils
sys.modules['triton.language._utils'] = ascend_utils
from .triton_patch.compiler import compiler as ascend_compiler
sys.modules['triton.compiler.compiler'] = ascend_compiler
from .triton_patch.compiler import code_generator as ascend_code_generator
sys.modules['triton.compiler.code_generator'] = ascend_code_generator
from .triton_patch.compiler import errors as ascend_errors
sys.modules['triton.compiler.errors'] = ascend_errors
from .triton_patch.runtime import autotuner as ascend_autotuner
sys.modules['triton.runtime.autotuner'] = ascend_autotuner
from .triton_patch import testing as ascend_testing
sys.modules['triton.testing'] = ascend_testing
"""
    insert_at_file_start(init_path, patched_content)

    content_to_append = f"""
from .triton_patch.language.core import dot, gather, insert, subview
from .triton_patch.language.standard import flip
from .triton_patch.language.math import umulhi, exp, exp2, log, log2, cos, sin, sqrt, sqrt_rn, rsqrt, div_rn, erf, tanh, floor, ceil
from . import language

language.dot = dot
language.flip = flip
language.gather = gather
language.insert = insert
language.subview = subview

# from .triton_patch.language.core import dtype, pointer_type, block_type, function_type
# language.core.dtype = dtype
# language.core.pointer_type = pointer_type
# language.core.block_type = block_type
# language.core.function_type = function_type

from .triton_patch.language.semantic import arange, floordiv
language.semantic.arange = arange
language.semantic.floordiv = floordiv

language.umulhi = umulhi
language.exp = exp
language.exp2 = exp2
language.log = log
language.log2 = log2
language.cos = cos
language.sin = sin
language.sqrt = sqrt
language.sqrt_rn = sqrt_rn
language.rsqrt = rsqrt
language.div_rn = div_rn
language.erf = erf
language.tanh = tanh
language.floor = floor
language.ceil = ceil
language.math.umulhi = umulhi
language.math.exp = exp
language.math.exp2 = exp2
language.math.log = log
language.math.log2 = log2
language.math.cos = cos
language.math.sin = sin
language.math.sqrt = sqrt
language.math.sqrt_rn = sqrt_rn
language.math.rsqrt = rsqrt
language.math.div_rn = div_rn
language.math.erf = erf
language.math.tanh = tanh
language.math.floor = floor
language.math.ceil = ceil
"""
    append_at_file_end(init_path, content_to_append)


def get_ascend_patch_packages(backends):
    packages = []
    # packages += get_language_extra_packages()
    packages += [
        "triton/triton_patch",
        "triton/triton_patch/language",
        "triton/triton_patch/compiler",
        "triton/triton_patch/runtime",
    ]
    return packages


def get_ascend_patch_package_dir(backends):
    package_dir = {}
    # language_extra_list = get_language_extra_packages()
    # for extra_full in language_extra_list:
    #     extra_name = extra_full.replace("triton/language/extra/", "")
    #     package_dir[extra_full] = f"{triton_root_rel_dir}/language/extra/{extra_name}"
    #
    triton_patch_root_rel_dir = "triton_patch/python/triton_patch"
    package_dir["triton/triton_patch"] = f"{triton_patch_root_rel_dir}"
    package_dir["triton/triton_patch/language"] = f"{triton_patch_root_rel_dir}/language"
    package_dir["triton/triton_patch/compiler"] = f"{triton_patch_root_rel_dir}/compiler"
    package_dir["triton/triton_patch/runtime"] = f"{triton_patch_root_rel_dir}/runtime"
    return package_dir
