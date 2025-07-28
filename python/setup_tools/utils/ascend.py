import os
import shutil
from pathlib import Path


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton-adapter-opt")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return [
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + src_ext_path,
    ]


def install_extension(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton-adapter-opt")
    src_ext_path = os.path.join(os.path.abspath(os.path.dirname(src_ext_path)), "triton-adapter-opt")
    python_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dst_ext_path = os.path.join(python_root_dir, "triton/backends/ascend/triton-adapter-opt")
    shutil.copy(src_ext_path, dst_ext_path)


def get_package_dir():
    package_dict = {}
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    triton_patch_prefix_dir = os.path.join(root_dir, "third_party/ascend/triton_patch/python/triton_patch")
    package_dict["triton/triton_patch"] = f"{triton_patch_prefix_dir}"
    package_dict["triton/triton_patch/language"] = f"{triton_patch_prefix_dir}/language"
    package_dict["triton/triton_patch/compiler"] = f"{triton_patch_prefix_dir}/compiler"
    package_dict["triton/triton_patch/runtime"] = f"{triton_patch_prefix_dir}/runtime"
    patch_paths = {
        "language/_utils.py",
        "compiler/compiler.py",
        "compiler/code_generator.py",
        "compiler/errors.py",
        "runtime/autotuner.py",
        "runtime/autotiling_tuner.py",
        "runtime/jit.py",
        "runtime/tile_generator.py",
        "runtime/utils.py",
        "runtime/libentry.py",
        "runtime/code_cache.py",
        "testing.py",
    }

    for path in patch_paths:
        package_dict[f"triton/{path}"] = f"{triton_patch_prefix_dir}/{path}"
    return package_dict


def get_extra_install_packages():
    return [
        "triton/triton_patch",
        "triton/triton_patch/language",
        "triton/triton_patch/compiler",
        "triton/triton_patch/runtime",
    ]


def precompile_hock(*args, **kargs):
    third_party_base_dir = Path(kargs['third_party_base_dir'])
    ascend_path = Path(third_party_base_dir) / "ascend"
    patch_path = Path(ascend_path) / "triton_patch"
    project_path = Path(third_party_base_dir) / "triton_ascend"
    if os.path.exists(ascend_path):
        shutil.rmtree(ascend_path)
    if not os.path.exists(project_path):
        raise RuntimeError(f"{project_path} can't be found. It might be due to a network issue")
    ascend_src_path = Path(project_path) / "ascend"
    patch_src_path = Path(project_path) / "triton_patch"
    shutil.copytree(ascend_src_path, ascend_path, dirs_exist_ok=True)
    shutil.copytree(patch_src_path, patch_path, dirs_exist_ok=True)
    shutil.rmtree(project_path)
    patched_code = """  set(triton_abs_dir "${TRITON_ROOT_DIR}/include/triton/Dialect/Triton/IR") """
    src_code = """set(triton_abs_dir"""

    filepath = Path(patch_path) / "include" / "triton" / "Dialect" / "Triton" / "IR" / "CMakeLists.txt"
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp_file:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if src_code in line:
                        tmp_file.writelines(patched_code)
                    else:
                        tmp_file.writelines(line)
        backup_path = str(filepath) + '.bak'
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


print(get_package_dir())
