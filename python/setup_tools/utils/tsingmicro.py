import os
import shutil


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("tsingmicro-opt")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return [
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + src_ext_path,
    ]


def install_extension(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("tsingmicro-opt")
    src_ext_path = os.path.join(os.path.abspath(os.path.dirname(src_ext_path)), "tsingmicro-opt")
    python_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dst_ext_path = os.path.join(python_root_dir, "triton/backends/tsingmicro/bin/tsingmicro-opt")
    bin_dir = os.path.dirname(dst_ext_path)
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    shutil.copy(src_ext_path, dst_ext_path)
