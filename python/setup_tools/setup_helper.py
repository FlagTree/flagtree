import os
import shutil
import sys
import functools
import tarfile
import zipfile
from io import BytesIO
import urllib.request
from pathlib import Path
import hashlib
from distutils.sysconfig import get_python_lib
from . import utils

extend_backends = []
default_backends = ["nvidia", "amd"]
plugin_backends = ["cambricon", "ascend", "aipu", "tsingmicro"]
ext_sourcedir = "triton/_C/"
flagtree_backend = os.getenv("FLAGTREE_BACKEND", "").lower()
flagtree_plugin = os.getenv("FLAGTREE_PLUGIN", "").lower()
offline_build = os.getenv("FLAGTREE_PLUGIN", "OFF")
device_mapping = {"xpu": "xpu", "mthreads": "musa", "ascend": "ascend"}
language_extra_backends = ['xpu', 'musa']
flagtree_backends = utils.flagtree_backends
backend_utils = utils.activate(flagtree_backend)

set_llvm_env = lambda path: set_env({
    'LLVM_INCLUDE_DIRS': Path(path) / "include",
    'LLVM_LIBRARY_DIR': Path(path) / "lib",
    'LLVM_SYSPATH': path,
})


def install_extension(*args, **kargs):
    try:
        backend_utils.install_extension(*args, **kargs)
    except Exception:
        pass


def get_backend_cmake_args(*args, **kargs):
    try:
        return backend_utils.get_backend_cmake_args(*args, **kargs)
    except Exception:
        return []


def get_device_name():
    return device_mapping[flagtree_backend]


def get_extra_packages():
    packages = []
    try:
        packages = backend_utils.get_extra_install_packages()
    except Exception:
        packages = []
    return packages


def get_language_extra():
    packages = []
    if flagtree_backend in language_extra_backends:
        device_name = device_mapping[flagtree_backend]
        extra_path = f"triton/language/extra/{device_name}"
        packages.append(extra_path)
    return packages


def get_package_data_tools():
    package_data = ["compile.h", "compile.c"]
    try:
        package_data += backend_utils.get_package_data_tools()
    except Exception:
        package_data
    return package_data


def git_clone(lib, lib_path):
    import git
    MAX_RETRY = 4
    print(f"Clone {lib.name} into {lib_path} ...")
    retry_count = MAX_RETRY
    while (retry_count):
        try:
            repo = git.Repo.clone_from(lib.url, lib_path)
            if lib.tag is not None:
                repo.git.checkout(lib.tag)
            sub_triton_path = Path(lib_path) / "triton"
            if os.path.exists(sub_triton_path):
                shutil.rmtree(sub_triton_path)
            print(f"successfully clone {lib.name} into {lib_path} ...")
            return True
        except Exception:
            retry_count -= 1
            print(f"\n[{MAX_RETRY - retry_count}] retry to clone {lib.name} to  {lib_path}")
    return False


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def download_flagtree_third_party(name, condition, required=False, hock=None):
    if not condition:
        return
    backend = None
    for _backend in flagtree_backends:
        if _backend.name in name:
            backend = _backend
            break
    if backend is None:
        return backend
    base_dir = dir_rollback(3, __file__) / "third_party"
    prelib_path = Path(base_dir) / name
    lib_path = Path(base_dir) / _backend.name

    if not os.path.exists(prelib_path) and not os.path.exists(lib_path):
        succ = git_clone(lib=backend, lib_path=prelib_path)
        if not succ and required:
            raise RuntimeError("Bad network ! ")
    else:
        print(f'Found third_party {backend.name} at {lib_path}\n')
    if callable(hock):
        hock(third_party_base_dir=base_dir, backend=backend, default_backends=default_backends)


def post_install():
    try:
        backend_utils.post_install()
    except Exception:
        pass


class FlagTreeCache:

    def __init__(self):
        self.flagtree_dir = os.path.dirname(os.getcwd())
        self.dir_name = ".flagtree"
        self.sub_dirs = {}
        self.cache_files = {}
        self.dir_path = self._get_cache_dir_path()
        self._create_cache_dir()
        if flagtree_backend:
            self._create_subdir(subdir_name=flagtree_backend)

    @functools.lru_cache(maxsize=None)
    def _get_cache_dir_path(self) -> Path:
        _cache_dir = os.environ.get("FLAGTREE_CACHE_DIR")
        if _cache_dir is None:
            _cache_dir = Path.home() / self.dir_name
        else:
            _cache_dir = Path(_cache_dir)
        return _cache_dir

    def _create_cache_dir(self) -> Path:
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)

    def _create_subdir(self, subdir_name, path=None):
        if path is None:
            subdir_path = Path(self.dir_path) / subdir_name
        else:
            subdir_path = Path(path) / subdir_name

        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        self.sub_dirs[subdir_name] = subdir_path

    def _md5(self, file_path):
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as file:
            while chunk := file.read(4096):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _download(self, url, path, file_name):
        MAX_RETRY_COUNT = 4
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
        headers = {
            'User-Agent': user_agent,
        }
        request = urllib.request.Request(url, None, headers)
        retry_count = MAX_RETRY_COUNT
        content = None
        print(f'downloading {url} ...')
        while (retry_count):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    content = response.read()
                    break
            except Exception:
                retry_count -= 1
                print(f"\n[{MAX_RETRY_COUNT - retry_count}] retry to downloading and extracting {url}")

        if retry_count == 0:
            raise RuntimeError("The download failed, probably due to network problems")

        print(f'extracting {url} ...')
        file_bytes = BytesIO(content)
        file_names = []
        if url.endswith(".zip"):
            with zipfile.ZipFile(file_bytes, "r") as file:
                file.extractall(path=path)
                file_names = file.namelist()
        else:
            with tarfile.open(fileobj=file_bytes, mode="r|*") as file:
                file.extractall(path=path)
                file_names = file.getnames()
        os.rename(Path(path) / file_names[0], Path(path) / file_name)

    def check_file(self, file_name=None, url=None, path=None, md5_digest=None):
        origin_file_path = None
        if url is not None:
            origin_file_name = url.split("/")[-1].split('.')[0]
            origin_file_path = self.cache_files.get(origin_file_name, "")
        if path is not None:
            _path = path
        else:
            _path = self.cache_files.get(file_name, "")
        empty = (not os.path.exists(_path)) or (origin_file_path and not os.path.exists(origin_file_path))
        if empty:
            return False
        if md5_digest is None:
            return True
        else:
            cur_md5 = self._md5(_path)
            return cur_md5[:8] == md5_digest

    def clear(self):
        shutil.rmtree(self.dir_path)

    def reverse_copy(self, src_path, cache_file_path, md5_digest):
        if src_path is None or not os.path.exists(src_path):
            return False
        if os.path.exists(cache_file_path):
            return False
        copy_needed = True
        if md5_digest is None or self._md5(src_path) == md5_digest:
            copy_needed = False
        if copy_needed:
            print(f"copying {src_path} to {cache_file_path}")
            if os.path.isdir(src_path):
                shutil.copytree(src_path, cache_file_path, dirs_exist_ok=True)
            else:
                shutil.copy(src_path, cache_file_path)
            return True
        return False

    def store(self, file=None, condition=None, url=None, copy_src_path=None, copy_dst_path=None, files=None,
              md5_digest=None, pre_hock=None, post_hock=None):

        if not condition or (pre_hock and pre_hock()):
            return
        is_url = False if url is None else True
        path = self.sub_dirs[flagtree_backend] if flagtree_backend else self.dir_path

        if files is not None:
            for single_files in files:
                self.cache_files[single_files] = Path(path) / single_files
        else:
            self.cache_files[file] = Path(path) / file
            if url is not None:
                origin_file_name = url.split("/")[-1].split('.')[0]
                self.cache_files[origin_file_name] = Path(path) / file
            if copy_dst_path is not None:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                dst_path = Path(dst_path_root) / file
                if self.reverse_copy(dst_path, self.cache_files[file], md5_digest):
                    return

        if is_url and not self.check_file(file_name=file, url=url, md5_digest=md5_digest):
            self._download(url, path, file_name=file)

        if copy_dst_path is not None:
            file_lists = [file] if files is None else list(files)
            for single_file in file_lists:
                dst_path_root = Path(self.flagtree_dir) / copy_dst_path
                os.makedirs(dst_path_root, exist_ok=True)
                dst_path = Path(dst_path_root) / single_file
                if not self.check_file(path=dst_path, md5_digest=md5_digest):
                    if copy_src_path:
                        src_path = Path(copy_src_path) / single_file
                    else:
                        src_path = self.cache_files[single_file]
                    print(f"copying {src_path} to {dst_path}")
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy(src_path, dst_path)
        post_hock(self.cache_files[file]) if post_hock else False

    def get(self, file_name) -> Path:
        return self.cache_files[file_name]


class CommonUtils:

    @staticmethod
    def unlink():
        cur_path = dir_rollback(2, __file__)
        if "editable_wheel" in sys.argv:
            installation_dir = cur_path
        else:
            installation_dir = get_python_lib()
        backends_dir_path = Path(installation_dir) / "triton" / "backends"
        # raise RuntimeError(backends_dir_path)
        if not os.path.exists(backends_dir_path):
            return
        for name in os.listdir(backends_dir_path):
            exist_backend_path = os.path.join(backends_dir_path, name)
            if not os.path.isdir(exist_backend_path):
                continue
            if name.startswith('__'):
                continue
            if os.path.islink(exist_backend_path):
                os.unlink(exist_backend_path)
            if os.path.exists(exist_backend_path):
                shutil.rmtree(exist_backend_path)

    @staticmethod
    def skip_package_dir(package):
        if 'backends' in package or 'profiler' in package:
            return True
        try:
            return backend_utils.skip_package_dir(package)
        except Exception:
            return False

    @staticmethod
    def get_package_dir(packages):
        package_dict = {}
        if flagtree_backend and flagtree_backend not in plugin_backends:
            connection = []
            backend_triton_path = f"../third_party/{flagtree_backend}/python/"
            for package in packages:
                if CommonUtils.skip_package_dir(package):
                    continue
                pair = (package, f"{backend_triton_path}{package}")
                connection.append(pair)
            package_dict.update(connection)
        try:
            package_dict.update(backend_utils.get_package_dir())
        except Exception:
            pass
        return package_dict


def handle_flagtree_backend():
    global ext_sourcedir
    if flagtree_backend:
        print(f"\033[1;32m[INFO] FlagtreeBackend is {flagtree_backend}\033[0m")
        extend_backends.append(flagtree_backend)
        if "editable_wheel" in sys.argv and flagtree_backend != "ascend":
            ext_sourcedir = os.path.abspath(f"../third_party/{flagtree_backend}/python/{ext_sourcedir}") + "/"


def set_env(env_dict: dict):
    for env_k, env_v in env_dict.items():
        os.environ[env_k] = str(env_v)


def check_env(env_val):
    return os.environ.get(env_val, '') != ''


download_flagtree_third_party("triton_shared", hock=utils.default.precompile_hock, condition=(not flagtree_backend))

download_flagtree_third_party("triton_ascend", condition=(flagtree_backend == "ascend"),
                              hock=utils.ascend.precompile_hock, required=True)

download_flagtree_third_party("cambricon", condition=(flagtree_backend == "cambricon"), required=True)

download_flagtree_third_party("flir", condition=(flagtree_backend == "aipu"), hock=utils.aipu.precompile_hock,
                              required=True)

handle_flagtree_backend()

cache = FlagTreeCache()

# iluvatar
cache.store(
    file="iluvatarTritonPlugin.so", condition=("iluvatar" == flagtree_backend) and (flagtree_plugin == ''), url=
    "https://github.com/FlagTree/flagtree/releases/download/v0.3.0-build-deps/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64.tar.gz",
    copy_dst_path="third_party/iluvatar", md5_digest="015b9af8")

cache.store(
    file="iluvatar-llvm18-x86_64",
    condition=("iluvatar" == flagtree_backend),
    url="https://github.com/FlagTree/flagtree/releases/download/v0.3.0-build-deps/iluvatar-llvm18-x86_64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# klx xpu
cache.store(
    file="XTDK-llvm19-ubuntu2004_x86_64",
    condition=("xpu" == flagtree_backend),
    url="https://github.com/FlagTree/flagtree/releases/download/v0.3.0-build-deps/XTDK-llvm19-ubuntu2004_x86_64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

cache.store(file="xre-Linux-x86_64", condition=("xpu" == flagtree_backend),
            url="https://github.com/FlagTree/flagtree/releases/download/v0.3.0-build-deps/xre-Linux-x86_64.tar.gz",
            copy_dst_path='python/_deps/xre3')

cache.store(
    files=("clang", "xpu-xxd", "xpu3-crt.xpu", "xpu-kernel.t", "ld.lld", "llvm-readelf", "llvm-objdump",
           "llvm-objcopy"), condition=("xpu" == flagtree_backend),
    copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/bin", copy_dst_path="third_party/xpu/backend/xpu3/bin")

cache.store(files=("libclang_rt.builtins-xpu3.a", "libclang_rt.builtins-xpu3s.a"),
            condition=("xpu" == flagtree_backend), copy_src_path=f"{os.environ.get('LLVM_SYSPATH','')}/lib/linux",
            copy_dst_path="third_party/xpu/backend/xpu3/lib/linux")

cache.store(files=("include", "so"), condition=("xpu" == flagtree_backend),
            copy_src_path=f"{cache.dir_path}/xpu/xre-Linux-x86_64", copy_dst_path="third_party/xpu/backend/xpu3")

# mthreads
cache.store(
    file="mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64",
    condition=("mthreads" == flagtree_backend),
    url=
    "https://github.com/FlagTree/flagtree/releases/download/v0.1.0-build-deps/mthreads-llvm19-glibc2.34-glibcxx3.4.30-x64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# ascend
cache.store(
    file="llvm-b5cc222d-ubuntu-arm64",
    condition=("ascend" == flagtree_backend),
    url="https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b5cc222d-ubuntu-arm64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# arm aipu
cache.store(
    file="llvm-a66376b0-ubuntu-x64",
    condition=("aipu" == flagtree_backend),
    url="https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)

# tsingmicro
cache.store(
    file="tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-x64",
    condition=("tsingmicro" == flagtree_backend),
    url=
    "https://github.com/FlagTree/flagtree/releases/download/v0.2.0-build-deps/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-x64.tar.gz",
    pre_hock=lambda: check_env('LLVM_SYSPATH'),
    post_hock=set_llvm_env,
)
