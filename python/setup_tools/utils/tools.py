import os
import shutil
from pathlib import Path
from dataclasses import dataclass

flagtree_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
flagtree_submoduel_dir = os.path.join(flagtree_root_dir, "third_party")
flagtree_backend = os.environ.get("FLAGTREE_BACKEND")
use_cuda_toolkit = ["aipu"]

network_configs = {
    "MAX_RETRY_COUNT": 4,
    "GIT_CLONE_TIMEOUT": 60,
}


@dataclass
class Module:
    name: str
    url: str
    commit_id: str = None
    dst_path: str = None


def is_skip_cuda_toolkits():
    return flagtree_backend and (flagtree_backend not in use_cuda_toolkit)


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def remove_triton_in_modules(model):
    model_path = model.dst_path
    triton_path = os.path.join(model_path, "triton")
    if os.path.exists(triton_path):
        shutil.rmtree(triton_path)


def py_clone(module):
    import git
    retry_count = network_configs["MAX_RETRY_COUNT"]
    has_specialization_commit = module.commit_id is not None
    while (retry_count):
        try:
            repo = git.Repo.clone_from(module.url, module.dst_path)
            if has_specialization_commit:
                repo.git.checkout(module.commit_id)
            return True
        except Exception:
            retry_count -= 1
            print(
                f"\n[{network_configs['MAX_RETRY_COUNT'] - retry_count}] retry to clone {module.name} to  {module.dst_path}"
            )
    return False


def sys_clone(module):
    retry_count = network_configs["MAX_RETRY_COUNT"]
    has_specialization_commit = module.commit_id is not None
    while (retry_count):
        try:
            os.system(f"git clone {module.url} {module.dst_path}")
            if has_specialization_commit:
                os.system("cd module.dst_path")
                os.system(f"git checkout {module.commit_id}")
                os.system("cd -")
            return True
        except Exception:
            retry_count -= 1
            print(
                f"\n[{network_configs['MAX_RETRY_COUNT'] - retry_count}] retry to clone {module.name} to  {module.dst_path}"
            )
    return False


def clone_module(module):
    succ = True if py_clone(module) else sys_clone(module)
    if not succ:
        print(f"[ERROR]: Failed to clone {module.name} from {module.url}")
        return False
    print(f"[INFO]: Successfully cloned {module.name} to {module.dst_path}")
    return True


def download_module(module, required=False):
    if module is None:
        return
    if not os.path.exists(module.dst_path):
        # Try to copy from offline build dir if offline build is enabled.
        module_offline_handler = OfflineBuildManager()
        if module_offline_handler.is_offline:
            src_path = os.path.join(module_offline_handler.offline_build_dir, module.name)
            succ = os.path.exists(src_path)
            if succ:
                print(f"[INFO] Offline Build: Found {module.name} at {src_path}")
                module_offline_handler.src = os.path.join(module_offline_handler.offline_build_dir, module.name)
                module_offline_handler.copy_to_flagtree_project(dst_path=module.dst_path)
            else:
                print(f"[INFO] Offline Build: {module.name} is not found in offline build directory.")
        else:
            # No offline build, clone from online source
            succ = clone_module(module)
    else:
        print(f'Found third_party {module.name} at {module.dst_path}\n')
        return True
    if not succ and required:
        raise RuntimeError(
            f"[ERROR]: Failed to download {module.name} from {module.url}, It's most likely the network!")
    remove_triton_in_modules(module)


def get_triton_cache_path():
    user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


class OfflineBuildManager:

    def __init__(self):
        self.is_offline = self.is_offline_build()
        self.offline_build_dir = os.environ.get("FLAGTREE_OFFLINE_BUILD_DIR") if self.is_offline else None
        self.triton_cache_path = get_triton_cache_path()

    def is_offline_build(self) -> bool:
        return os.getenv("TRITON_OFFLINE_BUILD", "OFF") == "ON" or os.getenv("FLAGTREE_OFFLINE_BUILD_DIR")

    def copy_to_flagtree_project(self, kargs):
        dst_path = os.path.join(flagtree_root_dir,
                                kargs['dst_path']) if 'dst_path' in kargs and kargs['dst_path'] else None
        src_path = self.src
        if not dst_path:
            return False
        src_path = self.src
        print(f"[INFO] Copying from {src_path} to {dst_path}")
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dst_path)

    def handle_flagtree_hock(self, kargs):
        if 'post_hock' in kargs and kargs['post_hock']:
            kargs['post_hock'](self.src)

    def handle_triton_origin_toolkits(self):
        triton_origin_toolkits = ["nvidia/ptxas", "nvidia/nvdisasm", "nvidia/cuobjdump", "nvidia/cudacrt", "nvidia/cudart", "nvidia/cupti", "pybind11", "json"]
        for toolkit in triton_origin_toolkits:
            toolkit_cache_path = os.path.join(self.triton_cache_path, toolkit)
            if os.path.exists(toolkit_cache_path):
                continue
            src_path = os.path.join(self.offline_build_dir, toolkit)
            if os.path.exists(src_path):
                print(f"[INFO] Copying {toolkit} from {src_path} to {toolkit_cache_path}")
                shutil.copytree(src_path, toolkit_cache_path, dirs_exist_ok=True)
            else:
                raise RuntimeError(
                    f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_backend} offline build dependency \033[93m{src_path}\033[0m does not exist.\n"
                )

    def validate_offline_build_dir(self, path, required=False):
        if (not path or not os.path.exists(path)) and required:
            raise RuntimeError(
                "\n\n\033[31m[ERROR]:\033[0m If you want to use the offline build method\n"
                "please set FLAGTREE_OFFLINE_BUILD_DIR as the path of the offline dependency package\n"
                "or please \033[31mdo not use\033[0m the environment variable \033[93mTRITON_OFFLINE_BUILD !\033[0m \n\n"
            )

    def validate_offline_build_deps(self, path, kargs, required=False):
        url = kargs.get('url', None)
        if (not path or not os.path.exists(path)) and required:
            raise RuntimeError(
                f"\n\n \033[31m[ERROR]:\033[0m The {flagtree_backend} offline build dependency \033[93m{path}\033[0m does not exist.\n"
                f" And you can download the dependency package from the  \n \033[93m{url}\033[0m \n"
                f" then extract it to the \033[93m{self.offline_build_dir}\033[0m directory you specified !\033[0m\n\n")

    def validate_offline_build(self, path, required=False, is_base_dir=False, kargs=None):
        if is_base_dir:
            self.validate_offline_build_dir(path, required)
        else:
            self.validate_offline_build_deps(path, kargs, required)

    def single_build(self, *args, **kargs):
        if not self.is_offline:
            return False
        required = kargs['required'] if 'required' in kargs else False
        self.validate_offline_build(self.offline_build_dir, required, is_base_dir=True)
        self.src = os.path.join(self.offline_build_dir, kargs['src']) if 'src' in kargs else None
        self.validate_offline_build(self.src, required, kargs=kargs)
        print(f"[INFO] Building in offline mode using directory: {self.src}")
        self.copy_to_flagtree_project(kargs)
        self.handle_flagtree_hock(kargs)
        if is_skip_cuda_toolkits():
            print(f"[INFO] Skipping CUDA toolkits for {flagtree_backend} backend in offline build.")
        else:
            self.handle_triton_origin_toolkits()
        return True
