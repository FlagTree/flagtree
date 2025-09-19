import os
import shutil
from pathlib import Path
import tarfile
import zipfile
from io import BytesIO
import urllib.request
from dataclasses import dataclass

flagtree_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
flagtree_submodule_dir = os.path.join(flagtree_root_dir, "third_party")


@dataclass
class NetConfig:
    max_retry: int = 4
    timeout: int = 300
    user_agent: str = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers: dict = None


@dataclass
class Module:
    name: str
    url: str
    commit_id: str = None
    dst_path: str = None


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


def decompress(url, content, dst_path, file_name=None):
    file_bytes = BytesIO(content)
    file_names = []
    if url.endswith(".zip"):
        with zipfile.ZipFile(file_bytes, "r") as file:
            file.extractall(path=dst_path)
            file_names = file.namelist()
    else:
        with tarfile.open(fileobj=file_bytes, mode="r|*") as file:
            file.extractall(path=dst_path)
            file_names = file.getnames()
    os.rename(Path(dst_path) / file_names[0], Path(dst_path) / file_name)


class DownloadManager:

    def __init__(self):
        self.src_list = {}
        self.current_url = None
        self.current_dst_path = None
        self.current_file_name = None
        NetConfig.headers = {'User-Agent': NetConfig.user_agent}

    def download(self, url=None, path=None, file_name=None, mode=None, module=None, required=False):
        self.init_single_src_settings(url, path, file_name, mode)
        if mode == "git" or module:
            return self.git_clone(module, required)
        else:
            return self.general_download(is_decompress=True)

    def init_single_src_settings(self, url, path, file_name, mode):
        self.current_url = url
        self.current_dst_path = path
        self.current_file_name = file_name
        self.src_list[self.current_url] = {"mode": mode, "path": path, "status": None, "content": None}

    def set_status(self, status, content):
        self.src_list[self.current_url]['status'] = status
        self.src_list[self.current_url]['content'] = content

    def git_clone(self, module, required=False):
        if module is None:
            return
        if not os.path.exists(module.dst_path):
            succ = self.clone_module(module)
        else:
            print(f'Found third_party {module.name} at {module.dst_path}\n')
            return True
        if not succ and required:
            raise RuntimeError(
                f"[ERROR]: Failed to download {module.name} from {module.url}, It's most likely the network!")
        remove_triton_in_modules(module)

    def py_clone(self, module):
        try:
            import git
        except ImportError:
            return False
        retry_count = NetConfig.max_retry
        has_specialization_commit = module.commit_id is not None
        while (retry_count):
            try:
                repo = git.Repo.clone_from(module.url, module.dst_path)
                if has_specialization_commit:
                    repo.git.checkout(module.commit_id)
                return True
            except Exception:
                retry_count -= 1
                print(f"\n[{NetConfig.max_retry - retry_count}] retry to clone {module.name} to  {module.dst_path}")
        return False

    def sys_clone(self, module):
        retry_count = NetConfig.max_retry
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
                print(f"\n[{NetConfig.max_retry - retry_count}] retry to clone {module.name} to  {module.dst_path}")
        return False

    def clone_module(self, module):
        succ = True if self.py_clone(module) else self.sys_clone(module)
        if not succ:
            print(f"[ERROR]: Failed to clone {module.name} from {module.url}")
            return False
        print(f"[INFO]: Successfully cloned {module.name} to {module.dst_path}")

    def general_download_impl(self, request):
        with urllib.request.urlopen(request, timeout=NetConfig.timeout) as response:
            content = response.read()
            return content

    def general_download(self, is_decompress=True):
        request = urllib.request.Request(self.current_url, None, NetConfig.headers)
        current_retry_count = NetConfig.max_retry
        content = None
        print(f'downloading {self.current_url} ...')
        while (current_retry_count):
            try:
                content = self.general_download_impl(request)
                break
            except Exception:
                current_retry_count -= 1
                residue = NetConfig.max_retry - current_retry_count
                print(f"\n [Note]: [{residue}] retry to downloading and extracting {self.current_url}")
        if current_retry_count == 0:
            self.set_status(status='fail', content=None)
            raise RuntimeError("The download failed, probably due to network problems!")

        self.set_status(status='succ', content=content)

        if is_decompress:
            decompress(self.current_url, content=content, dst_path=self.current_dst_path,
                       file_name=self.current_file_name)
