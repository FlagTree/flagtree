from pathlib import Path
import importlib.util
import os
from . import tools, default, xpu
from .tools import flagtree_submoduel_dir, download_module, OfflineBuildManager, is_skip_cuda_toolkits

flagtree_submoduels = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="380b87122c88af131530903a702d5318ec59bb33",
                 dst_path=os.path.join(flagtree_submoduel_dir, "triton_shared")),
}


def activate(backend, suffix=".py"):
    if not backend:
        backend = "default"
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pass
    return module


__all__ = [
    "default", "tsingmicro", "xpu", "tools", "flagtree_submoduels", "activate", "download_module",
    "OfflineBuildManager", "is_skip_cuda_toolkits"
]
