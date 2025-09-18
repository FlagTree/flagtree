from pathlib import Path
import importlib.util
import os
from . import tools, ascend, cambricon, xpu
from .tools import download_module, flagtree_submoduel_dir

flagtree_submoduels = {
    "triton_shared":
    tools.Module(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                 commit_id="380b87122c88af131530903a702d5318ec59bb33",
                 dst_path=os.path.join(flagtree_submoduel_dir, "triton_shared")),
    "ascend":
    tools.Module(name="ascend", url="https://gitee.com/flagtree/triton-ascend.git",
                 commit_id="flagtree-dot-hint",
                 dst_path=os.path.join(flagtree_submoduel_dir, "triton_ascend")),
}


def activate(backend, suffix=".py"):
    if not backend:
        backend = "default"
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["download_module", "tools", "ascend", "cambricon", "xpu"]
