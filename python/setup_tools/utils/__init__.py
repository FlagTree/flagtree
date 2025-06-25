from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
from . import ascend, aipu, cambricon, default, tsingmicro, xpu


@dataclass
class FlagTreeBackend:
    name: str
    url: str
    tag: str = None


flagtree_backends = (
    FlagTreeBackend(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                    tag="380b87122c88af131530903a702d5318ec59bb33"),
    FlagTreeBackend(name="cambricon", url="https://github.com/Cambricon/triton-linalg.git",
                    tag="00f51c2e48a943922f86f03d58e29f514def646d"),
    FlagTreeBackend(name="ascend", url="https://gitee.com/ascend/triton-ascend.git"),
)


def activate(backend, suffix=".py"):
    if not backend:
        return
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pass
    return module


__all__ = ["ascend", "aipu", "cambricon", "default", "tsingmicro", "xpu"]
