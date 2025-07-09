from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
from . import ascend
from . import aipu
from . import default


@dataclass
class FlagTreeBackend:
    name: str
    url: str
    tag: str = None


flagtree_backends = (
    FlagTreeBackend(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                    tag="5842469a16b261e45a2c67fbfc308057622b03ee"),
    FlagTreeBackend(name="cambricon", url="https://github.com/Cambricon/triton-linalg.git",
                    tag="00f51c2e48a943922f86f03d58e29f514def646d"),
    FlagTreeBackend(name="flir", url="https://github.com/FlagTree/flir.git"),
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


__all__ = ["ascend", "aipu", "default", "activate"]
