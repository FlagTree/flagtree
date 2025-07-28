from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
from . import ascend


@dataclass
class FlagTreeBackend:
    name: str
    url: str
    tag: str = None


flagtree_backends = (
    FlagTreeBackend(name="triton_shared", url="https://github.com/microsoft/triton-shared.git",
                    tag="380b87122c88af131530903a702d5318ec59bb33"),
    FlagTreeBackend(name="ascend", url="https://gitee.com/ascend/triton-ascend.git"),
)


def activate(backend, suffix=".py"):
    if not backend:
        return
    module_path = Path(os.path.dirname(__file__)) / backend
    module_path = str(module_path) + suffix
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["ascend"]
