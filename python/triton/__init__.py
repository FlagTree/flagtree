"""isort:skip_file"""
__version__ = '3.1.0'

# ---------------------------------------
# Note: import order is significant here.

# submodules
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
    reinterpret,
    TensorWrapper,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from .runtime.jit import jit
from .compiler import compile, CompilationError
from .errors import TritonError

from . import language
from . import testing
from . import tools
try:
    import torch
    HAS_TORCH = True
    HAS_PADDLE = False
    print("\033[91m🔥 Using Torch\033[0m") 
except Exception:
    import paddle
    HAS_TORCH = False
    HAS_PADDLE = True
    print("\033[94m🌊 Using Paddle\033[0m")

__all__ = [
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "heuristics",
    "impl",
    "InterpreterError",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "next_power_of_2",
    "ops",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "TensorWrapper",
    "TritonError",
    "testing",
    "tools",
]

# -------------------------------------
# misc. utilities that  don't fit well
# into any specific module
# -------------------------------------


def cdiv(x: int, y: int):
    return (x + y - 1) // y


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
