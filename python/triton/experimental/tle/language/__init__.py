# flagtree tle
from importlib import import_module

from .core import (
    load,
    make_partition_view,
    make_strided_view,
    make_gather_scatter_view
)

from . import gpu, raw

__all__ = [
    "load",
    "dsa",
    "make_partition_view",
    "make_strided_view",
    "make_gather_scatter_view",
]


def __getattr__(name):
    if name == "dsa":
        module = import_module(f"{__name__}.dsa")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
