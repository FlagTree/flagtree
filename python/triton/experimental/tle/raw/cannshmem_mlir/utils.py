# Copyright 2026-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Device-bitcode resolve for library=\"cannshmem_mlir\".

Late-links AscendNPU-IR Shmem template BC (``_mlir_ciface_aclshmem_*``).
Host helpers live in ``triton.experimental.tle.raw.cannshmem`` (re-exported below).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Re-export shared host helpers for tutorials that imported from cannshmem_mlir.
from triton.experimental.tle.raw.cannshmem.utils import (  # noqa: F401
    bench_us,
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

# library="cannshmem_mlir": Ascend backend late-links libdevice_cannshmem_mlir*.bc
_cannshmem_mlir_device_bc_enabled: bool = False


def enable_cannshmem_mlir_device_bc(enabled: bool = True) -> None:
    global _cannshmem_mlir_device_bc_enabled
    _cannshmem_mlir_device_bc_enabled = bool(enabled)


def is_cannshmem_mlir_device_bc_enabled() -> bool:
    return _cannshmem_mlir_device_bc_enabled


# Back-compat aliases
enable_cannshmem_device_bc = enable_cannshmem_mlir_device_bc
is_cannshmem_device_bc_enabled = is_cannshmem_mlir_device_bc_enabled


def _package_lib_dir() -> Path:
    return Path(__file__).resolve().parent / "lib"


def resolve_cannshmem_mlir_device_bitcode() -> Optional[Path]:
    """Resolve a single cannshmem_mlir device bitcode (first match)."""
    for env in ("FLAGTREE_CANNSHMEM_MLIR_DEVICE_BC", "FLAGTREE_CANNSHMEM_DEVICE_BC",
                "FLAGTREE_SHMEM_DEVICE_BC"):
        value = os.environ.get(env)
        if value:
            path = Path(value)
            if path.is_file():
                return path.resolve()
            raise FileNotFoundError(f"{env}={value!r} is not a file")

    lib = _package_lib_dir()
    for name in (
            "libdevice_cannshmem_mlir.bc",
            "libdevice_cannshmem_mlir.aiv.bc",
            "libdevice_cannshmem_mlir.mix.aiv.bc",
    ):
        candidate = lib / name
        if candidate.is_file():
            return candidate.resolve()
    return None


resolve_cannshmem_device_bitcode = resolve_cannshmem_mlir_device_bitcode


def get_cannshmem_mlir_link_bitcodes() -> list[str]:
    """Extra --link-aicore-bitcode paths when library=cannshmem_mlir is enabled."""
    if not is_cannshmem_mlir_device_bc_enabled():
        return []
    for env in ("FLAGTREE_CANNSHMEM_MLIR_DEVICE_BC", "FLAGTREE_CANNSHMEM_DEVICE_BC",
                "FLAGTREE_SHMEM_DEVICE_BC"):
        value = os.environ.get(env)
        if value:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"{env}={value!r} is not a file")
            return [str(path.resolve())]

    lib = _package_lib_dir()
    paths: list[str] = []
    for name in (
            "libdevice_cannshmem_mlir.mix.aiv.bc",
            "libdevice_cannshmem_mlir.mix.aic.bc",
            "libdevice_cannshmem_mlir.aiv.bc",
    ):
        candidate = lib / name
        if candidate.is_file():
            paths.append(str(candidate.resolve()))
    if paths:
        return paths
    alias = lib / "libdevice_cannshmem_mlir.bc"
    if alias.is_file():
        return [str(alias.resolve())]
    raise RuntimeError(
        "library='cannshmem_mlir' enabled but no device bitcode found; "
        "run raw/cannshmem_mlir/build_device_bc.sh or set FLAGTREE_CANNSHMEM_MLIR_DEVICE_BC")


def get_cannshmem_link_bitcodes() -> list[str]:
    return get_cannshmem_mlir_link_bitcodes()
