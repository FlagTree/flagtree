# Copyright 2026-     FlagOS Contributors
#
# Device-bitcode enablement for library="cannshmem_mlir".
# Host helpers: triton.experimental.tle.raw.cannshmem (re-exported).

from triton.experimental.tle.raw.cannshmem.utils import (
    bench_us,
    cleanup_symmetric_tensor,
    create_symmetric_tensor,
    init_cannshmem_by_torch_pg,
)

from .utils import (
    enable_cannshmem_device_bc,
    enable_cannshmem_mlir_device_bc,
    get_cannshmem_link_bitcodes,
    get_cannshmem_mlir_link_bitcodes,
    is_cannshmem_device_bc_enabled,
    is_cannshmem_mlir_device_bc_enabled,
    resolve_cannshmem_device_bitcode,
    resolve_cannshmem_mlir_device_bitcode,
)

__all__ = [
    "bench_us",
    "cleanup_symmetric_tensor",
    "create_symmetric_tensor",
    "enable_cannshmem_device_bc",
    "enable_cannshmem_mlir_device_bc",
    "get_cannshmem_link_bitcodes",
    "get_cannshmem_mlir_link_bitcodes",
    "init_cannshmem_by_torch_pg",
    "is_cannshmem_device_bc_enabled",
    "is_cannshmem_mlir_device_bc_enabled",
    "resolve_cannshmem_device_bitcode",
    "resolve_cannshmem_mlir_device_bitcode",
]
