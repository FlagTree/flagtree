from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Tuple,
)

from .utils import (
    get_byte_per_numel,
    next_power_of_2,
    num_vector_core,
)
from .autotuner import Config


@dataclass
class AxisInfo:
    name: str
    index: int
    length: int

    prefix: str = ""
    split_name: str = ""
    tiling_name: str = ""
    is_split_axis: bool = False
    is_tiling_axis: bool = False

    @property
    def is_reduction(self):
        return self.prefix == "r"


class KernelMeta:
    def __init__(
        self,
        axis_sizes: Dict[str, int],
        split_params: Dict[str, str],
        tiling_params: Dict[str, str],
        low_dims: List[str],
        dtype: torch.dtype,
        persistent_reduction: bool,
        dual_reduction: bool,
    ):
        self._validate_axis(axis_sizes, split_params, tiling_params, low_dims)

        axis_dict = {}
        idx = 0
        for name, length in axis_sizes.items():
            prefix = ""
            if name.startswith("r"):
                prefix = "r"
                name = name[1:]

            is_split_axis = name in split_params
            is_tiling_axis = name in tiling_params
            split_name = "" if name not in split_params else split_params[name]
            tiling_name = "" if name not in tiling_params else tiling_params[name]

            axis_dict[name] = AxisInfo(
                name=name,
                index=idx,
                length=length,
                prefix=prefix,
                split_name=split_name,
                tiling_name=tiling_name,
                is_split_axis=is_split_axis,
                is_tiling_axis=is_tiling_axis,
            )
            idx += 1

        self.axis_info = list(axis_dict.values())
        self.split_axis = [x for x in axis_dict.values() if x.is_split_axis]
        self.tiling_axis = [x for x in axis_dict.values() if x.is_tiling_axis]
        self.low_dims_axis = [x for x in axis_dict.values() if x.name in low_dims]
        self.dtype = dtype
        self.persistent_reduction = persistent_reduction
        self.dual_reduction = dual_reduction

    @classmethod
    def _validate_axis(
        cls,
        axis_sizes: Dict[str, int],
        split_params: Dict[str, str],
        tiling_params: Dict[str, str],
        low_dims: List[str],
    ) -> None:
        for axis_name in axis_sizes.keys():
            if axis_name.startswith("r") and len(axis_name) == 1:
                raise ValueError("The name of a reduction axis is empty!")

        def check_keys(params: List[str], context="parameter"):
            for k in params:
                if k not in axis_sizes and ("r" + k) not in axis_sizes:
                    raise KeyError(
                        f"{context} '{k}' not found in known axes: {axis_sizes.keys()}"
                    )

        check_keys(split_params.keys(), "split axis")
        check_keys(tiling_params.keys(), "tiling axis")
        check_keys(low_dims, "low dim axis")


@dataclass
class BlockInfo:
    block_name: str  # e.g., XBLOCK
    sub_block_name: str  # e.g., XBLOCK_SUB
    block_size: int
    sub_block_size: int


"""
Generate possible candidate tiling configs for benchmarking
"""
class TileGenerator:
    num_warps = 1
    num_stages = 1
    stop_bytes = 1024
    max_tile_bytes = 16384 * 4

    def __init__(self, kernel_meta: KernelMeta):
        self.kernel_meta = kernel_meta
        self.persistent_reduction = self.kernel_meta.persistent_reduction
        self.dual_reduction = self.kernel_meta.dual_reduction

        self.blocks = self.init_blocks_info(kernel_meta)
        self.split_magic_nums = []
        for axis in self.kernel_meta.axis_info:
            if axis.is_split_axis:
                self.split_magic_nums.append(
                    (axis.length + num_vector_core - 1) // num_vector_core
                )
            else:
                self.split_magic_nums.append(-1)

        self.candidates_blk_sizes: List[Tuple[int, ...]] = []
        self.configs = {}
        self.dtype_bytes = get_byte_per_numel(kernel_meta.dtype)
        self.stop_numel = self.stop_bytes // self.dtype_bytes
        self.max_tile_numel = self.max_tile_bytes // self.dtype_bytes

    @classmethod
    def init_blocks_info(cls, kernel_meta: KernelMeta) -> List[BlockInfo]:
        blocks = []
        for axis in kernel_meta.axis_info:
            block_name = axis.split_name
            sub_block_name = axis.tiling_name
            block_size = axis.length
            sub_block_size = block_size
            blocks.append(
                BlockInfo(block_name, sub_block_name, block_size, sub_block_size)
            )

        return blocks

    @classmethod
    def get_key_from_dict(cls, kwargs: Dict[str, int]):
        return tuple(sorted(kwargs.items()))

    def valid_tile_numel(self, tile_numel: int) -> bool:
        return tile_numel <= self.max_tile_numel

    def calculate_tile_numel(self) -> int:
        tile_numel = 1
        for axis in self.kernel_meta.axis_info:
            if axis.is_tiling_axis:
                tile_numel *= self.blocks[axis.index].sub_block_size
            else:
                # this axis's tiling size is the same as block size
                tile_numel *= self.blocks[axis.index].block_size

        return tile_numel

    def add_to_configs(self, cand_sizes) -> None:
        kwargs = {}
        for axis in self.kernel_meta.axis_info:
            if not (axis.is_split_axis or axis.is_tiling_axis):
                continue

            block_info = self.blocks[axis.index]
            if axis.is_split_axis:
                kwargs[block_info.block_name] = cand_sizes[axis.index]
            if axis.is_tiling_axis:
                kwargs[block_info.sub_block_name] = next_power_of_2(
                    block_info.sub_block_size
                )

        tile_numel = 1
        for axis in self.kernel_meta.axis_info:
            if not (axis.is_split_axis or axis.is_tiling_axis):
                tile_numel *= self.blocks[axis.index].block_size
                continue

            if axis.is_tiling_axis:
                tile_numel *= kwargs.get(self.blocks[axis.index].sub_block_name, 1)
            else:
                tile_numel *= kwargs.get(self.blocks[axis.index].block_name, 1)

        key = self.get_key_from_dict(kwargs)
        if self.valid_tile_numel(tile_numel) and key not in self.configs:
            self.configs[key] = Config(
                kwargs, num_warps=self.num_warps, num_stages=self.num_stages
            )

    def descend_one_axis(self, axis_idx: int, is_split=False) -> bool:
        def calc_total_programs():
            grids = []
            for axis in self.kernel_meta.split_axis:
                block_size = self.blocks[axis.index].block_size
                programs = (axis.length + block_size - 1) // block_size
                grids.append(programs)

            return functools.reduce(lambda x, y: x * y, grids) if grids else 1

        reached_stop_numel = False
        slow_descend_split = False
        magic_descend_split = False
        if not is_split and len(self.candidates_blk_sizes) == 0:
            self.candidates_blk_sizes.append(
                tuple([x.block_size for x in self.blocks])
            )

        axis = self.kernel_meta.axis_info[axis_idx]
        while True:
            for cand_sizes in self.candidates_blk_sizes:
                self.add_to_configs(cand_sizes)

            # tile numel reached threshold
            tile_numel = self.calculate_tile_numel()
            if tile_numel <= self.stop_numel:
                self.add_to_configs([x.block_size for x in self.blocks])
                reached_stop_numel = True
                break

            numel = (
                self.blocks[axis_idx].block_size
                if is_split
                else self.blocks[axis_idx].sub_block_size
            )
            if numel == 1:
                self.add_to_configs([x.block_size for x in self.blocks])
                break

            if is_split:
                if self.persistent_reduction and axis.is_reduction:
                    reached_stop_numel = True
                    break
                total_programs = calc_total_programs()
                if total_programs > num_vector_core:
                    break
                if total_programs > num_vector_core // 2 or self.dual_reduction:
                    if len(self.candidates_blk_sizes) > 2:
                        self.candidates_blk_sizes.pop(0)
                    self.candidates_blk_sizes.append(
                        tuple([x.block_size for x in self.blocks])
                    )

                if (
                    not magic_descend_split
                    and (numel // 2) <= self.split_magic_nums[axis_idx]
                ):
                    self.blocks[axis_idx].block_size = self.split_magic_nums[axis_idx]
                    self.blocks[axis_idx].sub_block_size = self.blocks[axis_idx].block_size
                    magic_descend_split = True
                    continue

                self.blocks[axis_idx].block_size = numel // 2
                self.blocks[axis_idx].sub_block_size = self.blocks[axis_idx].block_size
                if calc_total_programs() > num_vector_core:
                    slow_descend_split = True
                step = numel // 4 if numel // 4 > 1 else 1
                self.blocks[axis_idx].block_size = (
                    numel // 2 if not slow_descend_split else numel - step
                )
                self.blocks[axis_idx].sub_block_size = self.blocks[axis_idx].block_size
            else:
                self.blocks[axis_idx].sub_block_size = next_power_of_2(numel // 2)

        return reached_stop_numel

    def descend_all_low_dims(self) -> None:
        low_dim_numels = [self.blocks[x.index].sub_block_size for x in self.kernel_meta.low_dims_axis]
        if not low_dim_numels:
            return

        def descend_all_axis(min_numel):
            for axis in self.kernel_meta.low_dims_axis:
                if axis.is_reduction and self.persistent_reduction:
                    continue
                numel = self.blocks[axis.index].sub_block_size
                if numel == 1:
                    continue
                if min_numel > 1 and abs(numel - min_numel) / min_numel < 0.2:
                    continue
                self.blocks[axis.index].sub_block_size = next_power_of_2(numel // 2)

        if len(self.candidates_blk_sizes) == 0:
            # means there is no split axis and tiling_not_low_dim axis
            # so we need to init the candidates_blk_sizes
            self.candidates_blk_sizes.append(
                tuple([x.block_size for x in self.blocks])
            )
        count = 0
        tile_numel = self.calculate_tile_numel()
        while tile_numel > self.stop_numel and count < 100:
            count += 1
            tile_numel = self.calculate_tile_numel()
            for cand_sizes in self.candidates_blk_sizes:
                self.add_to_configs(cand_sizes)
            min_numel = min(low_dim_numels)
            descend_all_axis(min_numel)
            new_tile_numel = self.calculate_tile_numel()
            if tile_numel == new_tile_numel:
                descend_all_axis(0)

    def descend_split_tiling(self):

        tiling_not_low_dims = [
            x
            for x in self.kernel_meta.tiling_axis
            if x not in self.kernel_meta.low_dims_axis
        ]

        def descend_split_axis():
            for axis in self.kernel_meta.split_axis:
                if self.descend_one_axis(axis.index, is_split=True):
                    return True
            return self.calculate_tile_numel() <= self.stop_numel

        def descend_tiling_not_low_dims():
            for axis in tiling_not_low_dims:
                if axis.is_reduction and self.persistent_reduction:
                    continue
                if self.descend_one_axis(axis.index):
                    return True
            return self.calculate_tile_numel() <= self.stop_numel

        def descend_low_dims():
            for axis in self.kernel_meta.tiling_axis:
                if axis.is_reduction and self.persistent_reduction:
                    continue
                if axis in tiling_not_low_dims:
                    continue
                if self.descend_one_axis(axis.index):
                    return True
            return self.calculate_tile_numel() <= self.stop_numel

        while True:
            # descend split axis
            if descend_split_axis():
                break

            if len(self.candidates_blk_sizes) > 0:
                candi_blk = self.candidates_blk_sizes[0]
                for i, blk_size in enumerate(candi_blk):
                    self.blocks[i].sub_block_size = blk_size

            # descend tiling but not low dims
            if descend_tiling_not_low_dims():
                break

            # descend low dims
            self.descend_all_low_dims()
            break
