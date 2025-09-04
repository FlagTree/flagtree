import os
import pickle
import torch
import uuid
import numpy as np
from pathlib import Path
from itertools import chain
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

# ------------------------
# Utils
# ------------------------


def load_binary(name, kernel, shared, device):
    return None, kernel, 1, 0


class AIPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = load_binary
        properties_dict = {"max_shared_mem": 256 * 1024, "multiprocessor_count": 4, "max_num_regs": 32, "warpSize": 4}
        self.get_device_properties = lambda device: properties_dict


# ------------------------
# Launcher
# ------------------------


def _reset_output_path(ex):
    output_dir = f"{os.getcwd()}/compass_dsl_{ex._func_name}_restore_{uuid.uuid4().hex}"
    ex._output_dir = output_dir
    ex._gbuilder_dir = f"{ex._output_dir}/gbuilder"
    ex._op_lib_path = f"{ex._gbuilder_dir}/op_lib/{ex._func_name}.o"


def _get_cpu_origin_tensor(tensor):
    origin_tensor = tensor
    while (base := origin_tensor._base) is not None:
        origin_tensor = base

    return origin_tensor.cpu().contiguous()


def _get_np_array_from_strided_buffer(tensor, sb):
    dtype = str(sb.dtype).split(".")[-1]
    itemsize = sb.element_size()
    offset = (sb.data_ptr() - sb._base.data_ptr())
    shape = sb.size()
    stride = [x * itemsize for x in sb.stride()]

    return np.ndarray(
        shape,
        dtype,
        tensor.numpy(),
        offset,
        stride,
    )


class AIPULauncher(object):

    def __init__(self, src, metadata):
        self.constants = src.constants

    def lanch_kernel(self, ex, np_args, tail_args, totoal_pid_size):
        convert_map = {}
        real_args = []
        for i, arg in enumerate(np_args):
            if isinstance(arg, np.ndarray) and arg.dtype == "bool":
                real_args.append(arg.astype(np.int8))
                convert_map[i] = np.bool_
            elif isinstance(arg, np.ndarray) and arg.dtype == "int64":
                real_args.append(arg.astype(np.int32))
                convert_map[i] = np.int64
            else:
                real_args.append(arg)

        tec_num = 4
        for i in range((totoal_pid_size + tec_num - 1) // tec_num):
            tail_args[3] = i
            ex(*(real_args + tail_args))

        for i, arg in enumerate(real_args):
            if i in convert_map.keys():
                np.copyto(np_args[i], arg.astype(convert_map[i]))

    # TODO(aipu-teams): This is just a temporary solution for now, because the real driver interface is not ready yet.
    # These code will be refactor later.
    def __call__(self, gridx_size, gridy_size, gridz_size, stream, function, *args):
        try:
            from flag_gems.utils.tensor_wrapper import StridedBuffer
        except ImportError:
            StridedBuffer = torch.Tensor

        ex = pickle.loads(function)
        _reset_output_path(ex)
        np_args = []
        sb_maps = {}
        args = [arg for i, arg in enumerate(args[4:]) if i not in chain(*self.constants.keys())]

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                np_args.append(_get_cpu_origin_tensor(arg).numpy())
            elif isinstance(arg, StridedBuffer):
                tensor = _get_cpu_origin_tensor(arg)
                np_args.append(_get_np_array_from_strided_buffer(tensor, arg))
                sb_maps[i] = tensor
            else:
                np_args.append(arg)

        tail_args = [gridx_size, gridy_size, gridz_size, 0, 0, 0]
        total_pid_size = gridx_size * gridy_size * gridz_size
        self.lanch_kernel(ex, np_args, tail_args, total_pid_size)

        for i, param_info in enumerate(ex._cur_param_infos):
            if param_info.is_output_tensor:
                if isinstance(args[i], torch.Tensor):
                    args[i].copy_(torch.from_numpy(np_args[i]))
                else:
                    args[i]._base.copy_(sb_maps[i])


class AIPUDriver(DriverBase):

    def __init__(self):
        self.utils = AIPUUtils()  # TODO: make static
        self.launcher_cls = AIPULauncher

        import torch
        self.get_current_stream = lambda x: x
        self.get_current_device = torch.aipu.current_device

        super().__init__()

    def get_current_target(self):
        warp_size = 4
        return GPUTarget("aipu", "x2", warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("aipu", 0)

    def get_device_interface(self):
        import torch
        return torch.aipu

    @staticmethod
    def is_active():
        import torch
        from torch.utils import cpp_extension

        try:
            torch.aipu.is_available()
        except AttributeError:
            current_dir = Path(__file__).resolve().parent
            extra_ldflags = [f"-L{x.strip()}" for x in os.getenv("LD_LIBRARY_PATH", "").split(":") if x.strip() != ""]
            extra_ldflags.append("-laipudrv")
            module = cpp_extension.load(
                name="aipu", sources=[current_dir / "aipu_torch_dev.cpp"],
                extra_include_paths=[os.getenv("ZHOUYI_LINUX_DRIVER_HOME") + "/driver/umd/include"],
                extra_ldflags=extra_ldflags, verbose=True)

            torch.utils.rename_privateuse1_backend("aipu")
            torch._register_device_module("aipu", module)
            torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
        return torch.aipu.is_available()

    # TODO(aipu-teams): Support bechmarker later.
    def get_benchmarker(self):

        def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
            return [float("inf"), float("inf"), float("inf")]

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='aipu')
