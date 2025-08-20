import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl

# Framework selection: prefer torch
try:
    import torch
    HAS_TORCH = True
    HAS_PADDLE = False
except Exception:
    import paddle
    HAS_TORCH = False
    HAS_PADDLE = True


def nvsmi(attrs):
    attrs = ','.join(attrs)
    cmd = ['nvidia-smi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(',')
    ret = [int(x) for x in ret]
    return ret


def do_bench_cudagraph(fn, rep=20, grad_to_none=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    """
    assert return_mode in ["min", "max", "mean", "median"]
    if not HAS_TORCH:
        # Fallback on non-torch builds
        return do_bench(fn, warmup=5, rep=rep, grad_to_none=grad_to_none, return_mode=return_mode)

    import torch as _torch

    if _torch.cuda.current_stream() == _torch.cuda.default_stream():
        raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
    # warmup
    fn()
    # step 1 - we estimate the amount of time the kernel call takes
    # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
    #       but it is probably good enough
    if grad_to_none is not None:
        for x in grad_to_none:
            x.detach_()
            x.requires_grad_(True)
            x.grad = None
    g = _torch.cuda.CUDAGraph()
    with _torch.cuda.graph(g):
        fn()
    _torch.cuda.synchronize()
    start_event = _torch.cuda.Event(enable_timing=True)
    end_event = _torch.cuda.Event(enable_timing=True)
    start_event.record()
    g.replay()
    end_event.record()
    _torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event)
    n_repeat = max(1, int(rep / estimate_ms))
    # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
    # host overhead
    g = _torch.cuda.CUDAGraph()
    with _torch.cuda.graph(g):
        for i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
    _torch.cuda.synchronize()
    # measure time and return
    ret = []
    n_retries = 10
    for i in range(n_retries):
        start_event = _torch.cuda.Event(enable_timing=True)
        end_event = _torch.cuda.Event(enable_timing=True)
        start_event.record()
        g.replay()
        end_event.record()
        _torch.cuda.synchronize()
        ret += [start_event.elapsed_time(end_event) / n_repeat]
    times = _torch.tensor(ret)
    return getattr(_torch, return_mode)(times).item()


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
             device_type="cuda"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    assert return_mode in ["min", "max", "mean", "median"]

    if HAS_TORCH:
        import torch as _torch

        di = _torch._dynamo.device_interface.get_interface_for_device(device_type)

        fn()
        di.synchronize()

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2
        # doesn't contain any input data before the run
        if fast_flush:
            cache = _torch.empty(int(256e6 // 4), dtype=_torch.int, device=device_type)
        else:
            cache = _torch.empty(int(256e6), dtype=_torch.int8, device=device_type)

        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
        start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
        # Warm-up
        for _ in range(n_warmup):
            fn()
        # Benchmark
        for i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            cache.zero_()
            start_event[i].record()
            fn()
            end_event[i].record()
        # Record clocks
        di.synchronize()
        times = _torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=_torch.float)
        if quantiles is not None:
            ret = _torch.quantile(times, _torch.tensor(quantiles, dtype=_torch.float)).tolist()
            if len(ret) == 1:
                ret = ret[0]
            return ret
        return getattr(_torch, return_mode)(times).item()

    # Fallback (paddle or no torch): use perf_counter + synchronize
    import time
    import numpy as _np
    import paddle as _pd

    # make sure device is gpu
    try:
        _pd.device.set_device('gpu')
    except Exception:
        pass

    # warmup run + sync
    fn()
    try:
        _pd.device.synchronize()
    except Exception:
        pass

    # L2 flush buffer ~256MB
    if fast_flush:
        cache = _pd.empty([int(256e6 // 4)], dtype=_pd.int32)
    else:
        cache = _pd.empty([int(256e6)], dtype=_pd.int8)

    # estimate
    t0 = time.perf_counter()
    for _ in range(5):
        cache[:] = 0
        fn()
    try:
        _pd.device.synchronize()
    except Exception:
        pass
    t1 = time.perf_counter()
    estimate_ms = (t1 - t0) * 1000.0 / 5.0

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # warmup
    for _ in range(n_warmup):
        fn()
    try:
        _pd.device.synchronize()
    except Exception:
        pass

    times_ms = []
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                try:
                    x.grad = None  # not valid in paddle; safe-guard
                except Exception:
                    pass
        cache[:] = 0
        t0 = time.perf_counter()
        fn()
        try:
            _pd.device.synchronize()
        except Exception:
            pass
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times = _np.array(times_ms, dtype=_np.float32)
    if quantiles is not None:
        qs = _np.quantile(times, _np.array(quantiles, dtype=_np.float32)).tolist()
        if isinstance(qs, list) and len(qs) == 1:
            qs = qs[0]
        return qs
    if return_mode == "min":
        return float(times.min())
    if return_mode == "max":
        return float(times.max())
    if return_mode == "median":
        return float(_np.median(times))
    return float(times.mean())


def assert_close(x, y, atol=None, rtol=None, err_msg=''):
    import numpy as np

    # canonicalize arguments to be tensors -> numpy
    def to_numpy(t):
        if HAS_TORCH:
            import torch
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            if t.dtype == torch.bfloat16:
                t = t.float()
            return t.cpu().detach().numpy()
        else:
            import paddle
            if not isinstance(t, paddle.Tensor):
                t = paddle.to_tensor(t)
            # if Paddle has bfloat16 and used, cast to float32 for compare
            if hasattr(paddle, "bfloat16") and t.dtype == paddle.bfloat16:
                t = paddle.cast(t, dtype=paddle.float32)
            return t.numpy()

    x_np = to_numpy(x)
    y_np = to_numpy(y)

    # absolute tolerance
    if atol is None:
        atol = 1e-2
    atol = atol(x_np.dtype) if callable(atol) else atol
    # relative tolerance hook
    if rtol is None:
        rtol = 0.
    rtol = rtol(x_np.dtype) if callable(rtol) else rtol

    # handle size==1 separately
    if x_np.size > 1 or y_np.size > 1:
        np.testing.assert_allclose(x_np, y_np, atol=atol, rtol=rtol, equal_nan=True)
        return
    if not np.allclose(x_np, y_np, atol=atol, rtol=rtol):
        raise AssertionError(f'{err_msg} {x_np} is not close to {y_np} (atol={atol}, rtol={rtol})')


class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = '',
        ylabel: str = '',
        x_log: bool = False,
        y_log: bool = False,
        color=None,
        styles=None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        # plot info
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import os

        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max

        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = df[y + '-min'], df[y + '-max']
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                if not y_min.isnull().all() and not y_max.isnull().all():
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")
        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
            if save_path:
                html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
        if save_path:
            html.write("</body></html>\n")
            html.close()
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


def get_dram_gbps(device=None):
    ''' return DRAM bandwidth in GB/s '''
    from .runtime import driver
    if HAS_TORCH:
        import torch as _torch
        if not device:
            device = _torch.cuda.current_device()
    else:
        if device is None:
            device = 0
    mem_clock_khz = driver.active.utils.get_device_properties(device)["mem_clock_rate"]  # in kHz
    bus_width = driver.active.utils.get_device_properties(device)["mem_bus_width"]
    bw_gbps = mem_clock_khz * bus_width * 2 / 1e6 / 8  # In GB/s
    return bw_gbps


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    from .runtime import driver

    def _dtype_name(d):
        s = str(d)
        # torch.float32 -> float32; paddle.float32 -> float32
        if "." in s:
            s = s.split(".")[-1]
        return s

    if HAS_TORCH:
        import torch as _torch
        if not device:
            device = _torch.cuda.current_device()
        num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
        capability = _torch.cuda.get_device_capability(device)
        if capability[0] < 8:
            assert dtype == _torch.float16
            ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
        else:
            if dtype in [_torch.float32, _torch.int32]:
                ops_per_sub_core = 256
            elif dtype in [_torch.float16, _torch.bfloat16, _torch.int16]:
                ops_per_sub_core = 512
            elif dtype in [_torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
                ops_per_sub_core = 1024
            else:
                raise RuntimeError("dtype not supported")
        tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
        return tflops

    # paddle path
    import paddle as _pd
    if device is None:
        device = 0
    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    capability = _pd.device.cuda.get_device_capability()
    name = _dtype_name(dtype)
    if capability[0] < 8:
        if name != "float16":
            raise AssertionError("dtype must be float16 for pre-Ampere")
        ops_per_sub_core = 256
    else:
        if name in ["float32", "int32"]:
            ops_per_sub_core = 256
        elif name in ["float16", "bfloat16", "int16"]:
            ops_per_sub_core = 512
        elif name in ["int8"] or dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
            ops_per_sub_core = 1024
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


# create decorator that wraps test function into
# a cuda-memcheck system call


def cuda_memcheck(**target_kwargs):

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != "cuda-memcheck":
                path = os.path.realpath(test_fn.__globals__["__file__"])
                # get path of current file
                env = {"PATH": os.environ["PATH"], "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                out = subprocess.run(["cuda-memcheck", "pytest", "-vs", cmd], capture_output=True, env=env)
                assert out.returncode == 0, "cuda-memcheck returned an error: bounds checking failed"
                assert "ERROR SUMMARY: 0 errors" in str(out.stdout)
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "1"])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}",
        ])
        subprocess.check_output([
            "nvidia-smi",
            "-i",
            "0",
            f"--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}",
        ])
        cur_sm_clock = nvsmi(["clocks.current.sm"])[0]
        cur_mem_clock = nvsmi(["clocks.current.memory"])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f"GPU SMs must run at {ref_sm_clock} MHz"
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f"GPU SMs must run at {ref_mem_clock} MHz"
        tflops = 1e-6 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 1e-3
        yield tflops, gbps
    finally:
        subprocess.check_output(["nvidia-smi", "-i", "0", "-pm", "0"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rgc"])
        subprocess.check_output(["nvidia-smi", "-i", "0", "-rmc"])


def get_max_simd_tflops(dtype, clock_rate, device=None):
    from .runtime import driver

    def _torch_path():
        import torch as _torch
        if not device:
            dev = _torch.cuda.current_device()
        else:
            dev = device
        num_subcores = driver.active.utils.get_device_properties(dev)["multiprocessor_count"] * 4
        capability = _torch.cuda.get_device_capability()
        if capability[0] < 8:
            if dtype == _torch.float32:
                ops_per_sub_core = 32  # 2*16
            elif dtype == _torch.float16:
                ops_per_sub_core = 64
            else:
                raise RuntimeError("dtype not supported")
        else:
            if dtype == _torch.float32:
                ops_per_sub_core = 32
            elif dtype in [_torch.float16, _torch.bfloat16]:
                ops_per_sub_core = 64
            else:
                raise RuntimeError("dtype not supported")
        tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
        return tflops

    def _paddle_path():
        import paddle as _pd
        dev = 0 if device is None else device
        num_subcores = driver.active.utils.get_device_properties(dev)["multiprocessor_count"] * 4
        capability = _pd.device.cuda.get_device_capability()
        name = str(dtype).split(".")[-1]
        if capability[0] < 8:
            if name == "float32":
                ops_per_sub_core = 32
            elif name == "float16":
                ops_per_sub_core = 64
            else:
                raise RuntimeError("dtype not supported")
        else:
            if name == "float32":
                ops_per_sub_core = 32
            elif name in ["float16", "bfloat16"]:
                ops_per_sub_core = 64
            else:
                raise RuntimeError("dtype not supported")
        tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
        return tflops

    if HAS_TORCH:
        return _torch_path()
    return _paddle_path()