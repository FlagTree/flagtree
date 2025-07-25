from __future__ import annotations

import builtins
import os
import time
import inspect
from typing import Dict

import json
import os
import hashlib

from ..testing import do_bench, do_bench_cudagraph
from .jit import KernelInterface
from .errors import OutOfResources
from .cache import default_cache_dir

import filelock


def build_best_config_hash(args_names, key):
    cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
    hasher = hashlib.sha256()
    hasher.update(f"{'_'.join(args_names) + str(key)}\n".encode())
    cfg_hash = hasher.hexdigest()
    cfg_hash_dir = os.path.join(cache_dir, cfg_hash)
    cfg_hash_file = os.path.splitext(cfg_hash)[0] + ".best_config"
    cfg_hash_file = os.path.join(cfg_hash_dir, cfg_hash_file)
    return cfg_hash_dir, cfg_hash_file


def load_best_config(args_names, key):
    _, cfg_hash_file = build_best_config_hash(args_names, key)
    if os.path.exists(cfg_hash_file):
        with open(cfg_hash_file) as fd:
            best_config = json.loads(fd.read())
            num_warps = best_config.pop('num_warps') if 'num_warps' in best_config else 4
            num_stages = best_config.pop('num_stages') if 'num_stages' in best_config else 1
            return best_config, num_warps, num_stages
    return None


def save_best_config(cfg, args_names, key):
    cfg_hash_dir, cfg_hash_file = build_best_config_hash(args_names, key)
    if os.path.exists(cfg_hash_dir):
        return
    os.makedirs(cfg_hash_dir, exist_ok=True)
    lock = filelock.FileLock(f"{cfg_hash_file}.lock")
    with lock:
        if os.path.exists(cfg_hash_file):
            return
        with open(cfg_hash_file, "w") as fd:
            fd.write(json.dumps({
                **cfg.kwargs,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }))


class Autotuner(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        self.arg_names = arg_names

        # Reset to zero or restore values
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args, exception: 0
        if pre_hook:
            self.pre_hook = pre_hook
        elif (len(self.reset_idx) > 0 or len(self.restore_idx) > 0):

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
        elif len(self.restore_idx) > 0:

            def _post_hook(args, exception):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                self.restore_copies = []

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.num_warmups = warmup
        self.num_reps = rep
        import torch
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        # cache_fn_map fmt: {"fn_cache_key: [hash_cache_file_0, hash_cache_file_1, ...], [so_path_0, so_path_1, ...]]"}
        self.cache_fn_map = dict()

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(args, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(args, exception=None)

        try:
            if self.use_cuda_graph:
                import torch
                with torch.cuda.stream(torch.cuda.Stream()):
                    bench_res = do_bench_cudagraph(kernel_call, rep=self.num_reps, return_mode="median")
                return bench_res
            bench_results = do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure):
            bench_results = float("inf") if self.use_cuda_graph else [float("inf"), float("inf"), float("inf")]

        cache_key = str(self.get_jit_func().cache_key)
        check_key = self.cache_fn_map.get(str(cache_key), None)
        if not check_key:
            self.cache_fn_map.setdefault(cache_key, [[], []])
        hash_cache_file = str(self.get_jit_func().hash_cache_file)
        so_path = ''
        if self.get_jit_func().so_path:
            so_path = self.get_jit_func().so_path.split('/')[-2]
        self.cache_fn_map[cache_key][0].append(hash_cache_file)
        self.cache_fn_map[cache_key][1].append(so_path)
        return bench_results

    def get_jit_func(self):
        if hasattr(self.fn, "cache_key"):
            # for autotune + jit
            return self.fn
        elif hasattr(self.fn.fn, "cache_key"):
            # for autotune + heuristics + jit
            return self.fn.fn
        else:
            msg = f'Current {self.fn} or {self.fn.fn} has no attribute cache_key.'
            raise RuntimeError(msg)

    def run(self, *args, **kwargs):
        only_save_best_config_cache = os.environ.get("TRITON_ONLY_SAVE_BEST_CONFIG_CACHE", "0") == "1"
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            divisibility = 16
            for arg in args:
                if hasattr(arg, "data_ptr"):
                    key.append(arg.dtype)
                    key.append(arg.data_ptr() % divisibility == 0)
                elif isinstance(arg, int):
                    key.append(arg)
            key = tuple(key)
            if key not in self.cache:
                # else:
                if not only_save_best_config_cache:
                    # prune configs
                    used_cached_result = False
                    pruned_configs = self.prune_configs(kwargs)
                    bench_start = time.time()
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    self.pre_hook(args, reset_only=True)
                    self.configs_timings = timings
                else:
                    load_config = load_best_config(self.arg_names, key)
                    if load_config:
                        best_config, num_warps, num_stages = load_config
                        config = Config(best_config, num_warps, num_stages)
                        self.cache[key] = config
                        self.pre_hook(args, reset_only=True)
                    else:
                        pruned_configs = self.prune_configs(kwargs)
                        bench_start = time.time()
                        timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = builtins.min(timings, key=timings.get)
                        list_keys = list(timings.keys())
                        best_key_index = list_keys.index(builtins.min(timings, key=timings.get))
                        save_best_config(self.cache[key], self.arg_names, key)
                        self.pre_hook(args, reset_only=True)
                        self.configs_timings = timings
                        cache_key = str(self.get_jit_func().cache_key)
                        check_key = self.cache_fn_map.get(cache_key, None)
                        if check_key:
                            best_cache_file = self.cache_fn_map[cache_key][0][best_key_index]
                            best_so_path = self.cache_fn_map[cache_key][1][best_key_index]
                            ck_list = [best_cache_file, best_so_path]
                            for i in range(len(ck_list)):
                                for tmp_key in check_key[i]:
                                    if ck_list[i] != tmp_key:
                                        del_cache_file = os.path.join(
                                            os.environ.get('TRITON_CACHE_DIR', default_cache_dir()), tmp_key)
                                        import shutil
                                        shutil.rmtree(del_cache_file, ignore_errors=True)
                        self.cache_fn_map.clear()
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__} finished after "
                  f"{self.bench_time:.2f}s; best config selected: {self.best_config};")
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_ctas: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs, **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=25, rep=100, use_cuda_graph=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: Warmup time (in ms) to pass to benchmarking, defaults to 25.
    :type warmup: int
    :param rep: Repetition time (in ms) to pass to benchmarking, defaults to 100.
    :type rep: int
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph)

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[list[Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
