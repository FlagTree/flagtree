import ast
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.adjust_kernel_param import auto_adjust_block_sizes, clear_analysis_cache
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import BackendCompilerError, OutOfResources, PTXASError


@pytest.fixture(autouse=True)
def aabs_knobs(monkeypatch):
    monkeypatch.setattr(knobs.autotuning, "adjust_block_size", True)
    monkeypatch.setattr(knobs.autotuning, "cache", False)
    monkeypatch.setattr(knobs.runtime, "interpret", False)
    clear_analysis_cache()
    yield
    clear_analysis_cache()


def bench_once(kernel_call, quantiles):
    kernel_call()
    return [1.0, 1.0, 1.0]


@pytest.mark.parametrize("extent", [-3, 0, 64])
@pytest.mark.parametrize("block_size", [128, 2048])
def test_aabs_extent_adjustment(extent, block_size):

    @triton.jit
    def kernel(src, init_blocks, BLOCK_K: tl.constexpr):
        offsets = tl.arange(0, BLOCK_K)
        tl.load(src + offsets, mask=offsets < init_blocks, other=0.0)

    config = triton.Config({"BLOCK_K": block_size})
    current = config.all_kwargs()
    auto_adjust_block_sizes({"src": None, "init_blocks": extent}, kernel, [config], current, config)
    expected = block_size if extent <= 0 else 64
    assert current["BLOCK_K"] == expected
    assert config.kwargs["BLOCK_K"] == expected


@pytest.mark.parametrize("cached", [False, True])
def test_fallback_respects_early_config_prune(monkeypatch, cached):

    @triton.jit
    def kernel(src, dst, N, init_blocks, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
        tl.static_assert(BLOCK_K > 16)
        offsets = tl.arange(0, BLOCK_K)
        x = tl.load(src + offsets, mask=offsets < init_blocks, other=0.0)
        tl.store(dst + tl.arange(0, BLOCK_M), tl.sum(x, 0))

    configs = [triton.Config({"BLOCK_K": 2048, "BLOCK_M": 128}), triton.Config({"BLOCK_K": 128, "BLOCK_M": 64})]

    def prune(configs, named_args, **kwargs):
        return [config for config in configs if config.kwargs["BLOCK_M"] <= named_args["N"]]

    attempts = []

    def run(src, dst, N, init_blocks, **meta):
        attempts.append((meta["BLOCK_K"], meta["BLOCK_M"]))
        if meta["BLOCK_K"] <= 16:
            raise CompileTimeAssertionFailure(None, ast.Pass(), "BLOCK_K > 16")
        assert meta["BLOCK_M"] <= N
        return "launched"

    monkeypatch.setattr(kernel, "run", run)
    tuner = Autotuner(kernel, kernel.arg_names, configs, ["N"], None, None,
                      prune_configs_by={"early_config_prune": prune}, do_bench=bench_once)
    if cached:
        tuner.cache[(64, )] = triton.Config({"BLOCK_K": 8, "BLOCK_M": 64})

    assert tuner.run(None, None, 64, 8) == "launched"
    assert attempts == [(8, 64)] * (1 if cached else 2) + [(128, 64)]
    assert tuner.cache[(64, )].kwargs == {"BLOCK_K": 128, "BLOCK_M": 64}
    assert [config.kwargs["BLOCK_K"] for config in configs] == [2048, 128]


@pytest.mark.parametrize("error", [
    OutOfResources(256, 128, "test resource"),
    CompileTimeAssertionFailure(None, ast.Pass(), "invalid block size"),
    PTXASError("invalid block size"),
    BackendCompilerError("invalid block size"),
])
@pytest.mark.parametrize("cache_results", [False, True])
def test_fallback_config_hook_and_memory_cache(monkeypatch, error, cache_results):

    @triton.jit
    def kernel(desc, N, BLOCK_SIZE: tl.constexpr):
        pass

    descriptor = SimpleNamespace(block_shape=None)
    hooks = []

    def hook(nargs):
        assert nargs["desc"] is descriptor
        assert nargs["N"] == 64
        descriptor.block_shape = nargs["BLOCK_SIZE"]
        hooks.append(nargs["BLOCK_SIZE"])

    configs = [triton.Config({"BLOCK_SIZE": size}, pre_hook=hook) for size in (256, 128)]
    attempts = []

    def run(desc, N, **meta):
        attempts.append(meta["BLOCK_SIZE"])
        assert desc.block_shape == meta["BLOCK_SIZE"]
        if meta["BLOCK_SIZE"] == 256:
            raise error
        return "launched"

    def unexpected_hook(*args, **kwargs):
        pytest.fail("Benchmark reset/restore hooks must not run during fallback launch")

    monkeypatch.setattr(kernel, "run", run)
    tuner = Autotuner(kernel, kernel.arg_names, configs, ["N"], None, None, pre_hook=unexpected_hook,
                      post_hook=unexpected_hook, do_bench=bench_once, cache_results=cache_results)
    tuner.cache[(64, )] = configs[0]
    assert tuner.run(descriptor, N=64) == "launched"
    assert hooks == attempts == [256, 128]
    assert tuner.best_config.kwargs["BLOCK_SIZE"] == 128
    assert tuner.cache[(64, )].kwargs["BLOCK_SIZE"] == 128

    attempts.clear()
    hooks.clear()
    assert tuner.run(descriptor, N=64) == "launched"
    assert hooks == attempts == [128]


@pytest.mark.parametrize("existing_cache", [False, True])
def test_fallback_updates_disk_cache(monkeypatch, fresh_triton_cache, existing_cache):
    autotuner_module = importlib.import_module("triton.runtime.autotuner")
    compiler_module = importlib.import_module("triton.compiler.compiler")
    monkeypatch.setattr(autotuner_module, "driver",
                        SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: None)))
    monkeypatch.setattr(compiler_module, "make_backend", lambda target: SimpleNamespace(hash=lambda: "cpu-test"))
    monkeypatch.setattr(autotuner_module, "triton_key", lambda: "cpu-test")
    monkeypatch.setattr(autotuner_module, "get_cache_invalidating_env_vars", lambda: {})

    @triton.jit
    def kernel(N, BLOCK_SIZE: tl.constexpr):
        pass

    attempts = []
    benchmarking = False

    def run(N, **meta):
        attempts.append(meta["BLOCK_SIZE"])
        if benchmarking or meta["BLOCK_SIZE"] == 256:
            raise BackendCompilerError("invalid block size")
        return "launched"

    def bench(kernel_call, quantiles):
        nonlocal benchmarking
        benchmarking = True
        try:
            return bench_once(kernel_call, quantiles)
        finally:
            benchmarking = False

    monkeypatch.setattr(kernel, "run", run)
    configs = [triton.Config({"BLOCK_SIZE": size}) for size in (256, 128)]
    tuner = Autotuner(kernel, kernel.arg_names, configs, ["N"], None, None, do_bench=bench, cache_results=True)
    cache, file_name = tuner._get_disk_cache((64, ), configs)
    if existing_cache:
        # Also repair a failing selection loaded from the previous cache format.
        cache.put(json.dumps({"key": [64], "configs_timings": [(config.__dict__, [1.0] * 3) for config in configs]}),
                  file_name, binary=False)
    assert tuner.run(64) == "launched"
    assert attempts == ([256, 128] if existing_cache else [256, 128, 256, 128])
    with open(cache.get_file(file_name)) as cache_file:
        cached_result = json.load(cache_file)
    assert cached_result["best_config"]["kwargs"]["BLOCK_SIZE"] == 128

    attempts.clear()
    assert tuner.run(64) == "launched"
    assert attempts == [128]

    def unexpected_bench(*args, **kwargs):
        pytest.fail("The successful fallback should be loaded from disk")

    # A fresh instance has no memory cache, just as in a new process.
    reloaded = Autotuner(kernel, kernel.arg_names, configs, ["N"], None, None, do_bench=unexpected_bench,
                         cache_results=True)
    attempts.clear()
    assert reloaded.run(64) == "launched"
    assert attempts == [128]
    assert reloaded.cache[(64, )].kwargs["BLOCK_SIZE"] == 128


@pytest.mark.parametrize("error", [BackendCompilerError("compile failure"), RuntimeError("kernel failure")])
def test_fallback_propagates_failure(monkeypatch, error):

    @triton.jit
    def kernel(N, BLOCK_SIZE: tl.constexpr):
        pass

    configs = [triton.Config({"BLOCK_SIZE": size}) for size in (256, 128)]
    attempts = []

    def run(N, **meta):
        attempts.append(meta["BLOCK_SIZE"])
        if meta["BLOCK_SIZE"] == 8:
            raise error
        raise BackendCompilerError("fallback compile failure")

    monkeypatch.setattr(kernel, "run", run)
    tuner = Autotuner(kernel, kernel.arg_names, configs, ["N"], None, None, do_bench=bench_once)
    failed_config = triton.Config({"BLOCK_SIZE": 8})
    tuner.cache[(64, )] = failed_config
    with pytest.raises(type(error)) as caught:
        tuner.run(64)
    assert caught.value is error
    assert attempts == ([8, 256, 128] if isinstance(error, BackendCompilerError) else [8])
    assert tuner.cache[(64, )] is failed_config


def test_fallback_does_not_retry_identical_config(monkeypatch):

    @triton.jit
    def kernel(BLOCK_SIZE: tl.constexpr):
        pass

    error = BackendCompilerError("compile failure")
    attempts = []

    def run(**meta):
        attempts.append(meta["BLOCK_SIZE"])
        raise error

    monkeypatch.setattr(kernel, "run", run)
    tuner = Autotuner(kernel, kernel.arg_names, [triton.Config({"BLOCK_SIZE": 128})], [], None, None)
    with pytest.raises(BackendCompilerError) as caught:
        tuner.run()
    assert caught.value is error
    assert attempts == [128]
    assert not tuner.cache


def test_aabs_skips_nonpositive_extent(device):
    """AABS must not fit a block size to a zero-valued scalar argument.

    The dependency analysis pairs a block-size constexpr with the extent of the
    `tl.load` it sizes. When that extent resolves to a kernel argument that is
    legitimately 0 (an `init_blocks`-style count, an empty tensor), fitting the
    block size shrinks it to 1, which can violate an invariant the kernel states
    itself and makes the winning config fail to compile at launch. Kernel and
    arguments below mirror the vLLM MiniMax-M3 indexer, where this aborted the
    first prefill.
    """
    src = torch.randn(4096, device=device)
    dst = torch.zeros(1, device=device)

    @triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8),
            triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4),
        ],
        key=["BLOCK_SIZE_T"],
        do_bench=bench_once,
    )
    @triton.jit
    def _kernel(src, dst, topk: tl.constexpr, init_blocks, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_T: tl.constexpr):
        tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
        off = tl.arange(0, BLOCK_SIZE_K)
        x = tl.load(src + off, mask=off < init_blocks, other=0.0)
        tl.store(dst, tl.max(x, axis=0))

    _kernel[(1, )](src, dst, 16, 0)
    assert dst.item() == 0.0
    # A successful fallback must not hide a regression in the extent guard.
    assert _kernel.fn.configs_timings
    assert all(config.kwargs["BLOCK_SIZE_K"] in (128, 2048) for config in _kernel.fn.configs_timings)
    assert all(all(timing < float("inf") for timing in timings) for timings in _kernel.fn.configs_timings.values())
