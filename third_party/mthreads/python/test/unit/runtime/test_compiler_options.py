import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
from triton._C import libtriton
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.experimental.tle.language.gpu.mthreads import MTHREADS_GEMM_COMPILER_OPTIONS_VERSION

if not hasattr(libtriton, "mthreads"):
    pytest.skip("mthreads backend not built in libtriton", allow_module_level=True)


def _get_mthreads_backend():
    if "mthreads" not in backends:
        pytest.skip("mthreads backend not discovered")
    return backends["mthreads"].compiler(GPUTarget("musa", "ph1", 32))


def test_mthreads_can_disable_max_ilp_scheduler():
    from triton.backends.mthreads.compiler import _llc_extra_options

    assert MTHREADS_GEMM_COMPILER_OPTIONS_VERSION >= 1
    backend = _get_mthreads_backend()
    enabled = backend.parse_options({"enable_backend_opt": True})
    assert enabled.disable_max_ilp_scheduler is False
    enabled_args = _llc_extra_options({"uses_sqmma": True}, enabled, capability=31)
    assert "-misched=mtgpu-max-ilp" in enabled_args

    disabled = backend.parse_options(
        {
            "enable_backend_opt": True,
            "disable_max_ilp_scheduler": True,
        }
    )
    disabled_args = _llc_extra_options({"uses_sqmma": True}, disabled, capability=31)
    assert "-misched=mtgpu-max-ilp" not in disabled_args
    assert "-mtgpu-opt-level=1" in disabled_args
    assert disabled.hash() != enabled.hash()


def test_mthreads_disable_max_ilp_scheduler_does_not_change_non_sqmma_options():
    from triton.backends.mthreads.compiler import _llc_extra_options

    backend = _get_mthreads_backend()
    enabled = backend.parse_options({"enable_backend_opt": True})
    disabled = backend.parse_options(
        {
            "enable_backend_opt": True,
            "disable_max_ilp_scheduler": True,
        }
    )

    enabled_args = _llc_extra_options({"uses_sqmma": False}, enabled, capability=31)
    disabled_args = _llc_extra_options({"uses_sqmma": False}, disabled, capability=31)
    assert disabled_args == enabled_args


def test_mthreads_can_disable_sqmma_m_split_preference():
    assert MTHREADS_GEMM_COMPILER_OPTIONS_VERSION >= 2
    backend = _get_mthreads_backend()
    enabled = backend.parse_options({})
    disabled = backend.parse_options(
        {"disable_sqmma_m_split_preference": True}
    )

    assert enabled.disable_sqmma_m_split_preference is False
    assert disabled.disable_sqmma_m_split_preference is True
    assert disabled.hash() != enabled.hash()
