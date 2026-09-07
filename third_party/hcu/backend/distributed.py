"""HCU flagcx backend configuration.

Provides FlagcxRuntimeConfig and FlagCXBackendAdapter for HCU GPUs.
"""

from triton.experimental._flagcx_config import (
    Distributed,  # noqa: F401 # re-export
    FlagCXBackendAdapter,
    FlagcxRuntimeConfig,
)


class HcuFlagcxRuntimeConfig(FlagcxRuntimeConfig):
    """HCU-specific flagcx runtime configuration."""

    def _is_available_impl(self):
        from .flagcx_wrapper import FLAGCXLibrary  # noqa: F401

        return True


class HcuFlagCXBackendAdapter(FlagCXBackendAdapter):
    """HCU backend adapter for flagcx."""

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def distributed_backend_name(self) -> str:
        return "nccl"

    @property
    def allocator_class(self):
        import torch.cuda.memory

        return torch.cuda.memory.CUDAPluggableAllocator


flagcx_rt_conf = HcuFlagcxRuntimeConfig()
backend_adapter = HcuFlagCXBackendAdapter()
