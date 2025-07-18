from . import libdevice

from .utils import (globaltimer, num_threads, num_warps, smid, convert_custom_float8_sm70, convert_custom_float8_sm80)

# TODO: 0627
import os
from triton.backend_loader import get_backend

# 默认平台从环境变量或配置传入
PLATFORM = os.getenv("FLAGTREE_PLATFORM", "iluvatar")
backend = get_backend(PLATFORM)

__all__ = backend.get_language__all__()
