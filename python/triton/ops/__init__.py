# from .conv import _conv, conv
from . import blocksparse
from .cross_entropy import _cross_entropy, cross_entropy
from .flash_attention import attention
from .matmul import _matmul, get_higher_dtype, matmul

# TODO: 0627
import os
from triton.backend_loader import get_backend

# 默认平台从环境变量或配置传入
PLATFORM = os.getenv("FLAGTREE_PLATFORM", "iluvatar")
backend = get_backend(PLATFORM)

__all__ = backend.get_ops__all__()
