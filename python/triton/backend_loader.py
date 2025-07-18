import importlib
from triton.backend_context import get_backend

def load_backend(platform: str):
    try:
        importlib.import_module(f"triton.backends.{platform}.backend_impl")
    except ImportError:
        raise RuntimeError(f"Backend for platform '{platform}' not found")

    backend = get_backend(platform)
    if backend is None:
        raise RuntimeError(f"Backend '{platform}' failed to register itself")
    return backend
