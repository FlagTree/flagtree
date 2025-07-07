# TODO: 0624
import importlib

_backend_cache = {}

def get_backend(platform: str):
    if platform in _backend_cache:
        return _backend_cache[platform]

    try:
        module = importlib.import_module(
            f"triton.backends.{platform}.backend_impl"
        )
        backend = module.get_backend()
        _backend_cache[platform] = backend
        return backend
    except ImportError as e:
        raise RuntimeError(f"Backend for platform '{platform}' not found") from e

