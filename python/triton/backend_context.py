_backend_registry = {}

def register_backend(platform: str, backend_obj):
    _backend_registry[platform] = backend_obj

def get_backend(platform: str):
    return _backend_registry.get(platform)
