_backend_registry = {}

def register_backend(platform: str, backend_obj):
    _backend_registry[platform] = backend_obj

def get_backend():
    # TODO
    platform = "iluvatar"
    return _backend_registry.get(platform)
