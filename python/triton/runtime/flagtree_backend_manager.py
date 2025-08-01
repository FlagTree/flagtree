import importlib

_backend_registry = {}


def register_backend(vendor_name: str, backend_obj):
    _backend_registry[vendor_name] = backend_obj


def get_backend():
    from .driver import driver
    vendor_name = driver.active.get_vendor_name()
    importlib.import_module(f"triton.backends.{vendor_name}.specialized_impl")
    return _backend_registry.get(vendor_name)
