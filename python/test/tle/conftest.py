import pytest

from triton.experimental.tle import (
    is_primitive_supported,
    primitive_name,
)

_MARKER = "require_tle"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "require_tle(*primitives): run only when the active backend supports every named TLE primitive",
    )


def _required_primitives(item):
    required = []
    for marker in item.iter_markers(name=_MARKER):
        if marker.kwargs:
            raise pytest.UsageError("require_tle accepts primitive names as positional arguments only")
        if not marker.args:
            raise pytest.UsageError("require_tle requires at least one TLE primitive name")
        required.extend(primitive_name(primitive) for primitive in marker.args)
    return frozenset(required)


def pytest_runtest_setup(item):
    from triton._flagtree_backend import FLAGTREE_BACKEND

    required = _required_primitives(item)
    if not required:
        return
    missing = sorted(primitive for primitive in required if not is_primitive_supported(FLAGTREE_BACKEND, primitive))
    if missing:
        pytest.skip(f"backend {FLAGTREE_BACKEND!r} does not support required TLE primitive(s): {', '.join(missing)}")
