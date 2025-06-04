import sys
import os
from importlib.util import spec_from_file_location, module_from_spec

triton_root = os.path.dirname(__file__)
if triton_root not in sys.path:
    sys.path.append(triton_root)
triton_patch_init_path = os.path.join(triton_root, "triton_patch/__init__.py")
spec = spec_from_file_location("triton_patch", triton_patch_init_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
