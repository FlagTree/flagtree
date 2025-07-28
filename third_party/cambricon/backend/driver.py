import functools
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

dirname = os.path.dirname(os.path.realpath(__file__))
libraries = ['cndrv', 'cnrt']


@functools.lru_cache()
def default_neuware_dir():
    default_dir = "/usr/local/neuware/"
    return os.getenv("NEUWARE_HOME", default=default_dir)


@functools.lru_cache()
def library_dirs():
    return [os.path.join(default_neuware_dir(), "lib64")]


@functools.lru_cache()
def include_dirs():
    return [os.path.join(default_neuware_dir(), "include")]


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs(), include_dirs(), libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class BangUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(BangUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "bang_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.is_linear_pointer = mod.is_linear_pointer


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty, ptr_ty="CNaddr"):
    if ty[0] == '*':
        return ptr_ty
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "l",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    def serialize_idx(signature):
        flat_list = []
        mapping = {}
        idx = 0

        def traverse(sub_lst, path=()):
            nonlocal idx
            for i, item in enumerate(sub_lst):
                new_path = path + (i, )
                if isinstance(item, tuple):
                    traverse(item, new_path)
                else:
                    flat_list.append(item)
                    mapping[new_path] = idx
                    idx += 1

        traverse(signature)
        return mapping

    idx_map = serialize_idx(signature.values())
    constants = {idx_map[idx]: value for idx, value in constants.items() if idx in idx_map}
    remove_idx = {idx_map[(i, )] for i, ty in enumerate(signature.values()) if ty == "constexpr"}

    args_format = ''.join([format_of(ty) for ty in signature.values() if ty != "constexpr"])
    format = "iiiKKOOOO" + args_format
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    non_const_args = {i: ty for i, ty in signature.items() if i not in remove_idx}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in non_const_args.items()) if len(non_const_args) > 0 else ''

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items()
                          if ty != "constexpr" and i not in constants)
    # generate glue code
    params = [i for i, ty in signature.items() if ty != "constexpr" and i not in constants]

    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr" and i not in constants:
            internal_args_list.append(f"_arg{i}")

    src = f"""
#include \"cn_api.h\"
#include \"cnrt.h\"
#include \"cnrtc.h\"

#include <stdbool.h>
#include <stdio.h>
#include <Python.h>

static inline void cnAssert(CNresult code, const char *file, int line) {{
  if (code != CN_SUCCESS) {{
    const char *prefix = "Triton Error [MLU]: ";
    const char *str;
    cnGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }}
}}

#define CN_CHECK(ans) {{ cnAssert((ans), __FILE__, __LINE__); }}

static void _launch(unsigned int dimx, unsigned int dimy, unsigned int dimz, KernelClass func_type, CNqueue stream, CNkernel function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};

  if(dimx*dimy*dimz > 0) {{
    CN_CHECK(cnInvokeKernel(function, dimx, dimy, dimz, func_type, 0, stream, params, NULL));
  }}
}}

typedef struct _DevicePtrInfo {{
    uint64_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    cnrtPointerAttributes_t attributes;
    cnrtRet_t status = cnrtPointerGetAttributes(&attributes, (void*)ptr_info.dev_ptr);
    if (status != cnrtSuccess) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    attributes.devicePointer = (void*)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_warps;
  int promote_shared;
  if (!PyArg_ParseTuple(kernel_metadata, \"ip\", &num_warps, &promote_shared)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  gridX *= num_warps;

  int ordinal = -1;
  cnrtGetDevice(&ordinal);
  cnrtDeviceProp_t prop;

  cnrtGetDeviceProperties(&prop, ordinal);
  int cluster_cnt = prop.clusterCount;
  int core_num_per_cluster = prop.McorePerCluster;
  int total_cores = cluster_cnt * core_num_per_cluster;

  uint64_t func_type = ((num_warps == 1) && (gridX % core_num_per_cluster == 0) && (promote_shared == 1 || gridX * gridY * gridZ <= total_cores)) ? core_num_per_cluster : num_warps;

  if (launch_enter_hook != Py_None) {{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, (KernelClass)func_type, (CNqueue)_stream, (CNkernel)_function{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;

  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}

"""
    return src


class BangLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants | src.attrs.get_constants(), signature, ids)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class BangDriver(DriverBase):

    def __init__(self):
        self.utils = BangUtils()  # TODO: make static
        self.launcher_cls = BangLauncher
        import torch
        import torch_mlu
        self.get_current_device = torch.mlu.current_device
        self.set_current_device = torch.mlu.set_device
        self.get_current_stream = lambda idx: torch.mlu.current_stream(idx).mlu_stream
        self.is_linear_pointer = lambda ptr, device: self.utils.is_linear_pointer(ptr, device)
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.utils.get_device_properties(device).get('isa_version')
        # As compile func in compiler.py just support GPUTarget, and this type
        # can also represent MLU information, we will temporarily use GPUTarget here.
        return GPUTarget("mlu", capability, 0)

    def get_device_interface(self):
        import torch
        return torch.mlu

    def launch_as_union_task(self, device, grid):
        import math
        cluster_num = self.utils.get_device_properties(device).get('cluster_num')
        core_num_per_cluster = self.utils.get_device_properties(device).get('core_num_per_cluster')
        total_cores = cluster_num * core_num_per_cluster
        return grid[0] % core_num_per_cluster == 0

    @staticmethod
    def is_active():
        import torch
        try:
            if torch.mlu:
                return True
        except Exception as e:
            import torch_mlu
            return True
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='mlu')
