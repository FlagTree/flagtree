from pathlib import Path
import tempfile
import os
import subprocess
import sysconfig
from typing import Optional
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.utils import _build_npu_ext, _check_cxx11_abi


class NPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "npu_utils.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "npu_utils.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_npu_ext("npu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device):
        fnname, mix_mode = name.split()
        return self.npu_utils_mod.load_kernel_binary(fnname, kernel, shared, device, mix_mode)

    @functools.lru_cache()
    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_aicore_num()


class NPULauncher(object):

    def __init__(self, src, metadata):
        debug_mode = metadata.debug
        workspace_size = int(metadata.workspace_size) \
                              if hasattr(metadata, 'workspace_size') else -1
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        wrapper_src = generate_npu_wrapper_src(constants, signature, \
                                               workspace_size)
        so_launcher_path = make_npu_launcher_stub(wrapper_src, debug_mode)
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        profiler_registered = self.launch(*args, **kwargs)
        import triton
        triton.backends.ascend.utils.TRITON_PROFILER_REGISTERED = True if profiler_registered == 1 else False


class NPUDriver(DriverBase):

    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    @classmethod
    def is_active(cls):

        def test_npucompiler():
            from triton.backends.ascend.utils import _get_bisheng_path
            npucompiler = _get_bisheng_path()
            targets = subprocess.check_output([npucompiler, "-print-targets"]).decode().strip().split()
            return "hiipu64" in targets

        try:
            return test_npucompiler()
        except Exception as e_npucompiler:
            import warnings
            red = "\x1b[31;20m"
            reset = "\x1b[0m"
            warnings.warn(red + str(e_npucompiler) + reset)
            return False

    def get_current_target(self):
        backend = "npu"
        arch = self.utils.get_arch()
        warp_size = 0
        return GPUTarget(backend, arch, warp_size)

    def get_current_device(self):
        """
        Get current device
        """
        import torch
        import torch_npu
        return torch.npu.current_device()

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        import torch
        import torch_npu
        return torch.npu.set_device(device)

    def get_current_stream(self, device: Optional[int] = None) -> int:
        """
        Get stream for current device
        """
        # According to torch_npu, the content of a torch.npu.Stream is essentilly an rtStream_t
        # TODO: use CANN API instead of torchnpu
        import torch
        import torch_npu
        if device is None:
            device = self.get_current_device()
        return torch.npu.current_stream(device).npu_stream

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.npu

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 192 * 1024 * 1024
        return torch.empty(cache_size // 4, dtype=torch.int, device='npu')


def make_npu_launcher_stub(src, debug=False):
    """
    Generate the launcher stub to launch the kernel
    """
    # try to get cached file
    so_cache_key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    # append the cxx11_abi value to the launcher name to avoid
    # linking to a launcher with wrong cxx11_abi.
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(src, f"{name}.cxx", binary=False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    with tempfile.TemporaryDirectory() as tmpdir:
        if debug:
            so_cache_manager.put(src, f"{name}.cxx", binary=False)
        src_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build_npu_ext(name, src_path, tmpdir, kernel_launcher="torch")
        if debug:
            with open(so, "rb") as f:
                return dump_manager.put(f.read(), so_name, binary=True)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)


# the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
def generate_npu_wrapper_src(constants, signature, workspace_size):
    import os

    def _ty_to_cpp(ty):
        if ty[0] == '*':
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *functon;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])

    grid_info = {'X': 'i32', 'Y': 'i32', 'Z': 'i32'}

    enable_device_print = os.getenv("TRITON_DEVICE_PRINT", 'false').lower() in ('true', '1')

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <sys/syscall.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "experiment/runtime/runtime/rt.h"
{'#include "device_print.h"' if enable_device_print else ''}

extern "C" {{

  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {{
    if ((CtrlData == nullptr) || (DataLen == 0U)) {{
      return 1;
    }}

    if (CtrlType == 1) {{
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {{  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }}
    }}
    return 0;
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
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
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, int *profilerRegistered, {arg_decls}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  if (!(*profilerRegistered)) {{
    MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
    *profilerRegistered = 1;
  }}
  auto launch_call = [=]() {{
    uint32_t blockNum = gridX * gridY * gridZ;
    {'TTAscDebug::DebugTunnelData *DTData = TTAscDebug::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
      return ret;
    }}
    // stub argument for workspace
    void *workspace_addr = NULL;
    {f'''
    uint16_t ModuleId = 0;
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    ret = rtMalloc(reinterpret_cast<void **>(&workspace_addr),
                   totalWorkSpaceSize, RT_MEMORY_HBM, ModuleId);
    if (ret != RT_ERROR_NONE) {{
      return ret;
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants)}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
      static_cast<void*>(ffts_addr),
      static_cast<void*>(workspace_addr),
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants)},
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
      {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opName = 0;
    unsigned int threadId = 0;
    char* kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL0 || __MsprofFlagL1) {{
    {{
      beginTime = MsprofSysCycleTime();
    }}
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
    {'TTAscDebug::Close(DTData, stream);' if enable_device_print else ''}
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL0 || __MsprofFlagL1) {{
    {{
      endTime = MsprofSysCycleTime();
      opName = MsprofGetHashId(kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.level = 10000;             //MSPROF_REPORT_NODE_LEVEL
      info.type = 5;                  //MSPROF_REPORT_NODE_LAUNCH_TYPE
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opName;
      MsprofReportApi(0, &info);
    }}
    // FIXME: to avoid bug in msprof, currently we disable these checks
    // if (__MsprofFlagL1) {{
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.level = 10000;             //MSPROF_REPORT_NODE_LEVEL
      nodeBasicInfo.type = 0;                  //MSPROF_REPORT_NODE_BASIC_INFO_TYPE
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opName;
      nodeBasicInfo.data.nodeBasicInfo.taskType = 0; //MSPROF_GE_TASK_TYPE_AI_CORE
      nodeBasicInfo.data.nodeBasicInfo.opType = opName;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = gridX;
      MsprofReportCompactInfo(0, &nodeBasicInfo, sizeof(MsprofCompactInfo));
    }}
    return ret;
  }};
  at_npu::native::OpCommand cmd;
  cmd.Name(name.c_str())
    .SetCustomHandler(launch_call)
    .Run();
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  PyObject *profilerRegisteredObj = PyDict_GetItemString(packedMetadata, "profiler_registered");
  int profilerRegistered = PyObject_IsTrue(profilerRegisteredObj);
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0]=="*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, &profilerRegistered, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items())});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  return Py_BuildValue("I", profilerRegistered);
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
