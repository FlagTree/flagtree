#
# This file implements the triton kernel driver interfaces where are used in
# triton/python/triton/compiler/compiler.py.
# For how the interface in driver class is used, see the implementation of the
# file above.
#
import hashlib
import tempfile
import os
import subprocess
import importlib.util
import shutil
import sysconfig
import atexit
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import GPUDriver
from triton.backends.compiler import GPUTarget


def extend_torch():
    import torch
    from torch.utils import cpp_extension, rename_privateuse1_backend, generate_methods_for_privateuse1_backend
    module = cpp_extension.load(
        name="txda",
        sources=[os.path.dirname(__file__) + "/txda_device.cpp"],
        #runtime include path
        extra_include_paths=[""],
        #runtime *.so path
        extra_ldflags=[""],
        extra_cflags=["-g"],
        verbose=True,
    )
    torch.utils.rename_privateuse1_backend("txda")
    torch._register_device_module("txda", module)
    generate_methods_for_privateuse1_backend(for_storage=True)


def _get_tx8_path(bin_name: str) -> str:
    path = os.getenv("TX8_HOME", "")
    if path == "":
        raise Exception("TX8_HOME is not set.")
    return os.path.join(path, bin_name)


dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [
    os.path.join(dirname, "include"),
    os.path.realpath(_get_tx8_path("include")),
    os.path.join(sysconfig.get_path('platlib'), "pybind11", "include"),
    os.path.join(sysconfig.get_path('platlib'), "torch", "include"),
    os.path.join(sysconfig.get_path('platlib'), "torch", "include", "torch", "csrc", "api", "include"),
    os.path.join(sysconfig.get_path('platlib'), "numpy", "_core", "include")
]
library_dirs = [
    os.path.join(dirname, "lib"),
    os.path.realpath(_get_tx8_path("lib")),
    os.path.join(sysconfig.get_path('platlib'), "torch", "lib")
]
libraries = ['tx8_runtime', 'torch', 'torch_cpu', 'torch_python', 'c10']


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_DUMP_PATH", "")
    if not path:
        return

    os.makedirs(path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        cc = clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-std=c++17", "-Wno-psabi", "-o", so]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd += [f"-Wl,-rpath,{dir}" for dir in library_dirs]
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


# Build a native ELF on the platform running this python script
def compile_native(src, name):
    fname = "native_" + name
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{fname}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{name}.cpp")
            with open(src_path, "w") as f:
                f.write(src)
                _dump_ir_if_needed([src_path])
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{fname}.so", binary=True)
                _dump_ir_if_needed([cache_path])

    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
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


def _extracted_type(ty):
    if isinstance(ty, tuple):
        val = ','.join(map(_extracted_type, ty))
        return f"[{val}]"
    if ty[0] == '*':
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    if isinstance(ty, tuple):
        val = ''.join(map(format_of, ty))
        return f"({val})"
    if ty[0] == '*':
        return "O"
    if ty in ("constexpr", "nvTmaDesc"):
        return "O"
    return {
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "L",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[_ty_to_cpp(ty)]


def make_launcher(constants, signature, kernel_name):
    # Basic declarations
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    args_format = ''.join([_format_of(ty) for ty in signature.values()])
    format = "iiiOOOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    # Parameters to pass to the kernel function
    kernel_parameters = ', '.join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"tx81_ptr{i}, &ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/arrayobject.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <string>
#include <filesystem>
#include "hrt_interface.h"
#include "hrt_common.h"

// The design of kernel argument buffer:
// The offset starts from the whole kernel buffer
// +------------------------------------------------------------------------+
// |  4 bytes  | 4 bytes | 4 bytes |  4 bytes |   8 bytes    |     8 bytes  |
// | No. kargs |  gridX  |  gridY  |   gridZ  | karg1 offset | karg2 offset |
// +------------------------------------------------------------------------+
// .......................... Metadata buffer................................
//
// +------------------------------------------------------------------------+
// | ...  |    8 bytes    |    n bytes    |  n bytes   |     |   n bytes    |
// | ...  | kargn offset  |  karg1 data   | karg2 data | ... |  kargn data  |
// +------------------------------------------------------------------------+
//                        ^                ^                 ^
//                        karg1 offset     karg2 offset      kargn offset
// ... Metadata buffer... | ............ kernel arg buffer ..................

enum DATA_TYPE {{
    SCALAR,
    POINT,
}};

// A kernel argument
struct KernelArg {{
    // The actual kernel argument: tensor or scalar
    union Data {{
        void* ptr;        // Pointer to the tensor data
        uint64_t scalar;  // Scalar data
    }} data;
    size_t size;  // The size of the kernel argument
    int data_type;

    KernelArg(void *ptr, size_t s) : size(s) {{
        data.ptr = ptr;
        data_type = POINT;
    }}

    KernelArg(uint64_t v, size_t s) : size(0) {{
        data.scalar = v;
        data_type = SCALAR;
    }}

}};

extern "C" {{
  // The kernel arguments includes:
  //  1. The actual kernel argument in arg_decls
  //  2. The group size: gridX, gridY, gridZ
  //  3  The thread id in each direction: x, y, z
  void {kernel_name}({arg_decls}, int, int, int, int, int, int);
}}

// Global device vector
static std::vector<TsmDevice*> g_tx81_devices;
static bool g_runtime_initialized = false;

// FIXME: Hardcoded path
std::string chip_out = "/tmp/chip_out/node0/";
std::string kernel_file = "/tmp/kernel.so";
std::string kernel_fun_name = "{kernel_name}";
uint32_t sharedMemBytes = 0;

typedef void* Stream_t;

static uint64_t get_phy_addr(uint64_t logic_addr) {{
    uint32_t card_id;
    uint64_t addr;
    uint64_t size;
    TsmMemGetInfo(logic_addr, card_id, addr, size);
    return addr;
}}


// Initialize Tx81 runtime
bool init_tx81_runtime() {{
    if (g_runtime_initialized) {{
        return true;  // Already initialized
    }}

    // Initialize the Tx81 runtime
    if (TsmInitRuntime() != RET_SUCCESS) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Tx81 runtime");
        return false;
    }}

    // Get device count
    uint32_t device_num = 0;
    if (TsmGetDeviceNum(device_num) != RET_SUCCESS || device_num == 0) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to get Tx81 device count or no devices found");
        TsmDeInitRuntime();
        return false;
    }}

    // FIXME: Hardcoded
    // Set up devices - for simplicity, we're using a 1x1 configuration
    uint32_t first_phy_id = 0;
    uint32_t card_x = 1;
    uint32_t card_y = 1;

    TsmDevice *dev = new TsmDevice();
    if (TsmSetDevice(&dev, 0, first_phy_id) != RET_SUCCESS) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to set Tx81 devices");
        TsmDeInitRuntime();
        return false;
    }}
    g_tx81_devices.push_back(dev);

    // FIXME: Hardcoded
    TsmModel *new_model = new TsmModel();
    // Create a vector of models
    std::vector<TsmModel *> kmodel_vec = {{new_model}};
    std::string option = "-O2";
    CompileOption compl_option = {{}};
    compl_option.comp_enable = 0; // Use prebuilt binary
    compl_option.chip_x = 1; //单卡
    compl_option.chip_y = 1;
    compl_option.check_enable = true;
    compl_option.enable_kcore_bin = 1;
    compl_option.enable_kcore_so = 1;
    new_model->case_dir = chip_out;

    for (TsmDevice * dev : g_tx81_devices) {{
        if (TsmCompileMultiGraph(dev, *new_model, option, compl_option) != RET_SUCCESS) {{
            for (uint32_t dev_index = 0; dev_index < g_tx81_devices.size(); ++dev_index) {{
                if (TsmResetDevice(g_tx81_devices[dev_index]) != RET_SUCCESS) {{
                    return false;
                }}
            }}
            return false;
        }}
    }}

    // Initialize all devices
    for (auto* dev : g_tx81_devices) {{
        if (TsmLaunch(dev, *new_model) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "[Chip id] TsmLaunch failed.");
            TsmReleaseDevice(dev);
            TsmResetDevice(dev);
            return false;
        }}

        if (TsmSetMonitorInfo(dev) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "[Chip id] TsmLaunch failed.");
            TsmReleaseDevice(dev);
            TsmResetDevice(dev);
            return false;
        }}
    }}

    delete new_model;
    g_runtime_initialized = true;

    return true;
}}

// Clean up Tx81 runtime resources
static PyObject* cleanup_tx81_runtime(PyObject* self, PyObject* args)  {{
    if (!g_runtime_initialized) {{
        Py_RETURN_NONE;
    }}

    for (auto* dev : g_tx81_devices) {{
        if (TsmSetTerminate(dev) != RET_SUCCESS) {{
            Py_RETURN_NONE;
        }}
        // Reset and release each device
        TsmReleaseDevice(dev);
        TsmResetDevice(dev);
        delete dev;
    }}
    g_tx81_devices.clear();
    TsmDeInitRuntime();
    g_runtime_initialized = false;
    Py_RETURN_NONE;
}}

TSM_RETCODE argsToDevMemArray(TsmDevice *dev, std::vector<KernelArg> &kargs,
    std::vector<uint64_t> &rtKargs, std::vector<uint64_t> &devAddrs) {{
    int count = 0;
    for (KernelArg& karg : kargs) {{
        if (karg.data_type == POINT) {{
            TsmDevicePtr dev_buffer;
            if (TsmDeviceMalloc(dev, dev_buffer, karg.size) != RET_SUCCESS) {{
                PyErr_SetString(PyExc_RuntimeError, "Failed to TsmDeviceMalloc");
                return RET_ERROR;
            }}

            if (TsmMemcpyH2D(dev_buffer, karg.data.ptr, karg.size) != RET_SUCCESS) {{
                PyErr_SetString(PyExc_RuntimeError, "Failed to TsmMemcpyH2D");
                return RET_ERROR;
            }}
            devAddrs.push_back(dev_buffer);
            // FIXME: rank
            rtKargs.push_back(1);
            rtKargs.push_back(get_phy_addr(dev_buffer));

            count++;
        }}
        else {{
            rtKargs.push_back(karg.data.scalar);
        }}
    }}
    return RET_SUCCESS;
}}

TSM_RETCODE devMemArrayToArgs(TsmDevice *dev, std::vector<KernelArg> &kargs,
        std::vector<uint64_t> &devAddrs) {{

    int count = 0;
    for (KernelArg& karg : kargs) {{
        if (karg.data_type == POINT) {{
            uint64_t dev_buffer = devAddrs[count++];
            if (TsmMemcpyD2H(karg.data.ptr, dev_buffer, karg.size) != RET_SUCCESS) {{
                PyErr_SetString(PyExc_RuntimeError, "Failed to TsmMemcpyH2D");
                return RET_ERROR;
            }}
        }}
    }}
    return RET_SUCCESS;
}}

TSM_RETCODE devMemFree(TsmDevice *dev, std::vector<uint64_t> &devAddrs) {{
    for (uint64_t dev_buffer : devAddrs) {{
        if (TsmDeviceFree(dev_buffer) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to TsmDeviceFree");
            return RET_ERROR;
        }}
    }}
    return RET_SUCCESS;
}}

TSM_RETCODE freeMemPerStep(uint32_t chip_id, TsmDevicePtr &bootpm_dev) {{
    if (bootpm_dev != 0) {{
        if (TsmDeviceFree(bootpm_dev) != RET_SUCCESS) {{
            return RET_ERROR;
        }}
            bootpm_dev = 0;
        }}
    return RET_SUCCESS;
}}

static void _launch(int gridX, int gridY, int gridZ, std::vector<KernelArg> kargs) {{
    std::vector<TsmDevice *> &devices = g_tx81_devices;

    if (gridX*gridY*gridZ <= 0) {{
        return;  // No work to do
    }}

     // TODO::mv
    uint64_t kernel_len = 0;
    uint8_t* kernel_ptr = read_file_data(kernel_file, kernel_len);
    if (kernel_ptr == nullptr) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to read kernel so");
        TsmDeInitRuntime();
        return;
    }}

    // prepare data/ load kernel/run/unload kernel/get out data/release memory
    for (uint32_t dev_index = 0; dev_index < devices.size(); ++dev_index) {{
        // Allocate the device memory for all kernel arguments
        std::vector<uint64_t> devAddrs;
        std::vector<uint64_t> rtKargs;

        if (argsToDevMemArray(devices[dev_index], kargs, rtKargs, devAddrs) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to argsToDevMemArray");
            TsmDeInitRuntime();
            return;
        }}

        rtKargs.push_back(gridX);
        rtKargs.push_back(gridY);
        rtKargs.push_back(gridZ);
        rtKargs.push_back(0);
        rtKargs.push_back(0);
        rtKargs.push_back(0);

        // TSM_RETCODE TsmKernelLaunch(TsmDevice *dev, const char *func_name, void *kernel_ptr, uint32_t kernel_len,
        // uint32_t grid_dim, uint32_t block_dim, void *args, uint32_t args_len);
        if (TsmKernelLaunch(devices[dev_index], kernel_fun_name.c_str(), (void*)kernel_ptr, kernel_len,
            gridX, 1, (void*)(&rtKargs[0]), rtKargs.size()*sizeof(uint64_t)) != RET_SUCCESS){{
            PyErr_SetString(PyExc_RuntimeError, "Failed to TsmKernelLaunch");
            TsmDeInitRuntime();
        }}
        if (devMemArrayToArgs(devices[dev_index], kargs, devAddrs) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to devMemArrayToArgs");
            TsmDeInitRuntime();
            return;
        }}

        // getchar();

        // TsmUnloadKernel(devices[dev_index], kmodel_vec);

        if (devMemFree(devices[dev_index], devAddrs) != RET_SUCCESS) {{
            return;
        }}
    }}
}}

// Structure to represent a device pointer
typedef struct _DevicePtrInfo {{
    void *dev_ptr;
    bool valid;
}} DevicePtrInfo;

static size_t getTensorStorageSize(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    return tensor.storage().nbytes();
}}

// Extract tensor raw ptr
static void* extractTensor(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    torch::Tensor contiguous_tensor = tensor.contiguous();
    return contiguous_tensor.data_ptr();
}}

static PyObject* init_runtime(PyObject* self, PyObject* args) {{
    const char* _chip_out;
    if (!PyArg_ParseTuple(args, "s", &_chip_out)) {{
        return NULL;
    }}
    chip_out = _chip_out;

    // Initialize Tx81 runtime during module import
    if (!init_tx81_runtime()) {{
        return NULL;
    }}

    return Py_None;
}}

// Python module launch function
static PyObject* launch(PyObject* self, PyObject* args) {{
    int gridX, gridY, gridZ;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;
    // FIXME: Extra 2 args:
    PyObject *dummy1 = NULL;
    PyObject *dummy2 = NULL;
    // Define the actual kernel arguments
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}

    // Init kernel arguments from python side
    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                          &kernel_metadata, &launch_metadata,
                          &launch_enter_hook, &launch_exit_hook,
                          &dummy1, &dummy2{args_list})) {{
        return NULL;
    }}

#if 0 // FIXME: kernel_metadata is not correctly inited
    // Extract metadata for consistency with other drivers
    int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
    if (!PyArg_ParseTuple(kernel_metadata, "iiiiii", &num_warps, &num_ctas,
        &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
        PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
        return NULL;
    }}

    // Call the enter hook if provided
    if (launch_enter_hook != Py_None) {{
        PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_enter_hook, hook_args);
        Py_DECREF(hook_args);
        if (!ret)
            return NULL;
    }}
#endif

    // Construct a data kernel arguments list data structure
    std::vector<KernelArg> kargs;
    //{' '.join([f"kargs.emplace_back(_arg{i}, PyObject_Size(_arg{i})*4);" if ty[0]=="*" else f"kargs.emplace_back(_arg{i}, sizeof(_arg{i}));" for i, ty in signature.items() if ty != "constexpr"])}
    {' '.join([f"kargs.emplace_back(extractTensor(_arg{i}), getTensorStorageSize(_arg{i}));"
               if ty[0]=="*" else f"kargs.emplace_back(_arg{i}, sizeof(_arg{i}));"
                  for i, ty in signature.items() if ty != "constexpr"])}

    // Launch the kernel
    _launch(gridX, gridY, gridZ, kargs);
    if (PyErr_Occurred()) {{
        return NULL;
    }}

    // Call the exit hook if provided
    if (launch_exit_hook != Py_None) {{
        PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, hook_args);
        Py_DECREF(hook_args);
        if (!ret)
            return NULL;
    }}

    // Return None to Python
    Py_INCREF(Py_None);
    return Py_None;
}}

// Python module method definitions
static PyMethodDef ModuleMethods[] = {{
    {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
    {{"init_runtime", init_runtime, METH_VARARGS, "Init runtime with chip_out dir"}},
    {{NULL, NULL, 0, NULL}} // sentinel
}};

// Python module definition
static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,
    \"__triton_launcher\",
    NULL, // documentation
    -1,   // size
    ModuleMethods
}};

static PyMethodDef cleanup_method = {{
    "cleanup_tx81_runtime",
    (PyCFunction)cleanup_tx81_runtime,
    METH_NOARGS,
    "Cleanup Tx81 runtime resources"
}};

// Python module initialization function
PyMODINIT_FUNC PyInit___triton_launcher(void) {{
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL) {{
        return NULL;
    }}

    PyModule_AddFunctions(m, ModuleMethods);

    // Register an atexit handler to cleanup Tx81 runtime
    PyObject* atexit_module = PyImport_ImportModule("atexit");
    if (atexit_module) {{
        PyObject* cleanup_func = PyCFunction_New(&cleanup_method, NULL);
        if (cleanup_func) {{
            PyObject* result = PyObject_CallMethod(atexit_module, "register", "O", cleanup_func);
            Py_XDECREF(result);
            Py_DECREF(cleanup_func);
        }}
        Py_DECREF(atexit_module);
    }}

    return m;
}}
"""


class TXDAUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(TXDAUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(os.path.join(dirname, "driver.cpp")).read_text()
        mod = compile_native(src, "tx81_utils")
        # # NOTE: The triton compiler.py framework requires these 2 interface.
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


# Launch cross compiled runtime program on controller
class TXDALauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        # Compiler runtime kernel launcher source code
        launcher_src = make_launcher(constants, signature, src.fn.__name__)
        mod = compile_native(launcher_src, "__triton_launcher")
        self.launch = mod.launch
        chip_out = os.path.join(_get_tx8_path("chip_out"), "node0")
        chip_out = chip_out + os.sep
        mod.init_runtime(chip_out)

    def __call__(self, *args, **kwargs):
        # args: 0: gridX, 1: gridY, 2: gridZ,
        #       3: kernel_metadata?, 4: launch_metadata?,
        #       5: a tuple(0, 0, False, 1, 1, 1, 'add_kernel'), # this is probably kernel metadata
        #       6: None, 7: None, 8: None,
        #       9~N: Actual triton kernel args.
        self.launch(*args, **kwargs)


class TXDADriver(GPUDriver):

    def __init__(self):
        super().__init__()
        extend_torch()
        self.utils = TXDAUtils()
        self.launcher_cls = TXDALauncher
        import torch
        # Needs to overwrite GPUDriver base methods
        self.get_current_stream = torch.txda.current_stream
        self.get_current_device = torch.txda.current_device
        self.set_current_device = torch.txda.set_device
        atexit.register(torch.txda.cleanup_device)

    @staticmethod
    def is_active():
        try:
            #import torch
            #return torch.txda.is_available()
            return True
        except ImportError:
            return False

    def get_current_target(self):
        capability = 1
        warp_size = 16
        return GPUTarget("txda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        # torch.txda.init_device()
        return torch.device("txda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.txda
