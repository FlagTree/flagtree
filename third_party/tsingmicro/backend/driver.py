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
from pathlib import Path
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import GPUDriver
from triton.backends.compiler import GPUTarget

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [
    os.path.join(dirname, "include"),
    os.path.join(sysconfig.get_path('platlib'), "pybind11", "include"),
    os.path.join(sysconfig.get_path('platlib'), "torch", "include"),
    os.path.join(sysconfig.get_path('platlib'), "torch", "include", "torch", "csrc", "api", "include"),
    os.path.join(sysconfig.get_path('platlib'), "numpy", "_core", "include")
]
library_dirs = [os.path.join(dirname, "lib"), os.path.join(sysconfig.get_path('platlib'), "torch", "lib")]
libraries = ['tx8_runtime', 'torch', 'torch_cpu', 'torch_python', 'c10']


# Path configuration for cross compilation
def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _get_libc_root() -> str:
    path = os.getenv("LIB_C_ROOT", "")
    if path == "":
        raise Exception("LIB_C_ROOT is not set.")
    return path


def _get_vendor_runtime_path() -> str:
    path = os.getenv("LIB_VENDOR_RUNTIME_PATH", "")
    if path == "":
        raise Exception("LIB_VENDOR_RUNTIME_PATH is not set.")
    return path


def _dump_ir_if_needed(files):
    path = os.getenv("ZTC_DUMP_PATH", "")
    if not path:
        return

    os.makedirs(path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


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


# Build a accelerator controller ELF
def compile_accelerator(src, name, ext):
    name = "npu_" + name
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    libc_inc = os.path.join(_get_libc_root(), "riscv64-unknown-elf", "include")
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{name}.{ext}")
            # FIXME: Hardcoded path
            #dst_path = os.path.join(tmpdir, "wrapper.so")
            dst_path = "/tmp/wrapper.o"
            with open(src_path, "w") as f:
                f.write(src)
            _dump_ir_if_needed([src_path])
            clang_path = _get_llvm_bin_path("clang")
            # Compile
            subprocess.check_call([
                clang_path, src_path, "-O2", "-c", "-fPIC", f"-I{libc_inc}", "--target=riscv64-unknown-elf",
                "-march=rv64imafdc", "-o", dst_path
            ])

        with tempfile.TemporaryDirectory() as tmpdir:
            # FIXME: Hardcoded path
            #dst_path = os.path.join(tmpdir, f"{name}.so")
            dst_path = "/tmp/kernel.so"
            libc_lib = os.path.join(_get_libc_root(), "riscv64-unknown-elf", "lib", "rv64imafdc", "lp64d")
            libcrt_lib = os.path.join(_get_libc_root(), "lib", "gcc", "riscv64-unknown-elf", "15.0.0", "rv64imafdc",
                                      "lp64d")
            libvr_path = _get_vendor_runtime_path()
            clang_path = _get_llvm_bin_path("clang")
            # Link wrapper, kernel with Tx81 crt and intrinsics(libkcorert.a)
            subprocess.check_call([
                clang_path, "-nostdlib",
                # FIXME: Hardcoded path
                "/tmp/wrapper.o", "/tmp/kernel.o", "-O2", "--target=riscv64-unknown-elf", "-march=rv64imafdc", "-fPIC",
                # "-shared",  # ELF toolchain doesn't support -shared
                f"-L{libvr_path}", f"-L{libc_lib}", f"-L{libcrt_lib}",
                # Allow libkcorert symbol overwrite libc symbols, libkcorert
                # should be specified before libc
                "-Wl,--allow-multiple-definition", "-lvr",  # Wrapper API of Tx81 intrinsic
                "-lkcorert",  # Tx81 intrinsic API
                "-lc", "-lm", "-lgcc", "-T", f"{libvr_path}/gcc_tx8_smarth.ld", "-o", dst_path
            ])

            _dump_ir_if_needed([dst_path])
            with open(dst_path, 'rb') as f:
                so = f.read()
            return so


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
    if ty[0] == '*':
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
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
    }[ty]


# This function makes a single kernel invoker which wraps all the input args into
# a single input buffer.
def make_kernel_wrapper_v2(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    return f"""
#include <stdbool.h>
#include <stdint.h>

// Triton kernel forward declaration, the last 6 arguments are: gridXYZ and xyz
// Using -convert-func-to-llvm=use-bare-ptr-memref-call-conv=true.
void {kernel_name}({arg_decls}, int, int, int, int, int, int);

// Kernel entry point
// NOTE: Assuming the triton kernel can only take 2 kind of arguments:
//   1. 8 bytes scalar
//   2. Tensor buffer (8 bytes memory address)
//
// The input buffer has the following format:
// +--------------------------------------------------------------------------+
// |  4 bytes  | 4 bytes | 4 bytes |  4 bytes | 4 bytes | 4 bytes |   8 bytes |
// |    gridX  |  gridY  |  gridZ  |    x     |    y    |    z    |    karg1  |
// +--------------------------------------------------------------------------+
// |  8 bytes  |   ...   | 8 bytes |
// |   karg2   |   ...   |  kargn  |
// +-------------------------------+
void __{kernel_name}(void *args) {{
    void* basePtr = args;

    // Extract the kernel arguments from kernel buffer
    int gridX = *((int*)basePtr);
    int gridY = *((int*)basePtr+1);
    int gridZ = *((int*)basePtr+2);
    int x = *((int*)basePtr+3);
    int y = *((int*)basePtr+4);
    int z = *((int*)basePtr+5);
    void* krnArgOffsets = (void*) ((int*)basePtr + 6);

    if (gridX*gridY*gridZ <= 0)
        return;

    // Invoke the actual kernel.
    {kernel_name}({', '.join([f"(void*) (((uint64_t*)krnArgOffsets)[{i}])"
                              if ty[0] == "*" else
                              f"*({_ty_to_cpp(ty)}*)(((uint64_t*)krnArgOffsets)[{i}])"
                              for i, ty in signature.items()])},
                              gridX, gridY, gridZ, x, y, z);
}}
"""


def make_kernel_wrapper(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

// Tx81 target framwork related definition
typedef struct BootParamHead
{{
    uint32_t MaxLen;
    uint32_t LdmemLen;
    uint32_t InputNum;
    uint32_t OutputNum;
    uint32_t ParamNum;
    uint32_t reserved;
    uint64_t CacheMemLen;
    uint64_t CacheMemAttr;
    uint32_t Datalen;
    uint32_t reserved1;
    uint64_t DataAddr;
}} D_BootParamHead;

// Tx81 target framwork related definition
typedef struct BootParamDyninfo
{{
    uint64_t addr; // device
    uint64_t size;
    uint32_t dtype;
    uint32_t dim;
    uint32_t shape[6];
}} D_BootParamDyninfo;

// Triton kernel forward declaration, the last 6 arguments are: gridXYZ and xyz
void {kernel_name}({arg_decls}, int, int, int, int, int, int);

// Get the entry point of kernel arg buffer
void* getKernelArgBuffer(void *args) {{
    // Always use the first BootParam to carry the address points to kernel
    // arguments buffer
    D_BootParamHead *head = (D_BootParamHead *)args;
    assert(head->InputNum == 1);
    // Decode the first parameter from BootParam as the kernel buffer info.
    D_BootParamDyninfo* kernelBuffer = (D_BootParamDyninfo *)((char *)args +
        sizeof(D_BootParamHead));
    // Kernel buffer address on device DDR
    return (void*) kernelBuffer->addr;
}}

// Kernel wrapper
void task(void *krnArgBuf, void *krnArgOffsets,
          int gridX, int gridY, int gridZ, int x, int y, int z) {{

    // Invoke the actual kernel by passing in the triton kernel arguments stored
    // on device DDR and the other arguments which generated by compiler.
    {kernel_name}({', '.join([f"(void*) (krnArgBuf + ((uint64_t*)krnArgOffsets)[{i}])"
                              if ty[0] == "*" else
                              f"*({_ty_to_cpp(ty)}*)(krnArgBuf + ((uint64_t*)krnArgOffsets)[{i}])"
                              for i, ty in signature.items()])},
                              gridX, gridY, gridZ, x, y, z);
}}

// Kernel entry point, name is aligned that specified to TsmLoadKernel
void __kernel_entry(void *args) {{
    void* basePtr = getKernelArgBuffer(args);

    // Extract the kernel arguments from kernel buffer
    int krnArgCount = *(int*)basePtr;
    int gridX = *((int*)basePtr+1);
    int gridY = *((int*)basePtr+2);
    int gridZ = *((int*)basePtr+3);
    void* krnArgOffsets = (void*) ((int*)basePtr + 4);
    void* krnArgBuf = krnArgOffsets + krnArgCount * sizeof(uint64_t*);

    if (gridX*gridY*gridZ <= 0)
        return;

    // Cast "function" to the real function type.
    for(int x = 0; x < gridX; x++) {{
        for(int y = 0; y < gridY; y++) {{
            for(int z = 0; z < gridZ; z++) {{
                task (krnArgBuf, krnArgOffsets, gridX, gridY, gridZ, x, y, z);
            }}
        }}
    }}
}}
"""


def make_launcher(constants, signature, kernel_name):
    # Basic declarations
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = ''.join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    # Parameters to pass to the kernel function
    kernel_parameters = ', '.join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"tx81_ptr{i}, &ptr_arg{i}"
        for i, ty in signature.items()
        if i not in constants)
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
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <string>
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


// A kernel argument
struct KernelArg {{
    // The actual kernel argument: tensor or scalar
    union Data {{
        void* ptr;        // Pointer to the tensor data
        uint64_t scalar;  // Scalar data
    }} data;
    size_t size;  // The size of the kernel argument

    KernelArg(void *ptr, size_t s) : size(s) {{
        data.ptr = ptr;
    }}

    KernelArg(uint64_t v, size_t s) : size(s) {{
        data.scalar = v;
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

    // Set up devices - for simplicity, we're using a 1x1 configuration
    uint32_t first_phy_id = 0;
    uint32_t card_x = 1;
    uint32_t card_y = 1;

    if (TsmSetDevice(first_phy_id, card_x, card_y, g_tx81_devices) != RET_SUCCESS) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to set Tx81 devices");
        TsmDeInitRuntime();
        return false;
    }}

    // Initialize all devices
    for (auto* dev : g_tx81_devices) {{
        if (TsmInitDevice(dev) != RET_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Tx81 device");
            TsmDeInitRuntime();
            return false;
        }}
    }}

    g_runtime_initialized = true;
    return true;
}}

// Clean up Tx81 runtime resources
void cleanup_tx81_runtime() {{
    if (!g_runtime_initialized) {{
        return;
    }}

    for (auto* dev : g_tx81_devices) {{
        // Reset and release each device
        TsmResetDevice(dev);
        TsmReleaseDevice(dev);
    }}

    g_tx81_devices.clear();
    TsmDeInitRuntime();
    g_runtime_initialized = false;
}}


static void prepare_input(std::vector<TsmDevice *> devices, uint32_t dev_index,
    std::shared_ptr<chip_common_info_t> chip_info) {{
  for (uint32_t i = 0; i < chip_info->input_num; ++i) {{
    chip_info->input_dev_addr.push_back(0);
    if (TsmDeviceMalloc(devices[dev_index], chip_info->input_dev_addr[i],
        chip_info->input_size[i]) != RET_SUCCESS) {{
      printf("[Chip id %u] Input%d, DeviceMalloc failed!\\n", devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }}

    if (TsmMemcpyH2D((TsmDevicePtr)chip_info->input_dev_addr[i],
                     (void*) chip_info->input_host_addr[i],
                     chip_info->input_size[i]) != RET_SUCCESS) {{
      printf("[Chip id %u] Input%d, MemcpyH2D failed!\\n", devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }}
  }}
}}

static void prepare_output(std::vector<TsmDevice *> devices, uint32_t dev_index,
  std::shared_ptr<chip_common_info_t> chip_info) {{
  for (size_t i = 0; i < chip_info->output_num; ++i) {{
    chip_info->output_dev_addr.push_back(0);
    printf("[Chip id %u] output[%lu] data(size: %lu)\\n",
           devices[dev_index]->chip_id, i, chip_info->output_size[i]);

    if (TsmDeviceMalloc(devices[dev_index], chip_info->output_dev_addr[i],
        chip_info->output_size[i]) != RET_SUCCESS) {{
      printf("[Chip id %u] output[%lu], DeviceMalloc failed!\\n",
             devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }}
  }}
}}

TSM_RETCODE kernel_result_process(std::vector<TsmDevice *> devices, uint32_t dev_index,
              std::shared_ptr<HrtBootParam> hostboot,
              std::shared_ptr<chip_common_info_t> chip_info,
              TsmDevicePtr bootpm_dev, std::string case_dir) {{
  for (size_t i = 0; i < chip_info->output_num; ++i) {{
    // 动态shape, 需要处理真实的output size
    if (TsmMemcpyD2H(hostboot->get_bootpmbuffer(), bootpm_dev,
        hostboot->get_maxlen()) != RET_SUCCESS) {{
      return RET_ERROR;
    }}

    auto out_tensor = hostboot->get_dev_output_tensor_after_run(i);
    chip_info->output[i]->dim = out_tensor->dim;
    std::memcpy(chip_info->output[i]->shape, out_tensor->shape, sizeof(out_tensor->shape));
    chip_info->output_size[i] = hrt_get_dtype_size((DTYPE)chip_info->output[i]->dtype);
    for (uint32_t j = 0; j < out_tensor->dim; ++j) {{
      if (out_tensor->shape[j] > 0) {{
        chip_info->output_size[i] *= out_tensor->shape[j];
      }}
    }}

    TsmHostPtr output_host_addr = (TsmHostPtr)malloc(chip_info->output_size[i]);
    if (chip_info->output_size[i] > 0) {{
      if (TsmMemcpyD2H((void*)output_host_addr, chip_info->output_dev_addr[i],
          chip_info->output_size[i]) != RET_SUCCESS) {{
        return RET_ERROR;
      }}
    }}

    printf("[Chip id %u] output_dev_addr=%ld\\n", devices[dev_index]->chip_id,
          chip_info->output_dev_addr[i]);

    // TODO: Processing output
#if 0
    std::string file_path = case_dir + "/chip" + std::to_string(dev_index) +
        "/agent/data/out" + std::to_string(i) + "_riscv.bin";
    saveDataToFile(file_path, output_host_addr, chip_info->output_size[i]);
#endif

    if (output_host_addr != 0) {{
      free((void *)output_host_addr);
    }}
  }}
  return RET_SUCCESS;
}}

TSM_RETCODE freeMemPerStep(uint32_t chip_id, TsmDevicePtr &bootpm_dev) {{
  if (bootpm_dev != 0) {{
    printf("[Chip id %u] bootpm dev addr: 0x%lx \\n", chip_id, bootpm_dev);
    if (TsmDeviceFree(bootpm_dev) != RET_SUCCESS) {{
      return RET_ERROR;
    }}
    bootpm_dev = 0;
  }}

  return RET_SUCCESS;
}}

static void setHostBoot(std::shared_ptr<chip_common_info_t> &chip_info,
  std::shared_ptr<HrtBootParam> &hostboot) {{
  if (chip_info == nullptr) {{
    printf("chip_info is null.\\n");
    return;
  }}

  if (hostboot == nullptr) {{
    printf("hostboot is null.\\n");
    return;
  }}

  for (size_t i = 0; i < chip_info->input_dev_addr.size(); ++i) {{
    hostboot->set_dev_input(i, chip_info->input_dev_addr[i], chip_info->input_size[i]);
    hostboot->set_dev_input_tensor(i, chip_info->input[i]);
  }}

  for (size_t i = 0; i < chip_info->output_dev_addr.size(); ++i) {{
    hostboot->set_dev_output(i, chip_info->output_dev_addr[i], chip_info->output_size[i]);
  }}

  for (size_t i = 0; i < chip_info->param_num; ++i) {{
    hostboot->set_dev_param(i, chip_info->param_dev_addr[i], chip_info->param_size[i]);
  }}

  return;
}}


static void _launch(int gridX, int gridY, int gridZ, std::vector<KernelArg> &kargs) {{
    std::vector<TsmDevice *> devices;

    if (gridX*gridY*gridZ <= 0) {{
        return;  // No work to do
    }}

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
    // FIXME: Hardcoded path
    new_model->case_dir = "/tmp/kernel.so";

    printf("====> Calling TsmCompileMultiGraph\\n");
#if 0
    if (TsmCompileMultiGraph(devices, *new_model, option, compl_option) != RET_SUCCESS) {{
        for (uint32_t dev_index = 0; dev_index < devices.size(); ++dev_index) {{
            if (TsmResetDevice(devices[dev_index]) != RET_SUCCESS) {{
                printf("[Chip id %u] tx_engine: tx_reset, failed!\\n", dev_index);
            }} else {{
                printf("[Chip id %u] tx_engine: tx_reset, success!\\n", dev_index);
            }}
        }}
        printf("TsmCompile failed.\\n");
        return;
    }}
#endif
    // Calculate the total size of kernel arguments buffer
    uint64_t kernel_buffer_size = 0;
    for (auto karg : kargs)
        kernel_buffer_size += karg.size;

    // Calcuate The kernel argument buffer header size
    // 4 bytes header + n * kernel argument metadata + 3 * sizeof(gridXYZ)
    uint64_t kernel_meta_buf_size = sizeof(uint64_t*) * kargs.size() + 4 + 12;
    kernel_buffer_size += kernel_meta_buf_size;

    // We use input_num = 1 to set the whole kernel arguments buffer as a single
    // input
    uint32_t input_num = 1;
    uint32_t output_num = 0;
    uint32_t param_num = 0;

    // Create boot parameter
    std::shared_ptr<HrtBootParam> hostboot = std::make_shared<HrtBootParam>(input_num, output_num, param_num);

    // Create chip common info
    std::shared_ptr<chip_common_info_t> chip_info = std::make_shared<chip_common_info_t>();
    chip_info->input_num = input_num;
    chip_info->output_num = output_num;
    chip_info->param_num = param_num;
    chip_info->imm_size = 0; // Cache size

    // Prepare input/output sizes and addresses
    chip_info->input_size.resize(input_num);
    chip_info->input_host_addr.resize(input_num);
    chip_info->input_dev_addr.resize(input_num);
    chip_info->output_size.resize(output_num);
    chip_info->output_host_addr.resize(output_num);
    chip_info->output_dev_addr.resize(output_num);

    // Prepare whole kernel buffer info
    chip_info->input.push_back(std::make_shared<tensor_info_t>());
    chip_info->input[0]->dim = 1;
    chip_info->input[0]->dtype = FMT_FP32;  // Default to float
    chip_info->input[0]->shape[0] = 1;      // Default shape
    chip_info->input_size[0] = kernel_buffer_size;
    chip_info->input_host_addr = std::vector<uint64_t>{{(uint64_t) 0x0}};

    // prepare data/ load kernel/run/unload kernel/get out data/release memory
    for (uint32_t dev_index = 0; dev_index < devices.size(); ++dev_index) {{
        // input prepare
        prepare_input(devices, dev_index, chip_info);
        // output prepare
        prepare_output(devices, dev_index, chip_info);

        uint32_t chip_id = devices[dev_index]->chip_id;
        TsmSetMonitorInfo(devices[dev_index]);

        // load kernel
        char module_symbol[] = "__kernel_entry";
        TsmLoadKernel(devices[dev_index], kmodel_vec, module_symbol);
        printf("TsmLoadKernel finish!...\\n");

        printf("[Chip id %u] Set boot-params...\\n", chip_id);
        size_t dyn_mod_size = sizeof(DynMods) + sizeof(DynModule);
        TsmDevicePtr dev_dyn_mods_ptr;
        if (TsmDeviceMalloc(devices[dev_index], dev_dyn_mods_ptr, dyn_mod_size) != RET_SUCCESS)
            return;

        // Allocate the device memory for all kernel arguments
        TsmDevicePtr dev_kernel_buffer;
        if (TsmDeviceMalloc(devices[dev_index], dev_kernel_buffer, kernel_buffer_size) != RET_SUCCESS)
            return;

        // Kernel meta data and argument buffer
        int dev_karg_ptr = dev_kernel_buffer + kernel_meta_buf_size;

        // Kernel arguments address
        uint64_t arg_metadata[kargs.size()];

        // Copy kernel arguments to device DDR (immediately after the metadata)
        int i = 0;
        uint64_t offset = 0;
        for (auto karg : kargs) {{
            if (TsmMemcpyH2D(dev_karg_ptr, karg.data.ptr, karg.size) != RET_SUCCESS)
                return;

            // Calculate the offset of each kernel arg's buffer
            arg_metadata[i++] = offset;

            // Shift the offset and pointer for next kernel argument.
            offset += karg.size;
            dev_karg_ptr += karg.size;
        }}

        // Create the metadata buffer
        uint32_t* metadata = (uint32_t*) malloc(kernel_meta_buf_size);
        metadata[0] = (int) kargs.size();
        metadata[1] = gridX;
        metadata[2] = gridY;
        metadata[3] = gridZ;
        memcpy(metadata+20, arg_metadata, kernel_meta_buf_size - 16);

        // Copy kernel metadata to device DDR
        if (TsmMemcpyH2D(dev_kernel_buffer, metadata, kernel_meta_buf_size) != RET_SUCCESS)
            return;

        setHostBoot(chip_info, hostboot);
        set_multi_graph(kmodel_vec[0], hostboot, dev_dyn_mods_ptr, 0, dev_kernel_buffer);

        TsmDevicePtr bootpm_dev;
        if (TsmDeviceMalloc(devices[dev_index], bootpm_dev, hostboot->get_maxlen()) != RET_SUCCESS)
            return;

        if (TsmMemcpyH2D(bootpm_dev, hostboot->get_bootpmbuffer(), hostboot->get_maxlen()) != RET_SUCCESS)
            return;

        if (TsmRun(devices[dev_index], bootpm_dev) != RET_SUCCESS) {{
            printf("TsmRun bootpm_dev failed.\\n");
            return;
        }}

        TsmUnloadKernel(devices[dev_index], kmodel_vec);

        // Process kernel output data
        printf("[Chip id %u] Copy output from device...\\n", chip_id);
        if (kernel_result_process(devices, dev_index, hostboot, chip_info, bootpm_dev, new_model->case_dir) != RET_SUCCESS) {{
            printf("free dev memory failed.\\n");
            return;
        }}

        if (freeMemPerStep(chip_id, bootpm_dev) != RET_SUCCESS) {{
            printf("free dev memory failed.\\n");
            return;
        }}

        if (TsmDeviceFree(dev_kernel_buffer) != RET_SUCCESS) {{
            printf("free dev_kernel_param_ptr failed.\\n");
            return;
        }}

        if (TsmDeviceFree(dev_dyn_mods_ptr) != RET_SUCCESS) {{
            printf("free dev_dyn_mods_ptr failed.\\n");
            return;
        }}

        printf("[dev_index %u] Set Terminal Info...\\n", dev_index);
        if (TsmSetTerminate(devices[dev_index]) != RET_SUCCESS) {{
            printf("TsmSetTerminate failed.\\n");
            return;
        }}
    }}

    // Clean up the model
    delete new_model;
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
    {' '.join([f"kargs.emplace_back(extractTensor(_arg{i}), getTensorStorageSize(_arg{i}));"
               if ty[0]=="*" else f"kargs.emplace_back(_arg{i}, sizeof(_arg{i}));"
                  for i, ty in signature.items()])}

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

#if 0
    // Initialize Tx81 runtime during module import
    if (!init_tx81_runtime()) {{
        Py_DECREF(m);
        return NULL;
    }}

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
#endif

    return m;
}}
"""


class CrossUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CrossUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(os.path.join(dirname, "driver.cpp")).read_text()
        mod = compile_native(src, "tx81_utils")
        # NOTE: The triton compiler.py framework requires these 2 interface.
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


# Launch cross compiled runtime program on controller
class CrossLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        # Compiler kernel wrapper source code
        # NOTE: Replace this make_kernel_wrapper to v2 version by if you want
        # to call the triton kernel with single input buffer and with a '__'
        # prefixed name.
        wrapper_src = make_kernel_wrapper(constants, signature, src.fn.__name__)
        krn = compile_accelerator(wrapper_src, src.fn.__name__, "c")

        # Compiler runtime kernel launcher source code
        launcher_src = make_launcher(constants, signature, src.fn.__name__)
        mod = compile_native(launcher_src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        # args: 0: gridX, 1: gridY, 2: gridZ,
        #       3: kernel_metadata?, 4: launch_metadata?,
        #       5: a tuple(0, 0, False, 1, 1, 1, 'add_kernel'), # this is probably kernel metadata
        #       6: None, 7: None, 8: None,
        #       9~N: Actual triton kernel args.
        self.launch(*args, **kwargs)


class CrossDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = CrossUtils()
        self.launcher_cls = CrossLauncher
        # Needs to overwrite GPUDriver base methods
        self.get_current_device = self.get_npu_device
        self.set_current_device = self.set_npu_device
        self.get_current_stream = self.get_npu_stream

    @staticmethod
    def is_active():
        return True

    def get_npu_device(self):
        return "cpu"

    def set_npu_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_npu_stream(self, device):
        return None

    def get_current_target(self):
        capability = 1
        warp_size = 16
        return GPUTarget("cpu", capability, warp_size)
