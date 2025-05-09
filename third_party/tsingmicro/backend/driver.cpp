//===---------------------------- driver.c --------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Tx81 platform device side runtime interface for python.
//
//===----------------------------------------------------------------------===//
#include <hrt_interface.h>
#include <hrt_common.h>
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct Kernel_Param // Triton kernel arguments
{
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    // TODO...
};

struct Kernel_Head
{
    uint32_t param_type;
    uint32_t param_num;
    uint32_t param_addr;
    uint32_t xxxxx;
};

// Raises a Python exception and returns false if code is not RET_SUCCESS.
static bool tsmAssert(TSM_RETCODE code, const char *file, int line) {
  if (code == RET_SUCCESS)
    return true;

  const char *prefix = "Triton Error [TX81]: ";
  const char *str;

  // Map error codes to strings
  switch(code) {
    case RET_ERROR:
      str = "General error";
      break;
    case RET_PARAM1_ERROR:
    case RET_PARAM2_ERROR:
    case RET_PARAM3_ERROR:
      str = "Parameter error";
      break;
    case RET_DEVICE_OFFLINE:
      str = "Device offline";
      break;
    case RET_DEVICE_NOMEM:
      str = "Device out of memory";
      break;
    case RET_DEVICE_IN_IDLE:
      str = "Device in idle state";
      break;
    case RET_DEVICE_IN_ATTACH:
      str = "Device already attached";
      break;
    case RET_DEVICE_ATTACH_SUCCESS:
      str = "Device attach success";
      break;
    case RET_DEVICE_ATTACH_READY:
      str = "Device attach ready";
      break;
    case RET_DEVICE_LOSE_CONNECT:
      str = "Device connection lost";
      break;
    case RET_ENV_CLEAN_UP:
      str = "Environment cleanup required";
      break;
    default:
      str = "Unknown error";
  }

  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}


static void prepare_input(std::vector<TsmDevice *> devices, uint32_t dev_index,
    std::shared_ptr<chip_common_info_t> chip_info)
{
  for (uint32_t i = 0; i < chip_info->input_num; ++i) {
    chip_info->input_dev_addr.push_back(0);
    if (TsmDeviceMalloc(devices[dev_index], chip_info->input_dev_addr[i],
        chip_info->input_size[i]) != RET_SUCCESS) {
      printf("[Chip id %u] Input%d, DeviceMalloc failed!\n", devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }

    if (TsmMemcpyH2D((TsmDevicePtr)chip_info->input_dev_addr[i],
                     (void*) chip_info->input_host_addr[i],
                     chip_info->input_size[i]) != RET_SUCCESS) {
      printf("[Chip id %u] Input%d, MemcpyH2D failed!\n", devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }
  }
}

static void prepare_output(std::vector<TsmDevice *> devices, uint32_t dev_index,
  std::shared_ptr<chip_common_info_t> chip_info) {
  for (size_t i = 0; i < chip_info->output_num; ++i) {
    chip_info->output_dev_addr.push_back(0);
    printf("[Chip id %u] output[%lu] data(size: %lu)\n",
           devices[dev_index]->chip_id, i, chip_info->output_size[i]);

    if (TsmDeviceMalloc(devices[dev_index], chip_info->output_dev_addr[i],
        chip_info->output_size[i]) != RET_SUCCESS) {
      printf("[Chip id %u] output[%lu], DeviceMalloc failed!\n",
             devices[dev_index]->chip_id, i);
      TsmResetDevice(devices[dev_index]);
      return;
    }
  }
}

TSM_RETCODE kernel_result_process(std::vector<TsmDevice *> devices, uint32_t dev_index,
              std::shared_ptr<HrtBootParam> hostboot,
              std::shared_ptr<chip_common_info_t> chip_info,
              TsmDevicePtr bootpm_dev, std::string case_dir) {
  for (size_t i = 0; i < chip_info->output_num; ++i) {
    // 动态shape，需要处理真实的output size
    if (TsmMemcpyD2H(hostboot->get_bootpmbuffer(), bootpm_dev,
        hostboot->get_maxlen()) != RET_SUCCESS) {
      return RET_ERROR;
    }

    auto out_tensor = hostboot->get_dev_output_tensor_after_run(i);
    chip_info->output[i]->dim = out_tensor->dim;
    std::memcpy(chip_info->output[i]->shape, out_tensor->shape, sizeof(out_tensor->shape));
    chip_info->output_size[i] = hrt_get_dtype_size((DTYPE)chip_info->output[i]->dtype);
    for (uint32_t j = 0; j < out_tensor->dim; ++j) {
      if (out_tensor->shape[j] > 0) {
        chip_info->output_size[i] *= out_tensor->shape[j];
      }
    }

    TsmHostPtr output_host_addr = (TsmHostPtr)malloc(chip_info->output_size[i]);
    if (chip_info->output_size[i] > 0) {
      if (TsmMemcpyD2H((void*)output_host_addr, chip_info->output_dev_addr[i],
          chip_info->output_size[i]) != RET_SUCCESS) {
        return RET_ERROR;
      }
    }

    printf("[Chip id %u] output_dev_addr=%ld\n", devices[dev_index]->chip_id,
          chip_info->output_dev_addr[i]);

    // TODO: Processing output
#if 0
    std::string file_path = case_dir + "/chip" + std::to_string(dev_index) +
        "/agent/data/out" + std::to_string(i) + "_riscv.bin";
    saveDataToFile(file_path, output_host_addr, chip_info->output_size[i]);
#endif

    if (output_host_addr != 0) {
      free((void *)output_host_addr);
    }
  }
  return RET_SUCCESS;
}

TSM_RETCODE freeMemPerStep(uint32_t chip_id, TsmDevicePtr &bootpm_dev) {
  if (bootpm_dev != 0) {
    printf("[Chip id %u] bootpm dev addr: 0x%lx \n", chip_id, bootpm_dev);
    if (TsmDeviceFree(bootpm_dev) != RET_SUCCESS) {
      return RET_ERROR;
    }
    bootpm_dev = 0;
  }

  return RET_SUCCESS;
}

static void setHostBoot(std::shared_ptr<chip_common_info_t> &chip_info,
  std::shared_ptr<HrtBootParam> &hostboot) {
  if (chip_info == nullptr) {
    printf("chip_info is null.\n");
    return;
  }

  if (hostboot == nullptr) {
    printf("hostboot is null.\n");
    return;
  }

  for (size_t i = 0; i < chip_info->input_dev_addr.size(); ++i) {
    hostboot->set_dev_input(i, chip_info->input_dev_addr[i], chip_info->input_size[i]);
    hostboot->set_dev_input_tensor(i, chip_info->input[i]);
  }

  for (size_t i = 0; i < chip_info->output_dev_addr.size(); ++i) {
    hostboot->set_dev_output(i, chip_info->output_dev_addr[i], chip_info->output_size[i]);
  }

  for (size_t i = 0; i < chip_info->param_num; ++i) {
    hostboot->set_dev_param(i, chip_info->param_dev_addr[i], chip_info->param_size[i]);
  }

  return;
}


// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define TSM_CHECK_AND_RETURN_NULL(ans)                                         \
  do {                                                                         \
    if (!tsmAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define TSM_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                           \
  do {                                                                         \
    if (!tsmAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Global state for Tx81 devices
static std::vector<TsmDevice*> g_tx81_devices;
static bool g_runtime_initialized = false;

// Initialize the Tx81 runtime if not already initialized
static bool init_tx81_runtime_if_needed() {
  if (g_runtime_initialized) {
      return true;
  }

  // Initialize the Tx81 runtime
  if (TsmInitRuntime() != RET_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Tx81 runtime");
      return false;
  }

  // Get device count
  uint32_t device_num = 0;
  if (TsmGetDeviceNum(device_num) != RET_SUCCESS || device_num == 0) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to get Tx81 device count or no devices found");
      TsmDeInitRuntime();
      return false;
  }

  // Set up devices - for simplicity, we're using a 1x1 configuration
  uint32_t first_phy_id = 0;
  uint32_t card_x = 1;
  uint32_t card_y = 1;

  if (TsmSetDevice(first_phy_id, card_x, card_y, g_tx81_devices) != RET_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to set Tx81 devices");
      TsmDeInitRuntime();
      return false;
  }

  g_runtime_initialized = true;
  return true;
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
#if 0
  // FIXME: Extracting device_id
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;


  // Initialize the runtime if needed
  if (!init_tx81_runtime_if_needed()) {
    return NULL;
  }

  // Check device ID is valid
  if (device_id < 0 || (size_t)device_id >= g_tx81_devices.size()) {
    PyErr_SetString(PyExc_ValueError, "Invalid device ID");
    return NULL;
  }

  // Get device handle
  TsmDevice* device = g_tx81_devices[device_id];

  // Get device information
  TsmDeviceInfo info;
  memset(&info, 0, sizeof(TsmDeviceInfo));
  TSM_CHECK_AND_RETURN_NULL(TsmGetDeviceInfo(&info));
#endif
  // Extract device properties
  // Note: We're mapping Tx81 properties to fields expected by Triton
  int max_shared_mem = 1024 * 1024 * 4; // Default 4MB
  //int multiprocessor_count = device->tile_num;
  int multiprocessor_count = 1;
  int sm_clock_rate = 1000; // Placeholder
  int mem_clock_rate = 2000; // Placeholder
  int mem_bus_width = 256;   // Placeholder

#if 0
  // For the specified device, get more detailed info
  if (device_id < (int)info.card_num) {
    CardComputeInfo& card_info = info.card_compute_info[device_id];
    multiprocessor_count = card_info.all_tile_num;
  }
#endif

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}",
                       "max_shared_mem", max_shared_mem,
                       "multiprocessor_count", multiprocessor_count,
                       "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate,
                       "mem_bus_width", mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
#if 0
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }

  // Initialize the runtime if needed
  if (!init_tx81_runtime_if_needed()) {
    return NULL;
  }

  // Check device ID is valid
  if (device < 0 || (size_t)device >= g_tx81_devices.size()) {
    PyErr_SetString(PyExc_ValueError, "Invalid device ID");
    return NULL;
  }

  TsmDevice* tx81_device = g_tx81_devices[device];

  // First, we need to write binary data to a temporary file
  char temp_path[256];
  sprintf(temp_path, "/tmp/triton_tx81_kernel_XXXXXX");
  int fd = mkstemp(temp_path);
  if (fd == -1) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create temporary file");
    return NULL;
  }

  // Write the kernel data to the temporary file
  if (write(fd, data, data_size) != data_size) {
    close(fd);
    unlink(temp_path);
    PyErr_SetString(PyExc_RuntimeError,
        "Failed to write kernel data to temporary file");
    return NULL;
  }
  close(fd);

  // Create a model structure, the compiled kernel.so is specified via case_dir
  // and the name of the entry function is specified via case_name.
  TsmModel *model = new TsmModel();
  model->case_dir = std::string(temp_path);
  model->case_name = std::string(name);

  // Set compile options
  CompileOption compl_option = {};
  compl_option.comp_enable = 0; // Use precompiled kernel.so instead
  compl_option.chip_x = 1;
  compl_option.chip_y = 1;
  compl_option.check_enable = true;
  compl_option.enable_kcore_bin = 1;
  compl_option.enable_kcore_so = 1;

  std::vector<TsmDevice*> devices = {tx81_device};

  // Not really compile the kernel, as kernel is already compiled, so this
  // runtime API only configs the data structure of device firmware and the
  // information of the program and data that runs on it.
  Py_BEGIN_ALLOW_THREADS;
  TSM_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      TsmCompileMultiGraph(devices, *model, "", compl_option));
  Py_END_ALLOW_THREADS;

  // For Tx81, we'll use a simpler model than CUDA
  // We return a pointer to the TsmModel, which is analogous to CUmodule
  // For the function pointer, we'll use model_id+0, which will be interpreted
  // in the launcher code
  // n_regs and n_spills are placeholders for now
  int32_t n_regs = 256; // Default/placeholder value
  int32_t n_spills = 0; // Default/placeholder value

  // Clean up the temporary file
  unlink(temp_path);
#endif

  int32_t n_regs = 256;
  int32_t n_spills = 0;
  // Return values to Python including module, function, n_regs, n_spills
  return Py_BuildValue("(KKii)", "module {}", "void @add_kernel() {}", n_regs, n_spills);
}



static PyObject *launch(PyObject *self, PyObject* args) {
  std::vector<TsmDevice *> devices;
  // TODO:通过参数传递获取device信息


  // 需要的输入信息： devices, case_dir(按固定路径存放的kernelso)， input_host_addr/input_size/input_num,
  // output_host_addr/output_size/output_num, param信息（如果有权重）


  TsmModel *new_model = new TsmModel();  // 设备相关参数已在dev中
  std::string option = "-O2";
  CompileOption compl_option = {};
  compl_option.comp_enable = 0;
  compl_option.chip_x = 1; //单卡
  compl_option.chip_y = 1;
  compl_option.check_enable = true;
  compl_option.enable_kcore_bin = 1;
  compl_option.enable_kcore_so = 1;
  new_model->case_dir = "/tmp/todo"; // 参数传入， kernelso路径，同streambin/kcorebin文件夹路径

  if (TsmCompileMultiGraph(devices, *new_model, option, compl_option) != RET_SUCCESS) {
      for (uint32_t dev_index = 0; dev_index < devices.size(); ++dev_index) {
          if (TsmResetDevice(devices[dev_index]) != RET_SUCCESS) {
              printf("[Chip id %u] tx_engine: tx_reset, failed!\n", dev_index);
          } else {
              printf("[Chip id %u] tx_engine: tx_reset, success!\n", dev_index);
          }
      }
      printf("TsmCompile failed.\n");
      return NULL;
  }

  std::vector<TsmModel *> kmodel_vec = {new_model};

  uint32_t input_num = 2;  // TODO：根据kernel参数填写
  uint32_t output_num = 1; // TODO：根据kernel参数填写
  uint32_t param_num = 0;  // 权重数
  std::shared_ptr<HrtBootParam> hostboot = std::make_shared<HrtBootParam>(input_num, output_num, param_num);

  std::shared_ptr<chip_common_info_t> chip_info;
  // 填充chipinfo信息
  chip_info->input_num = input_num;
  chip_info->output_num = output_num;
  chip_info->param_num = param_num;
  chip_info->imm_size = 0; //缓存大小暂设置为0，和算子实际相关；
  // chip_info->tile_num = 16; // 未使用
  // chip_info->tile_x = 4; // 未使用
  // chip_info->tile_y = 4; // 未使用
  for(uint32_t i = 0; i < chip_info->input_num; ++i) {
      chip_info->input_size[i] = 6; // TODO：填写实际输入大小
      chip_info->input_host_addr = std::vector<uint64_t>{0x0, 0x0, 0x0, 0x0, 0x0, 0x0}; // TODO: 填写实际输入地址
  }

  for(uint32_t i = 0; i < chip_info->output_num; ++i) {
      chip_info->output_size[i] = 1; // TODO：填写实际输出大小
      chip_info->output_host_addr = std::vector<uint64_t>{0x0}; // TODO: 填写实际输出地址
  }

  //for(uint32_t i = 0; i < chip_info->param_num; ++i) {
  //    chip_info->param_size[i] = 0; // TODO：填写实际权重大小
  //    chip_info->param_host_addr = 0x0;
  //}

  // prepare data/ load kernel/run/unload kernel/get out data/release memory
  for (uint32_t dev_index = 0; dev_index < devices.size(); ++dev_index) {
      // input prepare
      prepare_input(devices, dev_index, chip_info);
      // output prepare
      prepare_output(devices, dev_index, chip_info);

      uint32_t chip_id = devices[dev_index]->chip_id;
      TsmSetMonitorInfo(devices[dev_index]);

      // load kernel
      char module_symbol[] = "main_kernel";
      TsmLoadKernel(devices[dev_index], kmodel_vec, module_symbol);
      printf("TsmLoadKernel finish!...\n");

      printf("[Chip id %u] Set boot-params...\n", chip_id);
      size_t dyn_mod_size = sizeof(DynMods) + sizeof(DynModule);
      TsmDevicePtr dev_dyn_mods_ptr;
      if (TsmDeviceMalloc(devices[dev_index], dev_dyn_mods_ptr, dyn_mod_size) != RET_SUCCESS) {
          return NULL;
      }
      TsmDevicePtr dev_tlv_ptr;
      if (TsmDeviceMalloc(devices[dev_index], dev_tlv_ptr, sizeof(DynTLV_DynMods)) != RET_SUCCESS) {
          return NULL;
      }

      TsmDevicePtr dev_kernel_head_ptr;
      if (TsmDeviceMalloc(devices[dev_index], dev_kernel_head_ptr, sizeof(Kernel_Head)) != RET_SUCCESS) {
          return NULL;
      }
      TsmDevicePtr dev_kernel_param_ptr;
      if (TsmDeviceMalloc(devices[dev_index], dev_kernel_param_ptr, sizeof(Kernel_Param)) != RET_SUCCESS) {
          return NULL;
      }

      Kernel_Head *host_kernel_head_ptr = (Kernel_Head*)malloc(sizeof(Kernel_Head));
      Kernel_Param *host_kernel_param_ptr = (Kernel_Param*)malloc(sizeof(Kernel_Param));

      host_kernel_head_ptr->param_type = 1;
      host_kernel_head_ptr->param_num = 1; // Number of kernel arguments
      host_kernel_head_ptr->param_addr = dev_kernel_param_ptr; // 将kernel 使用的参数地址赋值

      // TODO: Setup the triton kernel arguments
      host_kernel_param_ptr->gridX = 512;
      host_kernel_param_ptr->gridY = 512;
      host_kernel_param_ptr->gridZ = 512;

      TsmMemcpyH2D(dev_kernel_head_ptr,  host_kernel_head_ptr, sizeof(Kernel_Head));
      TsmMemcpyH2D(dev_kernel_param_ptr,  host_kernel_param_ptr, sizeof(Kernel_Param));

      free(host_kernel_head_ptr);
      free(host_kernel_param_ptr);

      // TODO: No such API
      setHostBoot(chip_info, hostboot);
      set_multi_graph(kmodel_vec[0], hostboot, dev_dyn_mods_ptr, dev_tlv_ptr, dev_kernel_head_ptr);

      TsmDevicePtr bootpm_dev;
      if (TsmDeviceMalloc(devices[dev_index], bootpm_dev, hostboot->get_maxlen()) != RET_SUCCESS) {
          return NULL;
      }
      if (TsmMemcpyH2D(bootpm_dev, hostboot->get_bootpmbuffer(), hostboot->get_maxlen()) != RET_SUCCESS) {
          return NULL;
      }

      if (TsmRun(devices[dev_index], bootpm_dev) != RET_SUCCESS) {
          printf("TsmRun bootpm_dev failed.\n");
          return NULL;
      }

      // 卸载kernel
      TsmUnloadKernel(devices[dev_index], kmodel_vec);

      // 得到输出数据，并进行处理
      printf("[Chip id %u] Copy output from device...\n", chip_id);
      if (kernel_result_process(devices, dev_index, hostboot, chip_info, bootpm_dev, new_model->case_dir) != RET_SUCCESS) {
          printf("free dev memory failed.\n");
          return NULL;
      }
      if (freeMemPerStep(chip_id, bootpm_dev) != RET_SUCCESS) {
          printf("free dev memory failed.\n");
          return NULL;
      }
      //释放多图相关tlv
      if (TsmDeviceFree(dev_kernel_head_ptr) != RET_SUCCESS) {
          printf("free dev_kernel_head_ptr failed.\n");
          return NULL;
      }
      if (TsmDeviceFree(dev_kernel_param_ptr) != RET_SUCCESS) {
          printf("free dev_kernel_param_ptr failed.\n");
          return NULL;
      }

      if (TsmDeviceFree(dev_dyn_mods_ptr) != RET_SUCCESS) {
          printf("free dev_dyn_mods_ptr failed.\n");
          return NULL;
      }
      if (TsmDeviceFree(dev_tlv_ptr) != RET_SUCCESS) {
          printf("free dev_tlv_ptr failed.\n");
          return NULL;
      }

      printf("[dev_index %u] Set Terminal Info...\n", dev_index);
      if (TsmSetTerminate(devices[dev_index]) != RET_SUCCESS) {
          printf("TsmSetTerminate failed.\n");
          return NULL;
      }
#if 0
      if (freeTensorData(chip_id, chip_info) != RET_SUCCESS) {
        printf("free tensor data dev memory failed.\n");
      }
#endif
  }

  Py_RETURN_NONE;
}


static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided binary into Tx81 driver"},
    {"launch", launch, METH_VARARGS, "tx8 launch kernel!"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given Tx81 device"},

    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "tx81_utils",
    NULL, // documentation
    -1,   // size
    ModuleMethods
};

PyMODINIT_FUNC PyInit_tx81_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}