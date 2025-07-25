#include "cn_api.h"
#include "cnrt.h"
#include "cnrtc.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdio.h>

static inline bool isResultSuccessed(CNresult code, const char *file,
                                     int line) {
  if (code == CN_SUCCESS)
    return true;

  const char *prefix = "Triton Error [MLU]: ";
  const char *str;
  cnGetErrorString(code, &str);
  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MLU_CHECK_AND_RETURN_NULL(ans)                                         \
  do {                                                                         \
    if (!isResultSuccessed((ans), __FILE__, __LINE__))                         \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MLU_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                           \
  do {                                                                         \
    if (!isResultSuccessed((ans), __FILE__, __LINE__)) {                       \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// FIXME(liangyuefeng): As CNDRV doesn't support legacy default context,
// we need init CNcontext for each thread now.
static void *initCNContext(int device) {
  CNcontext pctx;
  Py_BEGIN_ALLOW_THREADS;
  MLU_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      cnSharedContextAcquire(&pctx, device));
  MLU_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cnCtxSetCurrent(pctx));
  Py_END_ALLOW_THREADS;
  return NULL;
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  initCNContext(device_id);
  // Get device handle
  CNdev pdev;
  cnDeviceGet(&pdev, device_id);

  // create a struct to hold device properties
  CNcontext drv_ctx;
  MLU_CHECK_AND_RETURN_NULL(cnCtxGetCurrent(&drv_ctx));
  CNctxConfigParam ctx_conf_param;
  MLU_CHECK_AND_RETURN_NULL(cnGetCtxConfigParam(
      drv_ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM, &ctx_conf_param));
  int cluster_num = (int)ctx_conf_param.visibleClusterNumber;
  int core_num_per_cluster;
  int nram_size;
  int wram_size;
  int sram_size;
  int max_l2_cache_size;
  int cluster_clock_rate;
  int memory_clock_rate;
  int memory_bus_width;
  int isa_version;
  int max_block_task_dim_x;
  int max_block_task_dim_y;
  int max_block_task_dim_z;
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &core_num_per_cluster, CN_DEVICE_ATTRIBUTE_MAX_CORE_COUNT_PER_CLUSTER,
      pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &nram_size, CN_DEVICE_ATTRIBUTE_NRAM_SIZE_PER_CORE, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &wram_size, CN_DEVICE_ATTRIBUTE_WEIGHT_RAM_SIZE_PER_CORE, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &isa_version, CN_DEVICE_ATTRIBUTE_MLU_ISA_VERSION, pdev));
  if (isa_version >= 600)
    nram_size += wram_size;
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &sram_size, CN_DEVICE_ATTRIBUTE_MAX_SHARED_RAM_SIZE_PER_CLUSTER, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &max_l2_cache_size, CN_DEVICE_ATTRIBUTE_MAX_L2_CACHE_SIZE, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &cluster_clock_rate, CN_DEVICE_ATTRIBUTE_CLUSTER_CLOCK_RATE, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &memory_clock_rate, CN_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &memory_bus_width, CN_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &max_block_task_dim_x, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_X, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &max_block_task_dim_y, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_Y, pdev));
  MLU_CHECK_AND_RETURN_NULL(cnDeviceGetAttribute(
      &max_block_task_dim_z, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_Z, pdev));

  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}",
      "cluster_num", cluster_num, "core_num_per_cluster", core_num_per_cluster,
      "max_nram_size", nram_size, "max_shared_mem", sram_size,
      "max_l2_cache_size", max_l2_cache_size, "cluster_clock_rate",
      cluster_clock_rate, "memory_bus_width", memory_bus_width,
      "memory_clock_rate", memory_clock_rate, "isa_version", isa_version,
      "max_block_task_dim_x", max_block_task_dim_x, "max_block_task_dim_y",
      max_block_task_dim_y, "max_block_task_dim_z", max_block_task_dim_z);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  int data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }
  CNkernel fun;
  CNmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  initCNContext(device);
  Py_BEGIN_ALLOW_THREADS;
  MLU_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cnModuleLoadFatBinary(data, &mod));
  MLU_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cnModuleGetKernel(mod, name, &fun));
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyObject *isLinearPointer(PyObject *self, PyObject *args) {
  uintptr_t ptr;
  int device;
  if (!PyArg_ParseTuple(args, "Ki", &ptr, &device)) {
    return NULL;
  }
  CNmem_attribute attr = CN_MEM_ATTRIBUTE_ISLINEAR;
  int pi;
  initCNContext(device);
  MLU_CHECK_AND_RETURN_NULL(cnGetMemAttribute(&pi, attr, (CNaddr)ptr));
  if (PyErr_Occurred()) {
    return NULL;
  }
  return PyBool_FromLong(pi);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cnfatbin into mlu driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"is_linear_pointer", isLinearPointer, METH_VARARGS,
     "Check if a given data pointer is a linear pointer"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "bang_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_bang_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
