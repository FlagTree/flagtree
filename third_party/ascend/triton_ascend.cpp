/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
namespace py = pybind11;

// register huawei passes to triton
void init_triton_ascend(py::module &&m) {
  // currently no extra modules needed to plug-in libtriton.so
}
