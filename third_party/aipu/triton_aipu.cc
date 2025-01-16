#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;


void init_triton_aipu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
  });
  // register passes here
}
