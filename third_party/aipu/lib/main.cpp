#include "Passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>

using namespace mlir::aipu;

void init_aipu_passes(pybind11::module &&m) {
  m.def("register_all_passes", []() { registerAIPUConvertBoolArg2I8(); });
}

PYBIND11_MODULE(libaipu_interface, m) {
  init_aipu_passes(m.def_submodule("passes"));
}
