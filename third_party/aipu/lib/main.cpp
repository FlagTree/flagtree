#include "Passes/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "RegisterTritonSharedDialects.h"
#include "mlir-ext/Dialect/MathExt/IR/MathExt.h"

using namespace mlir::aipu;

void init_all_passes(nanobind::module_ &&m) {
  m.def("register_all_passes", []() {
    registerAIPUConvertBoolArg2I8();
    registerSCFLoopBufferizationPreprocessing();
    registerFlattenMemrefsPass();
    registerForwardStoreToLoadPass();
    registerConvertI64ToI32Pass();
  });
}

void init_all_dialects(nanobind::module_ &&m) {
  m.def("register_all_dialects", [](nanobind::object capsule) {
    mlir::DialectRegistry registry;

    registry.insert<mlir::tptr::TPtrDialect, mlir::ttx::TritonTilingExtDialect,
                    mlir::tts::TritonStructuredDialect,
                    mlir::triton::TritonDialect>();
    mlir::ttx::registerBufferizableOpInterfaceExternalModels(registry);

    MlirContext context = mlirPythonCapsuleToContext(capsule.ptr());
    auto &ctx = *unwrap(context);
    ctx.appendDialectRegistry(registry);
  });
}

NB_MODULE(aipu_interface, m) {
  init_all_passes(m.def_submodule("passes"));
  init_all_dialects(m.def_submodule("dialects"));
}
