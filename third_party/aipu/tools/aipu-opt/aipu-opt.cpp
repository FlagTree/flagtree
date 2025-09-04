#include "Passes/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllPasses();
  mlir::registerAllDialects(registry);
  mlir::aipu::registerAIPUConvertBoolArg2I8();
  mlir::aipu::registerSCFLoopBufferizationPreprocessing();
  mlir::aipu::registerFlattenMemrefsPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AIPU optimizer driver\n", registry));
}
