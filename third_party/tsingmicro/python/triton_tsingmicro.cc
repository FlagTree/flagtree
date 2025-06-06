#include <pybind11/pybind11.h>

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToCoreDialects/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"

#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h"
#include "magic-kernel/Conversion/LinalgToMK/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h"

#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h"

namespace py = pybind11;
using namespace mlir;

void init_triton_tsingmicro(py::module &&m) {}
