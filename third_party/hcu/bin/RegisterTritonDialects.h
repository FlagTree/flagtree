#pragma once
#include "hcu/include/triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "hcu/include/TritonHCUTransforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonHCUGPU/IR/Dialect.h"

// Below headers will allow registration to ROCm passes
#include "TritonHCUGPUToLLVM/Passes.h"
#include "TritonHCUTransforms/Passes.h"
#include "TritonHCUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"

#include "python/src/plugin.h"

using BackendRegisterFunc = void (*)();
BackendRegisterFunc load_backend_register_func(const char *backend_name,
                                               const char *func_name) {
  void *symbol = load_backend_symbol(backend_name, func_name);
  return reinterpret_cast<BackendRegisterFunc>(symbol);
}
inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();

  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerAllocateSharedMemoryPass();

  mlir::registerLLVMDIScope();

  // TritonHCUGPUToLLVM passes
  mlir::triton::registerConvertTritonHCUGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerDecomposeUnsupportedHCUConversions();
  mlir::triton::registerOptimizeHCULDSUsage();

  // TritonHCUTransforms passes
  mlir::registerTritonHCUGPUAccelerateMatmul();
  mlir::registerTritonHCUGPUOptimizeEpilogue();
  mlir::registerTritonHCUGPUReorderInstructions();
  mlir::registerTritonHCUGPUStreamPipeline();
  mlir::registerTritonHCUGPUStreamPipelineV2();
  mlir::registerTritonHCUGPUCanonicalizePointers();
  mlir::registerTritonHCUGPUConvertToBufferOps();
  // TODO: register Triton & TritonGPU passes
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,

                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,

                  mlir::triton::hcugpu::,
                  mlir::ROCDL::ROCDLDialect>();

}
