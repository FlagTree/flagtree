#ifndef TRITONHCUGPU_CONVERSION_PASSES_H
#define TRITONHCUGPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "TritonHCUGPUToLLVM/Passes.h.inc"

namespace HCU {
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass(StringRef targetArch);

/// @brief Creates pass that keep LDS consumption within specified limits.
/// @param arch target architecture name, for example "gfx940"
/// @param customLDSLimit defines LDS size available for one thread block
/// zero value tells pass that whole LDS is available on a device
/// @return created pass
std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(StringRef arch, int32_t customLDSLimit = 0);
} // namespace HCU

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonHCUGPUToLLVMPass(StringRef targetArch, bool ftz);
std::unique_ptr<OperationPass<ModuleOp>> createConvertBuiltinFuncToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createInsertInstructionSchedHintsPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLowerInstructionSchedHintsPass(std::string variant);

#define GEN_PASS_REGISTRATION
#include "TritonHCUGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
