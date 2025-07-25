#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>
#include <optional>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";
constexpr static char AttrNumCTAsName[] = "triton_gpu.num-ctas";
constexpr static char AttrTargetName[] = "triton_gpu.target";

constexpr static char AttrNumThreadsPerWarp[] = "triton_gpu.threads-per-warp";

#ifdef __ILUVATAR__
constexpr static char AttrNumStagesForDot[] = "triton_gpu.dot.num-stages";
#endif

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(const std::string &target, int numWarps,
                                   int threadsPerWarp = 32, int numCTAs = 1,
                                   int numStages = 1);

} // namespace triton
} // namespace mlir

#endif
