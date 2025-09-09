#ifndef TRITON_DIALECT_TRITONHCU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONHCU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "third_party/hcu/include/triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Dialect/TritonHCUGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass> createTritonHCUGPUStreamPipelinePass();

std::unique_ptr<Pass> createTritonHCUGPUStreamPipelineV2Pass(int numStages = 2);

std::unique_ptr<Pass>
createTritonHCUGPUAccelerateMatmulPass(std::string archGenName = std::string(),
                                       int matrixInstructionSize = 0,
                                       int kpack = 1,
                                       int num_ldmatrixes = 0,
                                       int enable_mmacfuse = 0);

std::unique_ptr<Pass> createTritonHCUGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonHCUGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonHCUGPUVerifier();

std::unique_ptr<Pass> createTritonHCUGPUConvertToBufferOpsPass();

std::unique_ptr<Pass> createTritonHCUGPUOptimizeEpiloguePass();

std::unique_ptr<Pass> createTritonHCUGPUCanonicalizePointersPass();

std::unique_ptr<Pass> createTritonHCUAccelerateFlashAttentionPass();

std::unique_ptr<Pass> createTritonHCUMoveLoadToFrontOfDOTPass();

std::unique_ptr<Pass> createTritonHCUGlobalToLocalSwizzlePass();

std::unique_ptr<Pass> createTritonHCUFaFwdControlCntPass(int cnt_control = 0);

std::unique_ptr<Pass> createTritonHCUFaFwdWaitPass(int cnt_control = 0);

std::unique_ptr<Pass> createTritonHCUUpdateAsyncWaitCountPass();

std::unique_ptr<Pass> createTritonHCUStreamPipelinePass(int numStages = 2, int globalPrefetch = 0,
                                     int localPrefetch = 0,
                                     bool useAsyncCopy = false);

#define GEN_PASS_REGISTRATION
#include "TritonHCUTransforms/Passes.h.inc"

}
#endif