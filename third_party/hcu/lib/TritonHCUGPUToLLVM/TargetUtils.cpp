#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "llvm/TargetParser/TargetParser.h"

namespace mlir::triton::HCU {

ISAFamily deduceISAFamily(llvm::StringRef arch) {
  llvm::AMDGPU::GPUKind kind = llvm::AMDGPU::parseArchAMDGCN(arch);

  // See https://llvm.org/docs/AMDGPUUsage.html#processors for how to categorize
  // the following target gfx architectures.

  // CDNA ISA cases
  switch (kind) {
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX941:
  case llvm::AMDGPU::GK_GFX940:
    return ISAFamily::CDNA3;
  case llvm::AMDGPU::GK_GFX90A:
  case llvm::AMDGPU::GK_GFX928:
  case llvm::AMDGPU::GK_GFX936:
    return ISAFamily::CDNA2;
  case llvm::AMDGPU::GK_GFX908:
    return ISAFamily::CDNA1;
  default:
    break;
  }

  // RNDA ISA cases
  if (kind >= llvm::AMDGPU::GK_GFX1100 && kind <= llvm::AMDGPU::GK_GFX1201)
    return ISAFamily::RDNA3;
  if (kind >= llvm::AMDGPU::GK_GFX1030 && kind <= llvm::AMDGPU::GK_GFX1036)
    return ISAFamily::RDNA2;
  if (kind >= llvm::AMDGPU::GK_GFX1010 && kind <= llvm::AMDGPU::GK_GFX1013)
    return ISAFamily::RDNA1;

  return ISAFamily::Unknown;
}

} // namespace mlir::triton::HCU
