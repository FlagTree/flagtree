//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_TO_MK_CONVERSION_PASSES_H
#define MEMREF_TO_MK_CONVERSION_PASSES_H

#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Tx81MemrefToLLVM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  MEMREF_TO_MK_CONVERSION_PASSES_H
