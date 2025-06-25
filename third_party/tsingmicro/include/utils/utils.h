//===------------------- utils.h ------------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Utility functions for ztc conversion.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_UTILS_H
#define ZTC_CONVERSION_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h" // Include the header for Type

using namespace mlir;

namespace mlir::triton::utils {
Value declareTx81Function(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes);
} // namespace mlir::triton::utils

#endif // ZTC_CONVERSION_UTILS_H
