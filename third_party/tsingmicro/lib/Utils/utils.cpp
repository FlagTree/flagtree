//===------------------- utils.cpp ----------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "utils/utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::utils {

// Function to declare Tx81 runtime function
Value declareTx81Function(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes) {
  // Check if the function already exists
  Operation *funcOp = module.lookupSymbol(name);
  if (funcOp)
    return builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), name);

  // Create function type
  Type funcType = LLVM::LLVMFunctionType::get(resultType, argumentTypes,
                                              /*isVarArg=*/false);

  // Create a function declaration
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(module.getBody());

  builder.create<LLVM::LLVMFuncOp>(loc, name, funcType,
                                   LLVM::Linkage::External);

  builder.restoreInsertionPoint(ip);

  // Return function pointer
  return builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), name);
}

} // namespace mlir::triton::utils
