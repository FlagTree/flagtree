//===- KernelArgBufferPass.cpp - Convert kernel args to single buffer -----===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms kernel function signatures by converting multiple
// arguments into a single void* buffer containing all the arguments.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

class KernelArgBufferPass
    : public PassWrapper<KernelArgBufferPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "kernel-arg-buffer"; }
  StringRef getDescription() const final {
    return "Convert kernel arguments to a single buffer argument";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  void runOnOperation() override;

private:
  // Identifies if a function should be processed
  bool isKernelFunction(func::FuncOp func);

  // Creates a new function with a single void* argument
  func::FuncOp createBufferizedFunction(OpBuilder &builder,
                                        func::FuncOp originalFunc);

  // Rewrites the function body to use the argument buffer
  void rewriteFunctionBody(func::FuncOp originalFunc, func::FuncOp newFunc);
};

bool KernelArgBufferPass::isKernelFunction(func::FuncOp func) {
  // For this example, we'll identify kernel functions by their name
  // containing "_kernel". In a real implementation, you might use attributes
  // or more sophisticated detection.
  return func.getName().contains("_kernel");
}

func::FuncOp
KernelArgBufferPass::createBufferizedFunction(OpBuilder &builder,
                                              func::FuncOp originalFunc) {
  // Create a new function type with a single void* argument
  auto voidPtrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto newFuncType =
      FunctionType::get(originalFunc.getContext(), {voidPtrType},
                        originalFunc.getFunctionType().getResults());

  // Create the new function with the same name but new type
  auto newFunc = func::FuncOp::create(originalFunc.getLoc(),
                                      originalFunc.getName(), newFuncType);

  // Copy over all attributes except those related to the function type
  for (const auto &attr : originalFunc->getAttrs()) {
    if (attr.getName() != "function_type" && attr.getName() != "arg_attrs" &&
        attr.getName() != "res_attrs") {
      newFunc->setAttr(attr.getName(), attr.getValue());
    }
  }

  return newFunc;
}

void KernelArgBufferPass::rewriteFunctionBody(func::FuncOp originalFunc,
                                              func::FuncOp newFunc) {
  if (originalFunc.empty())
    return;

  Block &oldEntryBlock = originalFunc.getBlocks().front();
  Block &newEntryBlock = newFunc.getBlocks().front();

  OpBuilder builder(&newEntryBlock, newEntryBlock.begin());
  Location loc = originalFunc.getLoc();

  Value argsBuffer = newEntryBlock.getArgument(0);
  SmallVector<Value, 8> extractedArgs;

  // Offset tracking for buffer access
  int64_t currentOffset = 0;
  // Size of scalar values in bytes (specified as 8 bytes)
  const int64_t scalarSize = 8;

  // Process each original argument
  for (auto argIndex : llvm::seq<unsigned>(0, originalFunc.getNumArguments())) {
    Type argType = originalFunc.getArgument(argIndex).getType();
    Value loadedArg;

    // Handle pointer types (like uint64_t*)
    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(argType)) {
      // For pointer types, we load the pointer value itself from the buffer
      auto offsetValue = builder.create<LLVM::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(currentOffset));

      // Get pointer to the current position in args buffer
      auto elementPtr = builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), argsBuffer,
          ArrayRef<Value>{offsetValue});

      // Cast to pointer-to-pointer type
      auto castedPtr = builder.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(ptrType), elementPtr);

      // Load the pointer
      loadedArg = builder.create<LLVM::LoadOp>(loc, castedPtr);

      // Increment offset (pointers are 8 bytes)
      currentOffset += scalarSize;
    }
    // Handle scalar types (like int64_t, int)
    else {
      auto offsetValue = builder.create<LLVM::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(currentOffset));

      // Get pointer to the current position in args buffer
      auto elementPtr = builder.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), argsBuffer,
          ArrayRef<Value>{offsetValue});

      // Cast to appropriate pointer type
      auto castedPtr = builder.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(argType), elementPtr);

      // Load the scalar value
      loadedArg = builder.create<LLVM::LoadOp>(loc, castedPtr);

      // Increment offset (all scalars use 8 bytes as specified)
      currentOffset += scalarSize;
    }

    extractedArgs.push_back(loadedArg);
  }

  // Clone the original function body, replacing uses of old arguments
  auto &oldRegion = originalFunc.getBody();
  auto &newRegion = newFunc.getBody();

  // Move operations from old entry block to new entry block
  for (auto &op : oldEntryBlock.getOperations()) {
    if (&op == &oldEntryBlock.back() && op.hasTrait<OpTrait::IsTerminator>()) {
      builder.clone(op);
    } else {
      auto clonedOp = builder.clone(op);

      // Replace uses of old arguments with new extracted values
      for (unsigned i = 0; i < originalFunc.getNumArguments(); ++i) {
        Value oldArg = oldEntryBlock.getArgument(i);
        clonedOp->replaceUsesOfWith(oldArg, extractedArgs[i]);
      }
    }
  }
}

void KernelArgBufferPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Collect functions to process
  SmallVector<func::FuncOp, 4> kernelFuncs;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (isKernelFunction(func)) {
      kernelFuncs.push_back(func);
    }
  }

  // Process each kernel function
  for (auto func : kernelFuncs) {
    // Create new function with bufferized signature
    builder.setInsertionPointAfter(func);
    auto newFunc = createBufferizedFunction(builder, func);

    // Add entry block to the new function
    newFunc.addEntryBlock();

    // Rewrite function body to use the argument buffer
    rewriteFunctionBody(func, newFunc);

    // Replace the old function with the new one
    func.erase();
  }
}

} // namespace

std::unique_ptr<Pass> createKernelArgBufferPass() {
  return std::make_unique<KernelArgBufferPass>();
}

// Pass registration
namespace {
#define GEN_PASS_REGISTRATION
#include "KernelArgBufferPass.h.inc"
} // namespace
