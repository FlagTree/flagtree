#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace aipu {

#define GEN_PASS_DEF_AIPUCONVERTBOOLARG2I8
#include "Passes/Passes.h.inc"

struct AIPUConvertBoolArg2I8Pass
    : public impl::AIPUConvertBoolArg2I8Base<AIPUConvertBoolArg2I8Pass> {

private:
  bool isBoolType(Type type) const {
    if (auto memref = dyn_cast<MemRefType>(type)) {
      type = memref.getElementType();
    } else if (auto unranked = dyn_cast<UnrankedMemRefType>(type)) {
      type = unranked.getElementType();
    }
    return type.isInteger(1);
  }

  Type createI8Type(Type origType) {
    if (auto memref = dyn_cast<MemRefType>(origType)) {
      return MemRefType::get(memref.getShape(),
                             IntegerType::get(origType.getContext(), 8),
                             memref.getLayout(), memref.getMemorySpace());
    } else if (auto unranked = dyn_cast<UnrankedMemRefType>(origType)) {
      return UnrankedMemRefType::get(IntegerType::get(origType.getContext(), 8),
                                     unranked.getMemorySpace());
    } else if (origType.isInteger(1)) {
      return IntegerType::get(origType.getContext(), 8);
    }
    return origType;
  }

  void modifyRelatedOps(BlockArgument arg, Type originType, OpBuilder builder) {
    builder.setInsertionPointToStart(arg.getParentBlock());
    if (originType.isInteger(1)) {
      auto trunci =
          builder.create<arith::TruncIOp>(arg.getLoc(), originType, arg);
      arg.replaceAllUsesExcept(trunci, trunci);
    } else {
      auto attr = builder.getNamedAttr("from_pass_convert_bool_arg",
                                       builder.getBoolAttr(true));
      auto origin_arg = builder.create<UnrealizedConversionCastOp>(
          arg.getLoc(), originType, arg, attr);
      arg.replaceAllUsesExcept(origin_arg.getResult(0), origin_arg);
    }
  }

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (funcOp.getBody().empty())
      return;
    Block &block = funcOp.getBody().front();
    bool modified = false;
    SmallVector<Type> newInputTypes;
    OpBuilder builder(&getContext());

    for (BlockArgument arg : block.getArguments()) {
      if (auto type = arg.getType(); isBoolType(type)) {
        Type newType = createI8Type(type);
        arg.setType(newType);
        newInputTypes.push_back(newType);
        modified = true;
        modifyRelatedOps(arg, type, builder);
      } else {
        newInputTypes.push_back(arg.getType());
      }
    }

    if (modified) {
      FunctionType funcType = funcOp.getFunctionType();
      FunctionType newFuncType = FunctionType::get(
          funcOp.getContext(), newInputTypes, funcType.getResults());
      funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));
    }
  }
};

} // namespace aipu

} // namespace mlir
