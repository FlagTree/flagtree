//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULOOPGRID
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPULoopGrid
    : public impl::TritonXPULoopGridBase<TritonXPULoopGrid> {

  using impl::TritonXPULoopGridBase<TritonXPULoopGrid>::TritonXPULoopGridBase;

  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT = 3;

  Value ceilDiv(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
    auto c1 = builder.create<arith::ConstantIntOp>(loc, 1, lhs.getType());
    auto sub = builder.create<arith::SubIOp>(loc, rhs, c1);
    auto add = builder.create<arith::AddIOp>(loc, lhs, sub);
    auto div = builder.create<arith::DivSIOp>(loc, add, rhs);
    return div.getResult();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp func) {
      OpBuilder b(func);

      auto i32Ty = b.getI32Type();
      auto origFuncType = func.getFunctionType();
      auto origInputTypes = origFuncType.getInputs();
      SmallVector<Type> newInputTypes(origInputTypes.begin(),
                                      origInputTypes.end());
      newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, i32Ty);

      auto newFuncType =
          b.getFunctionType(newInputTypes, origFuncType.getResults());

      func.setType(newFuncType);

      SmallVector<Operation *> operationsToMove;
      SmallVector<Operation *> terminatorsToErase;

      for (auto &body : func.getBlocks()) {
        for (auto &op : body.getOperations()) {
          if (&op == body.getTerminator()) {
            terminatorsToErase.push_back(&op);
          } else {
            operationsToMove.push_back(&op);
          }
        }
      }

      auto &body = func.getBody().front();
      for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
        body.addArgument(i32Ty, func.getLoc());
      }
      b.setInsertionPoint(&body, body.begin());
      auto loc = b.getUnknownLoc();
      auto idxTy = b.getIndexType();
      auto argIdx = func.getNumArguments() - TRITON_PROGRAM_INFO_ARG_COUNT;
      auto idxCluster = b.create<triton::xpu::GetClusterIdOp>(loc, i32Ty);
      auto numCluster = b.create<triton::xpu::GetNumClusterOp>(loc, i32Ty);
      auto gridX = func.getArgument(argIdx + 0);
      auto gridY = func.getArgument(argIdx + 1);
      auto gridZ = func.getArgument(argIdx + 2);
      auto gridXY = b.create<arith::MulIOp>(loc, gridX, gridY);
      auto gridXYZ = b.create<arith::MulIOp>(loc, gridXY, gridZ);
      auto numProgramsPerCluster = ceilDiv(b, loc, gridXYZ, numCluster);
      auto lower = b.create<arith::IndexCastOp>(loc, idxTy, idxCluster);
      auto upper = b.create<arith::IndexCastOp>(loc, idxTy, gridXYZ);
      auto step = b.create<arith::IndexCastOp>(loc, idxTy, numCluster);
      auto loopGrid = b.create<scf::ForOp>(loc, lower, upper, step);

      for (auto op : operationsToMove) {
        op->moveBefore(loopGrid.getBody()->getTerminator());
      }
      for (Operation *term : terminatorsToErase) {
        term->erase();
      }
      SmallVector<mlir::Block *> blocksToDelete;
      for (auto &bodyBlock : func.getBlocks()) {
        if (&bodyBlock != &func.getBody().front()) {
          blocksToDelete.push_back(&bodyBlock);
        }
      }
      for (mlir::Block *block : blocksToDelete) {
        block->erase();
      }
      b.setInsertionPointAfter(loopGrid);
      b.create<triton::ReturnOp>(loc);

      b.setInsertionPointToStart(loopGrid.getBody());
      Value index =
          b.create<arith::IndexCastOp>(loc, i32Ty, loopGrid.getInductionVar());
      auto pidZ = b.create<arith::RemSIOp>(loc, index, gridZ);
      index = b.create<arith::DivSIOp>(loc, index, gridZ);
      auto pidY = b.create<arith::RemSIOp>(loc, index, gridY);
      auto pidX = b.create<arith::DivSIOp>(loc, index, gridY);

      SmallVector<Value, 4> programId{pidX, pidY, pidZ};
      func.walk([&](triton::GetProgramIdOp op) {
        op.replaceAllUsesWith(programId[op.getAxisAsInt()]);
      });
      func.walk([&](triton::GetNumProgramsOp op) {
        op.replaceAllUsesWith(func.getArgument(argIdx + op.getAxisAsInt()));
      });
      func.walk([&](XPUPrintOp op) {
        OpBuilder replacer(op);
        Value outerIdx = replacer.create<arith::ExtSIOp>(
            op.getLoc(), replacer.getI64Type(), index);
        op->setOperand(3, outerIdx);
      });
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
