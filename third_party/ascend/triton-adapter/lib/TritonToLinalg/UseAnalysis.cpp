//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TritonToLinalg/UseAnalysis.h"
#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;
using namespace dataflow;

#define DEBUG_TYPE "triton-use-analysis"

std::string stringifyUseType(UseType useTy) {
  std::string ret;
  if (useTy == UseType::MetaUse) {
    ret = "MetaUse";
  } else if (useTy == UseType::DataUse) {
    ret = "DataUse";
  } else if (useTy == UseType::MixUse) {
    ret = "MixUse";
  } else if (useTy == UseType::Undefined) {
    ret = "Undefined";
  }
  return ret;
}

#if LLVM_VERSION_MAJOR >= 20
LogicalResult
triton::UseAnalysis::visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                                    ArrayRef<const UseInfo *> results) {
#else
void triton::UseAnalysis::visitOperation(Operation *op,
                                         ArrayRef<UseInfo *> operands,
                                         ArrayRef<const UseInfo *> results) {
#endif

  if (op->getResults().size() == 1) {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (resultType && isa<triton::PointerType>(resultType.getElementType())) {
      for (auto opnd : operands) {
        propagateUse(opnd, UseType::MetaUse);
      }
    }
  }

  TypeSwitch<Operation *>(op)
      .Case<triton::LoadOp>([&](auto load) {
        propagateUse(operands[0], UseType::MetaUse);
        auto mask = load.getMask();
        auto other = load.getOther();
        if (mask) {
          assert(mask != other && "mask and other cannot be the same");
          propagateUse(operands[1], UseType::MetaUse);
        }
        if (other) {
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::StoreOp>([&](auto store) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        auto value = store.getValue();
        auto mask = store.getMask();
        if (mask) {
          assert(mask != value && "mask and data cannot be the same");
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      // Consider triton::AtomicRMWOp as store operation
      .Case<triton::AtomicRMWOp>([&](auto atomicOp) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        auto value = atomicOp.getVal();
        auto mask = atomicOp.getMask();
        if (mask) {
          assert(mask != value && "mask and data cannot be the same");
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::DotOp>([&](auto dot) {
        propagateResults(operands[0], results);
        propagateResults(operands[1], results);

        auto opc = dot.getC();
        triton::SplatOp splat;
        if (opc) {
          splat = opc.template getDefiningOp<triton::SplatOp>();
        }

        if (opc && splat && splat.getSrc().getDefiningOp<arith::ConstantOp>()) {
          propagateUse(operands[2], UseType::MetaUse);
        } else {
          propagateUse(operands[2], UseType::DataUse);
        }
      })
      .Default([&](Operation *op) {
        // this condition account for tt.addptr
        for (auto operand : operands) {
          propagateResults(operand, results);
        }
      });
#if LLVM_VERSION_MAJOR >= 20
  return success();
#endif
}

LogicalResult triton::runUseAnalysis(triton::FuncOp &funcOp) {
  MLIRContext *context = funcOp.getContext();
  SymbolTableCollection symbolTable;

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<UseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(funcOp))) {
    return failure();
  }
  auto &os = llvm::dbgs();
  // Walk the func op, convert tags on operands to tags on operations
  funcOp.walk([&](Operation *op) {
    LLVM_DEBUG({ os << "[UseAnalysis] op is " << *op << "\n"; });
    UseType useType = UseType::Undefined;
    for (auto result : op->getResults()) {
      LLVM_DEBUG({ os << "[UseAnalysis] ===> result is " << result << "\n"; });
      auto use = solver.lookupState<UseInfo>(result);
      assert(use && "Lattice value not found");
      auto thisUseType = use->type;
      LLVM_DEBUG({
        os << "[UseAnalysis] ==========> useType is "
           << stringifyUseType(thisUseType) << "\n";
      });
      if (thisUseType == UseType::Undefined) {
        continue;
      }
      if (useType == UseType::Undefined) {
        useType = thisUseType;
      }
      if (thisUseType == UseType::MixUse || thisUseType != useType) {
        useType = UseType::MixUse;
        break;
      }
    }

    if (useType == UseType::Undefined) {
      LLVM_DEBUG({ op->setAttr("Undefined", UnitAttr::get(context)); });
      return;
    } else if (useType == UseType::MetaUse) {
      if (!isa<mlir::scf::IfOp>(op)) {
        assert(op->getNumResults() == 1 &&
               "Ops used for meta computation are expected to have one result");
      }
      for (auto it = 0; it < op->getNumResults(); ++it) {
        // Only set the tag if the operation uses tensors
        if (isa<ShapedType>(op->getResult(it).getType()) ||
            (isa<triton::BitcastOp>(op) &&
             isa<PointerType>(op->getResult(it).getType()))) {
          // Setting tag for erasing op later
          op->setAttr("MetaUse", UnitAttr::get(context));
        }
      }
      return;
    } else if (useType == UseType::DataUse) {
      LLVM_DEBUG({ op->setAttr("DataUse", UnitAttr::get(context)); });
      return;
    }

    assert(useType == UseType::MixUse);

    // If the operation only produces scalars, no need to clone it
    bool shapedResult = true;
    for (auto result : op->getResults())
      shapedResult &= isa<ShapedType>(result.getType());
    if (!shapedResult) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }

    llvm::SetVector<Operation *> metaUsers;
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<triton::LoadOp>([&](auto load) {
              auto ptr = load.getPtr();
              auto mask = load.getMask();
              auto other = load.getOther();
              if (result == ptr || result == mask || result == other) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::StoreOp>([&](auto store) {
              auto ptr = store.getPtr();
              auto mask = store.getMask();
              if (result == ptr || result == mask) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::AtomicRMWOp>([&](auto atomicOp) {
              auto ptr = atomicOp.getPtr();
              auto mask = atomicOp.getMask();
              if (result == ptr || result == mask)
                metaUsers.insert(user);
            })
            .Case<triton::DotOp>([&](auto dot) {
              auto opc = dot.getC();
              triton::SplatOp splat;
              if (opc) {
                splat = opc.template getDefiningOp<triton::SplatOp>();
              }

              if (opc && splat &&
                  splat.getSrc().getDefiningOp<arith::ConstantOp>()) {
                metaUsers.insert(user);
              }
            })
            .Default([&](Operation *op) {
              bool allMeta = true;
              for (auto res : op->getResults()) {
                auto resUse = solver.lookupState<UseInfo>(res);
                if (resUse->type != UseType::MetaUse) {
                  allMeta = false;
                  break;
                }
              }
              if (allMeta) {
                metaUsers.insert(user);
              }
            });
      }
    }

    // If the operation doesn't have direct meta users, no need to clone it
    if (metaUsers.empty()) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }

    // Clone the operation; switch all meta users to use the clone
    OpBuilder builder(op);
    auto clone = builder.clone(*op);
    LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });

    // Setting tag for erasing op later
    clone->setAttr("MetaUse", UnitAttr::get(context));

    for (auto [res_i, result] : llvm::enumerate(op->getResults())) {
      for (auto user : metaUsers) {
        for (auto &operand : user->getOpOperands()) {
          if (operand.get() == result) {
            operand.set(clone->getResult(res_i));
          }
        }
      }
    }
  });
  LLVM_DEBUG({
    os << "[UseAnalysis] Before post-process, funcOp is " << *funcOp << "\n";
  });
  // Post-process
  funcOp.walk([&](Operation *op) {
    // Handle indirect load case.
    // For example, load(1st) -> computeOp -> load(2nd).
    // The first load is IndirectLoadInterfaceOp.
    // Do not inplace replace MetaUse by MixUse. Because the condition checking
    // depends on that the op has the attr of MetaUse.
    // Handle the indirect load interface op
    // We first trace from the 1st load to the 2nd load with the ops between
    // them marked as MixUse. Then we traceback from the 2nd load to mark defs
    // MixUse.
    if (opIsIndirectLoad(op) || opIsIndirectCalc(op)) {
      LLVM_DEBUG({
        os << "[UseAnalysis] Found indirect load interface op: " << *op << "\n";
      });
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      // Modify the users of this op's result.
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [op](Operation *curOp) { return isMetaUse(curOp) && curOp != op; },
          /*stopFn*/
          [&](Operation *curOp) {
            // triton::LoadOp without MetaUse means it is an indirect load
            // instead of the load providing the offset.
            // The pattern is as follows,
            // load -> ops -> load
            // We need to ensure the intermediate ops are marked MixUse
            // so that they will be replaced instead of be erased without
            // conversion.
            return isa<triton::LoadOp>(curOp) && !curOp->hasAttr("MetaUse");
          },
          /*actionFn*/
          [](OpBuilder &b, Operation *op) {
            op->setAttr("MixUse", UnitAttr::get(op->getContext()));
          },
          stopOps);
      LLVM_DEBUG({
        os << "[UseAnalysis] stopOps are \n";
        int i = 0;
        for (auto it = stopOps.begin(); it != stopOps.end(); it++) {
          os << i++ << ": " << *(*it) << "\n";
        }
      });
      LLVM_DEBUG({
        os << "[UseAnalysis] After trace, funcOp is " << *funcOp << "\n";
      });
      for (auto it = stopOps.begin(); it != stopOps.end(); it++) {
        auto stopOp = *it;
        traverseBackwardUpdateOperandChainIf(
            stopOp,
            [stopOp](Operation *curOp) {
              return isMetaUse(curOp) && curOp != stopOp;
            },
            [](OpBuilder &b, Operation *op) {
              op->setAttr("MixUse", UnitAttr::get(op->getContext()));
            });
      }
      LLVM_DEBUG({
        os << "[UseAnalysis] After traceback of stopOp, funcOp is " << *funcOp
           << "\n";
      });
      // Modify this op.
      op->setAttr("MixUse", UnitAttr::get(op->getContext()));
    }
  });
  // Remove MetaUse in case of MixUse existing in the op
  funcOp.walk([&](Operation *op) {
    if (isMetaUse(op) && isMixUse(op)) {
      op->removeAttr("MetaUse");
    }
  });
  LLVM_DEBUG({
    os << "[UseAnalysis] After post-process, funcOp is " << *funcOp << "\n";
  });
  return success();
}

MetaUseEraser::MetaUseEraser(MLIRContext *context)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

LogicalResult MetaUseEraser::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (isa<triton::AddPtrOp>(op)) {
    return rewriter.notifyMatchFailure(op,
                                       "AddPtrOp will be handled separately");
  }
  if (isMetaUse(op)) {
    rewriter.eraseOp(op);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "requires meta ops");
}
