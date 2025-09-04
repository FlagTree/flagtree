#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "third_party/hcu/include/triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;

void collectDependentOps(Operation *op,
                         std::vector<Operation *> &dependentOps) {
  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      int count = std::count(dependentOps.begin(), dependentOps.end(), defOp);
      if (defOp->getBlock() == op->getBlock() && count <= 0) {
        collectDependentOps(defOp, dependentOps);
        dependentOps.push_back(defOp);
      }
    }
  }
}

void moveOpWithDependencies(Operation *op, Operation *target,
                            std::set<Operation *> &moveOps) {
  std::vector<Operation *> dependentOps;
  collectDependentOps(op, dependentOps);

  Block *block = target->getBlock();

  for (Operation *depOp : dependentOps) {
    if (moveOps.count(depOp) <= 0) {
      depOp->moveBefore(target);
      moveOps.insert(depOp);
    }
  }

  op->moveBefore(target);
}

bool isDependentOnTarget(Operation *op, Operation *target) {
  for (auto operand : op->getOperands()) {
    if (operand.getDefiningOp() == target) {
      return true;
    }
  }

  std::vector<Operation *> dependentOps;
  collectDependentOps(op, dependentOps);

  for (auto op1 : dependentOps) {
    for (auto operand : op1->getOperands()) {
      if (operand.getDefiningOp() == target) {
        return true;
      }
    }
  }
  return false;
}

// 定义比较函数
static bool
compareMaps(const std::map<Operation *, std::pair<Operation *, int>> &map1,
            const std::map<Operation *, std::pair<Operation *, int>> &map2) {
  // 找到map1和map2中的最大int值
  int max1 =
      std::max_element(
          map1.begin(), map1.end(),
          [](const std::pair<Operation *, std::pair<Operation *, int>> &p1,
             const std::pair<Operation *, std::pair<Operation *, int>> &p2) {
            return p1.second.second < p2.second.second;
          })
          ->second.second;

  int max2 =
      std::max_element(
          map2.begin(), map2.end(),
          [](const std::pair<Operation *, std::pair<Operation *, int>> &p1,
             const std::pair<Operation *, std::pair<Operation *, int>> &p2) {
            return p1.second.second < p2.second.second;
          })
          ->second.second;

  // 返回比较结果
  return max1 > max2;
}

#define GEN_PASS_CLASSES
#include "TritonHCUTransforms/Passes.h"

class TritonHCUMoveLoadToFrontOfDOTPass
    : public TritonHCUMoveLoadToFrontOfDOTBase<
          TritonHCUMoveLoadToFrontOfDOTPass> {
public:
  TritonHCUMoveLoadToFrontOfDOTPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](Operation *op) {
      // 检查操作是否为 BranchOp
      if (isa<cf::CondBranchOp>(op)) {
        // 这里是你找到 BranchOp 后的处理代码
        auto cop = dyn_cast<cf::CondBranchOp>(op);
        auto block = cop.getTrueDest(); // 获取到 tt.dot 所在的 block

        SmallVector<Operation *> dots;  // dot
        SmallVector<int> dots_num;      // tt.load, tt.reduce 中的 scale load
        SmallVector<Operation *> loads; // tt.load
        SmallVector<int> loads_num;     // tt.load 中的 scale load
        SmallVector<Operation *> reduces; // tt.reduce

        int cout = 0;
        block->walk([&](Operation *op) {
          if (isa<triton::DotOp>(op)) {
            dots.push_back(op);
            dots_num.push_back(cout++);
          }
          if (isa<triton::LoadOp, triton::hcugpu::BufferLoadOp,
                  triton::gpu::AsyncCopyGlobalToLocalOp>(op)) {
            loads.push_back(op);
            loads_num.push_back(cout++);
          }

          if (isa<triton::ReduceOp>(op)) {
            reduces.push_back(op);
          }
        });

        // 判断 load 是否在 dot 之前
        int index = 0;
        loads.erase(std::remove_if(loads.begin(), loads.end(),
                                   [this, dots, &index, loads_num,
                                    dots_num](Operation *load) {
                                     if (dots.size() == 1) {
                                       if (loads_num[index] < dots_num[0]) {
                                         return true;
                                       }
                                     }
                                     index++;
                                     return false;
                                   }),
                    loads.end());

        // 判断有依赖就不移动
        loads.erase(std::remove_if(loads.begin(), loads.end(),
                                   [this, dots](Operation *load) {
                                     bool flag = false;
                                     for (auto dot : dots) {
                                       if (isDependentOnTarget(dot, load)) {
                                         flag = true;
                                       }
                                     }
                                     return flag;
                                   }),
                    loads.end());

        /*
            移除一些不是 dot 指令直接相关的 load
            通过 dot 两个参数中的形状来判断是否是与 dot 指令直接相关的 load
           操作。 (反量化因子相关load指令, 目前只有 loads 里面仅有scales的时候，
           把scales相关的load去除掉，如果既有 dot load 又有 scales load
           那就不消除)
        */
        // if(loads.size() != 4)
        loads.erase(
            std::remove_if(
                loads.begin(), loads.end(),
                [this, dots](Operation *load) {
                  bool flag = true;
                  for (auto op : dots) {
                    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
                      RankedTensorType aType =
                          cast<RankedTensorType>(dotOp.getA().getType());
                      RankedTensorType bType =
                          cast<RankedTensorType>(dotOp.getB().getType());
                      // a type, shape
                      Type aElementType = aType.getElementType();
                      ArrayRef<int64_t> aShape = aType.getShape();
                      // b type, shape
                      Type bElementType = bType.getElementType();
                      ArrayRef<int64_t> bShape = bType.getShape();

                      RankedTensorType loadType;
                      Type loadValueTy;
                      if (auto loadOp = dyn_cast<triton::LoadOp>(load)) {
                        loadValueTy = loadOp.getType();
                      } else if (auto bufferLoadOp =
                                     dyn_cast<triton::hcugpu::BufferLoadOp>(
                                         load)) {
                        loadValueTy = bufferLoadOp.getType();
                      } else if (auto asyncCopyOp = dyn_cast<
                                     triton::gpu::AsyncCopyGlobalToLocalOp>(
                                     load)) {
                        loadValueTy =
                            asyncCopyOp.getType(); // todo 这里没测试过
                      } else {
                        continue; // 如果不是以上任何类型，跳过
                      }
                      if (auto tensorType =
                              dyn_cast<RankedTensorType>(loadValueTy)) {
                        // load type, shape
                        Type loadElementType = tensorType.getElementType();
                        ArrayRef<int64_t> loadShape = tensorType.getShape();
                        auto getElementBitWidth =
                            [](Type elementType) -> std::optional<unsigned> {
                          if (auto intType =
                                  dyn_cast<IntegerType>(elementType)) {
                            return intType.getWidth();
                          } else if (auto floatType =
                                         dyn_cast<FloatType>(elementType)) {
                            return floatType.getWidth();
                          }
                          return std::nullopt; // 对于不支持的类型返回空值
                        };

                        if ((getElementBitWidth(loadElementType) ==
                                 getElementBitWidth(aElementType) &&
                             loadShape == aShape) ||
                            (getElementBitWidth(loadElementType) ==
                                 getElementBitWidth(bElementType) &&
                             loadShape == bShape))
                          flag = false;
                      }
                    }
                  }
                  return flag;
                }),
            loads.end());

        // 判断 loadOp 的操作数是否与 dotOp
        // 有依赖关系，如果没有依赖关系的话，就可以将 loadOp 相关的代码移动到
        // dot 前面
        SmallVector<std::map<Operation *, std::pair<Operation *, int>>> v;

        for (auto loadOp : loads) {
          std::map<Operation *, std::pair<Operation *, int>> myMap;
          int inc = -1;
          myMap[loadOp] = std::make_pair(nullptr, inc);
          for (auto dotOp = dots.rbegin(); dotOp != dots.rend(); ++dotOp) {
            inc++;
            // 判断 loadOp 是否与 dotOp 有依赖关系
            if (!isDependentOnTarget(loadOp, *dotOp)) {
              myMap[loadOp] = std::make_pair(*dotOp, inc);
            } else {
              break;
            }
          }
          v.push_back(myMap);
        }
        std::stable_sort(v.begin(), v.end(), compareMaps);

        //  deal fa casual = True
        if (loads.size() == 2 && dots.size() == 2 && reduces.size() == 2) {
          v.clear();
          std::map<Operation *, std::pair<Operation *, int>> myMap;
          std::map<Operation *, std::pair<Operation *, int>> myMap1;
          myMap[loads[0]] = std::make_pair(dots[0], 0);
          myMap1[loads[1]] = std::make_pair(dots[1], 1);
          v.push_back(myMap);
          v.push_back(myMap1);
        }

        std::set<Operation *> moveOps;
        for (auto map1 : v) {
          for (const auto &pair : map1) {
            auto &pairDot = pair.second;
            if (pairDot.first)
              moveOpWithDependencies(pair.first, pairDot.first, moveOps);
          }
        }
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonHCUMoveLoadToFrontOfDOTPass() {
  return std::make_unique<TritonHCUMoveLoadToFrontOfDOTPass>();
}
