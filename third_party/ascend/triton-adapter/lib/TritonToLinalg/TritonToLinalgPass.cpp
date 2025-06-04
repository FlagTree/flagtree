#include "TritonToLinalg/TritonToLinalgPass.h"
#include "TritonToLinalg/ArgMinMaxConverter.h"
#include "TritonToLinalg/FunctionConverter.h"
#include "TritonToLinalg/LoadStoreConverter.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "TritonToLinalg/UseAnalysis.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

TritonTypeConverter::TritonTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion([](triton::PointerType ptrType) {
    return MemRefType::get({ShapedType::kDynamic}, ptrType.getPointeeType());
  });

  addConversion([](TensorType tensorType) -> Type {
    auto elemType = tensorType.getElementType();
    if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
      elemType = ptrType.getPointeeType();
    }
    return MemRefType::get(tensorType.getShape(), elemType);
  });
}

void TritonToLinalgPass::addProgramInfo(triton::FuncOp func,
                                        bool globalKernel) {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newFuncType =
      b.getFunctionType(newInputTypes, origFuncType.getResults());

  func.setFunctionType(newFuncType);

  // 如果需要，给参数新增属性
  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  // 添加对应参数到函数体中
  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  }

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}

void TritonToLinalgPass::convertTTFunc(triton::FuncOp func,
                                       const bool existDot) {
  OpBuilder builder(func);

  auto name = func.getName();
  auto type = func.getFunctionType();

  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  func.getAllArgAttrs(argAttrs);
  func.getAllResultAttrs(resAttrs);

  // bit-casted tt.ptr的特殊处理
  SmallVector<Type> inputTypes{type.getInputs()};
  SmallVector<Type> retTypes{type.getResults()};
  if (func.getSymVisibility() == "public" && !func.isDeclaration()) {
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      auto arg = func.getArgument(i);
      if (!isa<BaseMemRefType>(arg.getType())) {
        continue;
      }
      // FIXME: Why arg.getUsers() cannot return the user inside scf.for?
      llvm::SmallVector<Operation *, 16> arg_users;
      func.walk([&](Operation *op) {
        if (op->use_empty()) {
          return WalkResult::advance();
        }
        for (auto operand : op->getOperands()) {
          if (operand == arg) {
            arg_users.push_back(op);
          }
        }
        return WalkResult::advance();
      });

      bool arg_use_empty = arg_users.size() == 0;
      if (!arg_use_empty) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : arg_users) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        if (llvm::all_of(arg_users, [](Operation *userOp) {
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          auto castOp = cast<UnrealizedConversionCastOp>(*arg_users.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
        }
      } else {
        // Process unused bool ptr type specially, which guarantees bool pointer
        // argument's type is realistic and don't mislead backend compiler.
        BaseMemRefType argType = dyn_cast<BaseMemRefType>(arg.getType());
        if (argType.getElementTypeBitWidth() == 1) {
          // realistic memory layout of bool pointer is 8 bit width
          auto memType = argType.cloneWith(std::nullopt, builder.getI8Type());
          arg.setType(memType);
          inputTypes[i] = arg.getType();
        }
      }
    }
  }
  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  auto kernelAttr = func->getAttr(globalKernelAttr);
  if (kernelAttr) {
    funcFunc->setAttr(globalKernelAttr, kernelAttr);
  }
  std::string kernelMixMode = "aiv";
  if (existDot) {
    // mix also works for pure cube kernel by using the same MAGIC_ELF keyword
    kernelMixMode = "mix";
  }
  // Set mix_mode in the func attrs so that the backend could know
  // the mix_mode by parse the func attrs.
  // The backend needs to know the mix_mode because the host wrapper
  // needs to set the devbin.magic. Check npu_utils.cpp.
  funcFunc->setAttr(kernelMixModeName, builder.getStringAttr(kernelMixMode));

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
  }
  func.erase();
}

void TritonToLinalgPass::addDynamicLegal(
    ConversionTarget &target, TritonTypeConverter &tritonTypeConverter) {
  target.addLegalDialect<
      func::FuncDialect, arith::ArithDialect, math::MathDialect,
      linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
      cf::ControlFlowDialect, tensor::TensorDialect,
      bufferization::BufferizationDialect, memref::MemRefDialect>();

  // add legal dialect on condition
  target.addLegalOp<ModuleOp>();

  // 根据条件判断需要转换的OP
  target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
      [](mlir::Operation *op) {
        if (op->use_empty()) {
          return false;
        } else {
          return true;
        }
      });

  target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
    return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
    auto res = op.getResult();
    if (!isa<RankedTensorType>(res.getType())) {
      return true;
    }

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (!denseAttr.isSplat() ||
          !isa<FloatType, IntegerType>(denseAttr.getElementType())) {
        return true;
      }
      if (res.hasOneUse() && isa<tensor::ReshapeOp>(*res.user_begin())) {
        return true;
      }
      return false;
    }
    return true;
  });

  target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type t) {
      if (isa<triton::PointerType>(t)) {
        return false;
      }
      if (auto shapedType = dyn_cast<ShapedType>(t)) {
        return shapedType.getElementType().isIntOrFloat();
      }
      assert(t.isIntOrIndexOrFloat());
      return true;
    });
  });

  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      [this](Operation *op) {
        if (op->hasAttr("MetaUse")) {
          return false;
        }

        if (isa<arith::ConstantOp>(op)) {
          return true;
        }

        bool operateOnTensors =
            llvm::all_of(op->getOperandTypes(),
                         [](Type type) { return isa<RankedTensorType>(type); });

        return this->namedOps || !operateOnTensors;
      });
}

void TritonToLinalgPass::populateTritonToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TTOpConverters::AssertConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::LoadStoreCanonicalizer<triton::LoadOp>,
               LoadStoreConverter::LoadStoreCanonicalizer<triton::StoreOp>>(
      patterns.getContext());
  patterns.add<TTOpConverters::SelectConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastCanonicalizer>(patterns.getContext());
  patterns.add<LoadStoreConverter::ScalarStoreCanonicalizer>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::ScalarAtomicRMWCanonicalizer>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicMaxMinCanonicalizer>(
      patterns.getContext());
  patterns.add<TTOpConverters::ScalarMathCanonicalizer<math::AbsFOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AcosOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AcoshOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AsinOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AsinhOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AtanOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::Atan2Op>,
               // TTOpConverters::ScalarMathCanonicalizer<math::AtanhOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::CeilOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::CosOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::CoshOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::ErfOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::ExpOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::Exp2Op>,
               // TTOpConverters::ScalarMathCanonicalizer<math::ExpM1Op>,
               TTOpConverters::ScalarMathCanonicalizer<math::FloorOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::FmaOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::LogOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::Log10Op>,
               // TTOpConverters::ScalarMathCanonicalizer<math::Log1pOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::Log2Op>,
               // TTOpConverters::ScalarMathCanonicalizer<math::PowFOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::RoundOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::RsqrtOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::SinOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::SinhOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::SqrtOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::TanOp>,
               TTOpConverters::ScalarMathCanonicalizer<math::TanhOp>,
               // TTOpConverters::ScalarMathCanonicalizer<math::TruncOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::AddFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::SubFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::MulFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::DivFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::NegFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::RemFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::MaxNumFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::MaximumFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::MinNumFOp>,
               TTOpConverters::ScalarMathCanonicalizer<arith::MinimumFOp>
               // By test, the following ops do not need canonicalization.
               // TTOpConverters::ScalarMathCanonicalizer<arith::CmpFOp>
               // TTOpConverters::ScalarMathCanonicalizer<arith::ExtFOp>
               // TTOpConverters::ScalarMathCanonicalizer<arith::TruncFOp>
               >(patterns.getContext());
}

void TritonToLinalgPass::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);

  patterns.add<triton::MetaUseEraser>(patterns.getContext());
  patterns.add<LoadStoreConverter::StoreConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AddPtrConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetProgramIDConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetNumProgramsConverter>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::LoadConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicRMWConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MakeRangeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ClampFConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PreciseDivConverter>(patterns.getContext());
  // reduce converters
  patterns.add<TTOpConverters::ArgMinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ArgMaxConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReduceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReshapeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExpandDimsConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BroadcastConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DenseConstantConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExternElementwiseClOpConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::TritonMulhiuiConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TritonPreciseSqrtConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<TTOpConverters::AdvanceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TransposeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplitConverter>(patterns.getContext());
  patterns.add<TTOpConverters::JoinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::CatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastConverter>(patterns.getContext());
  patterns.add<TTOpConverters::LoopConverter>(patterns.getContext());
  patterns.add<TTOpConverters::YieldConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MatmulConverter>(patterns.getContext());

  if (!this->namedOps) {
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
  }
}

void TritonToLinalgPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect>();
}

void TritonToLinalgPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Check if the kernel contains tl.dot. Without tl.dot,
  // the kernel would be pure AIV kernel.
  bool existDot = false;
  moduleOp.walk([&](triton::DotOp dotOp) {
    existDot = true;
    return WalkResult::interrupt();
  });

  RewritePatternSet canonicalizerPatterns(&getContext());
  // 1.标准化 LoadStore ScalarStoreCanonicalizer
  this->populateTritonToLinalgCanonicalizationPatterns(canonicalizerPatterns);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(canonicalizerPatterns)))) {
    moduleOp->emitError("failed to apply Canonicalizer Patterns");
    signalPassFailure();
  }

  // 2.使用分析
  moduleOp.walk([this](triton::FuncOp op) {
    if (failed(runUseAnalysis(op))) {
      signalPassFailure();
    }
  });

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  TritonTypeConverter tritonTypeConverter{};

  // 3.标注合法方言
  this->addDynamicLegal(target, tritonTypeConverter);

  // 5.对非法Op注册Converter
  this->populateTritonToLinalgConversionPatterns(tritonTypeConverter, patterns,
                                                 LAUNCH_GRID_RANK);

  // 6.遍历kernel中的function，修改program id、number of programs参数
  for (auto func : getOperation().getOps<triton::FuncOp>()) {
    addProgramInfo(func, globalKernel);
  }

  // 7.做Op转换
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Convertion Patterns");
    signalPassFailure();
  }

  // 8.函数头尾转换
  moduleOp.walk(
      [&](triton::FuncOp func) { this->convertTTFunc(func, existDot); });

  // 9.清除无效代码，简化代码。
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }

  // Try interleave optimization
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>> interleaveCandidate;
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>>
      interleaveCandidateWithMask;
  moduleOp.walk([&](bufferization::MaterializeInDestinationOp materializeOp) {
    if (auto reinterpretCastOp =
            materializeOp.getDest()
                .getDefiningOp<memref::ReinterpretCastOp>()) {
      if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
          reinterpretCastOp.getStaticStrides().back() == 2) {
        interleaveCandidate[llvm::cast<BlockArgument>(
                                reinterpretCastOp.getSource())]
            .push_back(materializeOp);
      }
    }

    // Difference is that converted op chain of store with mask has
    // `memref::SubViewOp`
    if (auto subviewOp =
            materializeOp.getDest().getDefiningOp<memref::SubViewOp>()) {
      if (!llvm::isa<tensor::ExtractSliceOp>(
              materializeOp.getSource().getDefiningOp()))
        return WalkResult::advance();

      if (auto reinterpretCastOp =
              subviewOp.getSource()
                  .getDefiningOp<memref::ReinterpretCastOp>()) {
        if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
            reinterpretCastOp.getStaticStrides().back() == 2) {
          interleaveCandidateWithMask[llvm::cast<BlockArgument>(
                                          reinterpretCastOp.getSource())]
              .push_back(materializeOp);
        }
      }
    }

    return WalkResult::advance();
  });

  for (auto [blockArg, materializeVec] : interleaveCandidate) {
    // Just enable optimization where exists double materializeOp with same
    // block argument destination.
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusOptimization(materializeVec);
  }

  for (auto [blockArg, materializeVec] : interleaveCandidateWithMask) {
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusWithMaskOptimization(materializeVec);
  }

  // Force to add an argument at the beginning of function arguments, which
  // represents stub arg for workspace. Default type is memref<?xi8>
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    if (!func->hasAttr("global_kernel"))
      continue;

    auto context = func.getContext();
    constexpr int64_t workspaceArgIdx = 0;
    MemRefType workspaceArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                    UnitAttr::get(context));

    func.insertArgument(/*argIndex*/ workspaceArgIdx,
                        /*argType*/ workspaceArgType,
                        /*dicAttr*/ nullptr, func->getLoc());
    func->setAttr("WorkspaceArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64), 0));
  }

  // Fix the Location info
  moduleOp.walk([&](Operation *op) {
    auto loc = op->getLoc();
    if (isa<UnknownLoc>(loc)) {
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [](Operation *curOp) { return false; },
          /*stopFn*/
          [](Operation *curOp) { return !isa<UnknownLoc>(curOp->getLoc()); },
          /*actionFn*/
          nullptr, stopOps);
      if (stopOps.empty()) {
        op->emitWarning() << *op << " and its users all have no location!";
      } else {
        Operation *goodOp = *stopOps.begin();
        op->setLoc(goodOp->getLoc());
      }
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
