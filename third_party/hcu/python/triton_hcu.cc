#include "triton/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUTransforms/Passes.h"
#include "TritonHCUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/TargetParser.h"
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace {
const char *const hcuTargetTriple = "amdgcn-amd-amdhsa";

void init_triton_hcu_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(createConvertTritonHCUGPUToLLVMPass(arch, ftz));
        });
  m.def("add_builtin_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(createConvertBuiltinFuncToLLVMPass());
  });
  m.def("insert_instruction_sched_hints", [](mlir::PassManager &pm) {
    pm.addPass(createInsertInstructionSchedHintsPass());
  });
  m.def("lower_instruction_sched_hints",
        [](mlir::PassManager &pm, std::string variant) {
          pm.addPass(createLowerInstructionSchedHintsPass(variant));
        });
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm,
                                                    const std::string &arch) {
    pm.addPass(
        mlir::triton::HCU::createDecomposeUnsupportedConversionsPass(arch));
  });
  ADD_PASS_WRAPPER_2("add_optimize_lds_usage",
                     mlir::triton::HCU::createOptimizeLDSUsagePass,
                     const std::string &, int32_t);
  ADD_PASS_WRAPPER_0("add_hcu_accelerate_flash_attention",
                     mlir::createTritonHCUAccelerateFlashAttentionPass);
  ADD_PASS_WRAPPER_0("add_global_copy_local_swizzle",
                    mlir::createTritonHCUGlobalToLocalSwizzlePass);
  ADD_PASS_WRAPPER_0("add_move_load_tofront_dot",
                    mlir::createTritonHCUMoveLoadToFrontOfDOTPass);
  ADD_PASS_WRAPPER_5("add_accelerate_matmul",
                     mlir::createTritonHCUGPUAccelerateMatmulPass,
                     const std::string, int, int, int, int);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonHCUGPUOptimizeEpiloguePass);
  ADD_PASS_WRAPPER_0("add_canonicalize_pointers",
                     mlir::createTritonHCUGPUCanonicalizePointersPass);
  ADD_PASS_WRAPPER_0("add_convert_to_buffer_ops",
                     mlir::createTritonHCUGPUConvertToBufferOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     mlir::createTritonHCUGPUReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_stream_pipeline",
                     mlir::createTritonHCUGPUStreamPipelinePass);
  ADD_PASS_WRAPPER_1("add_stream_pipelinev2",
                     mlir::createTritonHCUGPUStreamPipelineV2Pass, int);
  ADD_PASS_WRAPPER_4("add_hcu_stream_pipeline",
                     mlir::createTritonHCUStreamPipelinePass, int, int, int, bool);   
  ADD_PASS_WRAPPER_1("add_control_fa_fwd_bufferload_cnt",
                     mlir::createTritonHCUFaFwdControlCntPass, int);
  ADD_PASS_WRAPPER_1("add_fa_fwd_insert_wait",
                    mlir::createTritonHCUFaFwdWaitPass, int);
  ADD_PASS_WRAPPER_0("add_update_async_wait_count",
                    mlir::createTritonHCUUpdateAsyncWaitCountPass);
}

void addControlConstant(llvm::Module *module, const char *name,
                        uint32_t bitwidth, uint32_t value) {
  using llvm::GlobalVariable;

  llvm::IntegerType *type =
      llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
  auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
  auto *constant = new llvm::GlobalVariable(
      *module, type, /*isConstant=*/true,
      GlobalVariable::LinkageTypes::LinkOnceODRLinkage, initializer, name,
      /*before=*/nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal,
      /*addressSpace=*/4);
  constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
  constant->setVisibility(GlobalVariable::VisibilityTypes::ProtectedVisibility);
}
} // namespace

void init_triton_hcu(py::module &&m) {
  m.doc() = "Python bindings to the HCU Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_hcu_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.attr("TARGET_TRIPLE") = hcuTargetTriple;
  m.attr("CALLING_CONV_HCUGPU_KERNEL") =
      (unsigned)llvm::CallingConv::AMDGPU_KERNEL;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::hcugpu::TritonHCUGPUDialect>();
    // registry.insert<mlir::ROCDL::ROCDLDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple",
        [](llvm::Module *module) { module->setTargetTriple(hcuTargetTriple); });

  // Set target architecture ISA version
  m.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  // Set boolean control constant
  m.def("set_bool_control_constant",
        [](llvm::Module *module, const std::string &name, bool enable) {
          addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
        });

  // Set code object ABI version
  m.def("set_abi_version", [](llvm::Module *module, int version) {
    // Inject the control constant into the LLVM module so that device libraries
    // linked against module can resolve their references to it.
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

    // Also attach the control attribute on the LLVM module. This is also needed
    // in addition to the above for various transformations to know what code
    // object version we are targeting at.
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });

  m.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    // We can have Clang version metadata from device libraries linked in. We
    // don't care about them so drop them.
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    // Also various OpenCL version details.
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  m.def(
      "assemble_amdgcn",
      [](const std::string &assembly, const std::string &arch,
         const std::string &features) {
        std::string error;

        llvm::Triple triple(hcuTargetTriple);
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
        if (!target)
          throw std::runtime_error("target lookup error: " + error);

        llvm::SourceMgr srcMgr;
        srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                                  llvm::SMLoc());

        const llvm::MCTargetOptions mcOptions;
        std::unique_ptr<llvm::MCRegisterInfo> mri(
            target->createMCRegInfo(hcuTargetTriple));
        std::unique_ptr<llvm::MCAsmInfo> mai(
            target->createMCAsmInfo(*mri, hcuTargetTriple, mcOptions));
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target->createMCSubtargetInfo(hcuTargetTriple, arch, features));

        llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                            &mcOptions);
        std::unique_ptr<llvm::MCObjectFileInfo> mofi(
            target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                           /*LargeCodeModel=*/false));
        ctx.setObjectFileInfo(mofi.get());

        llvm::SmallString<128> cwd;
        if (!llvm::sys::fs::current_path(cwd))
          ctx.setCompilationDir(cwd);

        llvm::SmallVector<char, 0> result;
        llvm::raw_svector_ostream svos(result);

        std::unique_ptr<llvm::MCStreamer> mcStreamer;
        std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

        std::unique_ptr<llvm::MCCodeEmitter> ce(
            target->createMCCodeEmitter(*mcii, ctx));
        std::unique_ptr<llvm::MCAsmBackend> mab(
            target->createMCAsmBackend(*sti, *mri, mcOptions));
        std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
        mcStreamer.reset(target->createMCObjectStreamer(
            triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

        std::unique_ptr<llvm::MCAsmParser> parser(
            createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
        std::unique_ptr<llvm::MCTargetAsmParser> tap(
            target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
        if (!tap)
          throw std::runtime_error("assembler initializtion error");

        parser->setTargetParser(*tap);
        parser->Run(/*NoInitialTextSection=*/false);

        return py::bytes(std::string(result.begin(), result.end()));
      },
      py::return_value_policy::take_ownership);

  m.def("need_extern_lib", [](llvm::Module *module, const std::string &lib) {
    for (llvm::Function &f : module->functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        llvm::StringRef funcName = f.getName();
        // The rule for linking the extern lib:
        //    if the function name includes ocml or ockl, link
        //    ocml or ockl accordingly.
        if (funcName.contains(lib))
          return true;
        if (funcName.contains("__nv_")) {
          std::stringstream message;
          message << "Implicit conversion of CUDA " << funcName.str()
                  << " device function has been dropped; "
                  << "please, update your source program to use "
                     "triton.language.extra.<op> "
                  << "to replace triton.language.extra.cuda.<op>";
          throw std::runtime_error(message.str());
        }
      }
    }
    return false;
  });

  m.def("has_matrix_core_feature", [](const std::string &arch) {
    using mlir::triton::HCU::ISAFamily;
    switch (mlir::triton::HCU::deduceISAFamily(arch)) {
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA1:
    case ISAFamily::RDNA3:
      return true;
    default:
      return false;
    }
  });
}
