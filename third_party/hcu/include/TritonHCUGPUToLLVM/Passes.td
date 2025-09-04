#ifndef TRITONHCUGPU_CONVERSION_PASSES
#define TRITONHCUGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeUnsupportedHCUConversions : Pass<"decompose-unsupported-hcu-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose conversions that are not supported by TritonGPU -> LLVM";
    let constructor = "mlir::triton::HCU::createDecomposeUnsupportedConversionsPass(\"\")";

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
    ];
}

def OptimizeHCULDSUsage : Pass<"optimize-hcu-lds-usage", "mlir::ModuleOp"> {
    let summary = "Minimize LDS usage";
    let constructor = "mlir::triton::HCU::createOptimizeLDSUsagePass(\"\")";

    let options = [
        Option<"targetArch", "target-arch", "std::string", /*default*/"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"customLDSLimit", "lds-limit", "int", /*default*/"0",
               "custom limit of LDS consumption, if not provided, maximum LDS size is used">,
    ];
}

def ConvertTritonHCUGPUToLLVM : Pass<"convert-triton-hcugpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonHCUGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertBuiltinFuncToLLVM : Pass<"convert-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertBuiltinFuncToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

}

def InsertInstructionSchedHints : Pass<"insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Insert instruction scheduling hints after the dot ops in the main loop";
    let constructor = "mlir::triton::createInsertInstructionSchedHintsPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def LowerInstructionSchedHints : Pass<"lower-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Lower instruction scheduling hints to LLVM intrinsics";
    let constructor = "mlir::triton::createLowerInstructionSchedHintsPass(\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"variant", "variant", "std::string", /*default*/"\"default\"",
               "instruction scheduling variant">,
    ];
}


#endif
