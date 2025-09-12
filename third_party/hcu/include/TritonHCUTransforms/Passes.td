#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonHCUGPUStreamPipeline : Pass<"tritonhcugpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonHCUGPUStreamPipelinePass()";

  let dependentDialects = [];
}

def TritonHCUGPUStreamPipelineV2 : Pass<"tritonhcugpu-stream-pipeline-v2", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonHCUGPUStreamPipelineV2Pass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num_stages",
           "int32_t", /*default*/"2",
           "Number of Pipeline stages">
  ];
}

def TritonHCUGPUConvertToBufferOps : Pass<"tritonhcugpu-convert-buffer-ops", "mlir::ModuleOp"> {
  let summary = "Convert memory operations to buffer operations";

  let description = "This pass converts memory operations (e.g., tt.load/tt.store) to  hcugpu buffer operations, if possible";

  let constructor = "mlir::createTritonHCUGPUConvertToBufferOpsPass()";

  let dependentDialects = ["mlir::triton::hcugpu::TritonHCUGPUDialect"];
}



def TritonHCUGPUAccelerateMatmul : Pass<"tritonhcugpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., HCU matrix cores)
  }];

  let constructor = "mlir::createTritonHCUGPUAccelerateMatmulPass()";

  let dependentDialects = [];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"kPack", "kPack",
           "int32_t", /*default*/"1",
           "KWidth / kBase">,
    Option<"num_ldmatrixes", "num_ldmatrixes",
           "int32_t", /*default*/"0",
           "use ds_read_mat instruction">,
    Option<"enable_mmacfuse", "enable_mmacfuse",
           "int32_t", /*default*/"0",
           "use mmacfuse 16x64x32 instruction">
  ];
}

def TritonHCUGPUOptimizeEpilogue : Pass<"tritonhcugpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonHCUGPUOptimizeEpiloguePass()";

  let dependentDialects = [];

}

def TritonHCUGPUCanonicalizePointers : Pass<"tritonhcugpu-canonicalize-pointers", "mlir::ModuleOp"> {
  let summary = "Canonicalize pointers: rewrite pointers passed to load/store operation as a `<basePtr, offset>` pair.";

  let description = [{
  This pass pushes all the constant pointer arithmetic on a scalar basePtr, while all the vector
  pointer arithmetic to a vector offset. I.e., if we consider the following IR:
  ```
    %v_ptr = tt.splat %s_ptr
    %c_offset = tt.splat %s_offset
    %v_offset0 = tt.make_range
    %v_offset1 = tt.make_range
    %v_ptr0 = tt.addptr %v_ptr, %c_offset
    %v_ptr1 = tt.addptr %v_ptr0, %v_offset0
    %v_ptr2 = tt.addptr %v_ptr0, %v_offset1
    %data = tt.load(%v_ptr2)
  ```
  We transform this into:
  ```
    %s_ptr0 = tt.addptr %s_ptr, %s_offset
    %v_offset = %zero
    %v_offset = arith.addi %v_offset, %v_offset0
    %v_offset = arith.addi %v_offset, %v_offset1
    %c_ptr = tt.splat %s_ptr0
    %v_ptr = tt.addptr %c_ptr, %v_offset
    %data = tt.load(%v_ptr)
  ```
  In the above IR:
  -  `v_` means "variable vector across the program"
  -  `c_` means "constant vector across the program"
  -  `s_` means "scalar"
  So we transform the IR such that the constant updates become scalar updates, and the variable updates happen on the offset. Note that
  when we have to load the data, we splat the scalar pointer, add the "variable" offset and then issue the load.
  }];

  let constructor = "mlir::createTritonHCUGPUCanonicalizePointersPass()";

  let dependentDialects = [];

}

def TritonHCUGPUReorderInstructions: Pass<"tritonhcugpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonHCUGPUReorderInstructionsPass()";

  let dependentDialects = [];
}

def TritonHCUGlobalToLocalSwizzle : Pass<"tritonhcu-globalToLocal-swizzle", "mlir::ModuleOp"> {
  let summary = "global to local swizzle";

  let description = "This pass swizzle load for globalToLocalOp";

  let constructor = "mlir::createTritonHCUGlobalToLocalSwizzlePass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::triton::TritonDialect",
                          "mlir::arith::ArithDialect"];
}

def TritonHCUMoveLoadToFrontOfDOT : Pass<"tritonhcu-moveLoad-toFrontOf-dot", "mlir::ModuleOp"> {
  let summary = "move load to front of dot";

  let description = "This pass move load/globalToLocalOp to the front of dot ";

  let constructor = "mlir::createTritonHCUMoveLoadToFrontOfDOTPass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::triton::TritonDialect",
                          "mlir::arith::ArithDialect"];

}

def TritonHCUFaFwdControlCnt : Pass<"tritonhcu-faFwd-controlCnt", "mlir::ModuleOp"> {
  let summary = "FA FWD Control vmcnt num";

  let description = "This pass control fa fwd buffer load vmcnt num";
  let constructor = "mlir::createTritonHCUFaFwdControlCntPass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::triton::TritonDialect",
                          "mlir::arith::ArithDialect"];

  let options = [
    Option<"cnt_control", "cnt_control",
           "int32_t", /*default*/"0",
           "vmcnt control">
  ];
}

def TritonHCUFaFwdWait : Pass<"tritonhcu-faFwd-wait", "mlir::ModuleOp"> {
  let summary = "FA FWD Insert wait";

  let description = "This pass insert fa fwd buffer load vmcnt num";
  let constructor = "mlir::createTritonHCUFaFwdWaitPass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::triton::TritonDialect",
                          "mlir::arith::ArithDialect"];

  let options = [
    Option<"cnt_control", "cnt_control",
           "int32_t", /*default*/"0",
           "vmcnt control">
  ];
}

def TritonHCUAccelerateFlashAttention : Pass<"tritonhcu-accelerate-flash-attention", "mlir::ModuleOp"> {
  let summary = "accelerate flash attention";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., HCU tensor cores)
  }];

  let constructor = "mlir::createTritonHCUAccelerateFlashAttentionPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect"];

  let options = [];
}

def TritonHCUUpdateAsyncWaitCount: Pass<"tritonhcu-update-async-wait-count", "mlir::ModuleOp"> {
  let summary = "Adjust async wait count to allow prefetching over multiple loop iterations";

  let description = [{
      LLVM cannot see the dependency across loop iterations between AsyncCopy and local_reads. So we
      compute the number of interleaving global memory instructions to emit the correct waitcnt during lowering.
  }];

  let constructor = "mlir::createTritonHCUUpdateAsyncWaitCountPass()";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect", "mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect"];
  let options = [];
}

def TritonHCUStreamPipeline : Pass<"tritonhcu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonHCUStreamPipelinePass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num_stages",
           "int32_t", /*default*/"2",
           "Number of Pipeline stages">,
    Option<"globalPrefetch", "global_prefetch",
           "int32_t", /*default*/"0",
           "Set global prefetch stage count">,
    Option<"localPrefetch", "local_prefetch",
           "int32_t", /*default*/"0",
           "Set local prefetch stage count">,
    Option<"useAsyncCopy", "use_async_copy",
           "bool", /*default*/"false",
           "Use AsyncCopyGlobalToLocal to directly load to shared memory">,
  ];
}

#endif
