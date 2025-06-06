get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
target_link_libraries(triton-opt PRIVATE
  TritonLLVMIR
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  MLIRGPUToROCDLTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
set_target_properties(triton-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
)

mlir_check_all_link_libraries(triton-opt)

add_llvm_executable(triton-reduce triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-reduce)

llvm_update_compile_flags(triton-reduce)
target_link_libraries(triton-reduce PRIVATE
  TritonLLVMIR
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRTransforms
)
set_target_properties(triton-reduce PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
)

mlir_check_all_link_libraries(triton-reduce)

add_llvm_executable(triton-lsp triton-lsp.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-lsp)

llvm_update_compile_flags(triton-lsp)
target_link_libraries(triton-lsp PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRTransforms
)
set_target_properties(triton-lsp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
)

mlir_check_all_link_libraries(triton-lsp)


add_llvm_executable(triton-llvm-opt
  triton-llvm-opt.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)
set_target_properties(triton-llvm-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
)
