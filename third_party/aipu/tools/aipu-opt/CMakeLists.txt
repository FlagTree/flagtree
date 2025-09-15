get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(aipu-opt aipu-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(aipu-opt)
target_link_libraries(aipu-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  # AIPU pass
  MLIRAIPUPasses
  # Triton
  TritonIR
)

mlir_check_all_link_libraries(aipu-opt)
