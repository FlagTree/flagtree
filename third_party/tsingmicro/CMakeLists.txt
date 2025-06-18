if(NOT DEFINED TX8_HOME)
  if(DEFINED ENV{TX8_HOME})
    set(TX8_HOME $ENV{TX8_HOME})
  else()
    message(FATAL_ERROR "TX8_HOME environment variable is not defined")
  endif()
endif()

set(TSM_BACKEND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend)
set(XUANTIE_NAME Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/crt/include)
include_directories(${TX8_HOME}/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(crt)
if(TRITON_BUILD_PYTHON_MODULE)
  # FIXME: Unify the libraries for TsingMicro into fewer ones
  add_triton_plugin(TritonTsingMicro ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_tsingmicro.cc
                    LINK_LIBS ZTCAnalysis ZTCAnalysisStructured MagicKernelIR
                    Tx81IR TritonTilingExtIR TritonStructuredIR TritonToCoreDialects
                    TritonToLinalg TritonToStructured StructuredToMemref LinalgToMagicKernel
                    TritonArithToLinalg CoreDialectsToMK Tx81ToLLVM Tx81MemrefToLLVM MKToTx81)
  target_link_libraries(TritonTsingMicro PRIVATE Python3::Module pybind11::headers)
endif()
#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()
#add_subdirectory(test)
