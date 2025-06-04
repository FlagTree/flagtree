option(TRITON_ADAPTER_BUILD_CPU_BACKEND "Build triton-adapter CPU backend" ON)

set(TRITON_ADAPTER_SOURCE_DIR ".")
set(TRITON_ADAPTER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(./include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if (TRITON_ADAPTER_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonAdapter triton_adapter.cc LINK_LIBS TritonToLinalg)
endif()
