cmake_minimum_required(VERSION 3.18)

set(TARGET Tx81)
# Set TARGET from environment variable
if(NOT DEFINED TARGET)
  if(DEFINED ENV{CRT_TARGET})
    set(TARGET $ENV{CRT_TARGET})
  else()
    message(FATAL_ERROR "CRT_TARGET environment variable is not defined")
  endif()
endif()

if(NOT DEFINED XUANTIE_NAME)
  if(DEFINED ENV{XUANTIE_NAME})
    set(XUANTIE_NAME $ENV{XUANTIE_NAME})
  else()
    message(FATAL_ERROR "XUANTIE_NAME environment variable is not defined")
  endif()
endif()

# Set LLVM_SYSPATH from environment variable
if(NOT DEFINED LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH $ENV{LLVM_SYSPATH})
  else()
    message(FATAL_ERROR "LLVM_SYSPATH environment variable is not defined")
  endif()
endif()

if(NOT DEFINED TX8_HOME)
  if(DEFINED ENV{TX8_HOME})
    set(TX8_HOME $ENV{TX8_HOME})
  else()
    message(FATAL_ERROR "TX8_HOME environment variable is not defined")
  endif()
endif()

# Project name and version
project(VendorRuntime LANGUAGES CXX C)

# Define RISC-V target triple
set(RISCV_TRIPLE "riscv64-unknown-elf")
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)
set(CMAKE_C_COMPILER ${LLVM_SYSPATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${LLVM_SYSPATH}/bin/clang++)

# Define standard include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/${TARGET})
include_directories(${TX8_HOME}/include)
include_directories(${TX8_HOME}/${XUANTIE_NAME}/riscv64-unknown-elf/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Set build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# Library name: vr stands for Vendor Runtime
set(VENDOR_RUNTIME_LIB vr)

# Collect all source files from the vendor directory
file(GLOB_RECURSE VENDOR_SOURCES lib/${TARGET}/*.c)

# Define RISC-V specific compile options
set(RISCV_COMPILE_OPTIONS
  --target=${RISCV_TRIPLE}
  -march=rv64gc
  -mabi=lp64d
  -mcmodel=medany
)

# Add the library target
add_library(${VENDOR_RUNTIME_LIB} STATIC ${VENDOR_SOURCES})

# Apply RISC-V specific settings to our target
target_compile_options(${VENDOR_RUNTIME_LIB} PRIVATE ${RISCV_COMPILE_OPTIONS})
target_link_options(${VENDOR_RUNTIME_LIB} PRIVATE --target=${RISCV_TRIPLE})

if (DEFINED TSM_BACKEND_DIR)
  set_target_properties(${VENDOR_RUNTIME_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${TSM_BACKEND_DIR}/lib
  )
else ()
  # Set properties for the library
  set_target_properties(${VENDOR_RUNTIME_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
endif ()

# Setup compiler and environment for RISC-V compilation
if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Use the existing Clang installation with target triple
  message(STATUS "Using Clang with RISC-V target triple")
else()
  # Override compiler paths if using explicit RISC-V toolchain
  message(STATUS "Setting explicit RISC-V compiler from LLVM_SYSPATH")

  foreach(source ${VENDOR_SOURCES})
    if(source MATCHES "\\.(c)$")
      set_source_files_properties(${source} PROPERTIES
        COMPILE_FLAGS "-xc --target=${RISCV_TRIPLE}"
        LANGUAGE C)
    elseif(source MATCHES "\\.(cpp)$")
      set_source_files_properties(${source} PROPERTIES
        COMPILE_FLAGS "-xc++ --target=${RISCV_TRIPLE}"
        LANGUAGE CXX)
    endif()
  endforeach()

  # Set compiler launch commands for the target
  add_custom_command(TARGET ${VENDOR_RUNTIME_LIB} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Building ${VENDOR_RUNTIME_LIB} for RISC-V target"
  )
endif()

# Install targets
install(TARGETS ${VENDOR_RUNTIME_LIB}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers (optional)
file(GLOB_RECURSE VENDOR_HEADERS Target/lib/${TARGET}/*.h)
install(FILES ${VENDOR_HEADERS} DESTINATION include/${TARGET})
install(FILES Target/lib/${TARGET}/libkcorert.a DESTINATION lib/${TARGET})
