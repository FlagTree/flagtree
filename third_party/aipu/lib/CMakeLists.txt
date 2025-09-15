add_subdirectory(Passes)

add_triton_library(Registrar Registrar.cc)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import mlir; print(str(mlir.__path__).split(\"'\")[1])"
  OUTPUT_VARIABLE MLIR_PYTHON_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_library(MLIRPythonCAPI
    NAMES MLIRPythonCAPI
    PATHS ${MLIR_PYTHON_PATH}/_mlir_libs
)
if (NOT MLIRPythonCAPI)
    message(FATAL_ERROR "Cannot find the libMLIRPythonCAPI.so.")
endif()

message(STATUS "Checking for nanobind in python path...")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}"
    -c "import nanobind;print(nanobind.include_dir(), end='')"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE PACKAGE_DIR
    ERROR_QUIET
)
if(NOT STATUS EQUAL "0")
    message(FATAL_ERROR "not found (install via 'pip install nanobind' or set nanobind_DIR)")
endif()
set(nanobind_INCLUDE_DIR "${PACKAGE_DIR}")

nanobind_add_module(aipu_interface SHARED main.cpp)
target_include_directories(aipu_interface
    PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/flir/include
    ${CMAKE_SOURCE_DIR}/third_party/flir/tools
    ${CMAKE_BINARY_DIR}/third_party/flir/include
    ${nanobind_INCLUDE_DIR}
)
target_link_libraries(aipu_interface PRIVATE
    MLIRAIPUPasses
    ${MLIRPythonCAPI}
    MLIRMathExtDialect
    TPtrIR
    TritonStructuredIR
    TritonTilingExtIR
)
