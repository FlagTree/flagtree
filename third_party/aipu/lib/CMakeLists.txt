add_subdirectory(Passes)

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

add_library(aipu_interface SHARED main.cpp)
target_link_libraries(aipu_interface PRIVATE
    Python3::Module
    pybind11::headers
    MLIRAIPUPasses
    ${MLIRPythonCAPI}
)
