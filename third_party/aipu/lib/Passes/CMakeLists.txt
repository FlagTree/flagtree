add_mlir_dialect_library(MLIRAIPUPasses
    ConvertBoolArg2I8.cpp
    SCFLoopBufferizationPreprocess.cpp
    FlattenMemRefs.cpp
    ForwardStoreToLoad.cpp
    ConvertI64ToI32.cpp

    DEPENDS
    MLIRAIPUPassesIncGen
    TritonIR
)
target_include_directories(MLIRAIPUPasses PRIVATE ${CMAKE_SOURCE_DIR}/include)
