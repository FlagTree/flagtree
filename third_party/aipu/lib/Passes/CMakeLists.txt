add_mlir_dialect_library(MLIRAIPUPasses
    ConvertBoolArg2I8.cpp

    DEPENDS
    MLIRAIPUPassesIncGen
)
