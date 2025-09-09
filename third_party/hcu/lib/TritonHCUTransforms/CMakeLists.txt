add_triton_library(TritonHCUTransforms
    AccelerateHcuFlashAttention.cpp
    ControlFaFwdBufferLoadCnt.cpp
    CopyGlobalToLocalSwizzle.cpp
    FaFwdInsertWait.cpp
    MoveLoadToFrontOfDOT.cpp
    UpdateAsyncWaitCount.cpp
    StreamPipeline.cpp
    Utility.cpp
    AccelerateHCUMatmul.cpp
    CanonicalizePointers.cpp
    OptimizeEpilogue.cpp
    ReorderInstructions.cpp
    HCUStreamPipeline.cpp
    StreamPipelineV2.cpp
    MfmaGroup.cpp
    ConvertToBufferOps.cpp
    DEPENDS
    TritonHCUTransformsIncGen
    TritonGPUIR
)

target_include_directories(TritonHCUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonHCUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
