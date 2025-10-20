if (FLAGTREE_BACKEND)
    set(NVGPUIR "")
else()
    set(NVGPUIR "NVGPUIR")
endif()

if(NOT FLAGTREE_BACKEND STREQUAL "iluvatar")
  set(_EXTRA_LINK_LIBS TritonNvidiaGPUTransforms)
else()
  set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonConversionTritonGPUToLLVM)
endif()

add_triton_library(TritonGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    DotOpToLLVM/FMA.cpp
    TypeConverter.cpp
    Utility.cpp
    ElementwiseOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    AssertOpToLLVM.cpp
    ViewOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    AllocateSharedMemory.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    FuncOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    PrintOpToLLVM.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    ${NVGPUIR}
    ${_EXTRA_LINK_LIBS}
)
