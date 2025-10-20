add_triton_library(FlagTree_iluvatar_TritonConversionTritonGPUToLLVM
  ElementwiseOpToLLVM.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)