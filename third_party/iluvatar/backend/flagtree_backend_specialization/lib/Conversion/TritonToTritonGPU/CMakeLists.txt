add_triton_library(FlagTree_iluvatar_TritonConversionTritonToTritonGPU
  TritonToTritonGPUPass.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)