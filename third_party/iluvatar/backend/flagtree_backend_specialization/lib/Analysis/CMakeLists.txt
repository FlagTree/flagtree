add_triton_library(FlagTree_iluvatar_TritonAnalysis
  AxisInfo.cpp
  Membar.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)