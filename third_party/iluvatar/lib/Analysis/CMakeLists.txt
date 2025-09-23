add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
  FlagTree_${FLAGTREE_BACKEND}_TritonAnalysis

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  #TritonNvidiaGPUIR
  FlagTree_${FLAGTREE_BACKEND}_TritonAnalysis
)
