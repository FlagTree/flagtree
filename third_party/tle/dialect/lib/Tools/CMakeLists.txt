add_triton_library(TleTools
  FlagcxUtils.cpp

  DEPENDS
  TleTableGen
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUOpInterfacesIncGen
  ${TLE_TRITON_GPU_ATTR_DEPENDS}

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
