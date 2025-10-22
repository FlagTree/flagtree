if(FLAGTREE_BACKEND STREQUAL "iluvatar")
  set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonDialectTritonGPUIR)
else()
  set(_EXTRA_LINK_LIBS "")
endif()

add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
  ${_EXTRA_LINK_LIBS}
)
