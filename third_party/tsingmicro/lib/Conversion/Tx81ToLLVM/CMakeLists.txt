add_triton_library(Tx81ToLLVM
  Tx81ToLLVM.cpp

  DEPENDS
  Tx81ToLLVMConversionPassIncGen
  MLIRMemRefToLLVM

  LINK_LIBS PUBLIC
  MLIRMemRefToLLVM
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRArithToLLVM
  MLIRAffineToStandard
  MLIRLinalgToStandard
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTransforms
)
