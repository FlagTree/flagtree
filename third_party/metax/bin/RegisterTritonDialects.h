/* 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved. */
#pragma once

#include "mlir/InitAllPasses.h"
#include "python/src/plugin.h"
#include "triton/Target/LLVMIR/Passes.h"

using BackendRegisterFunc = void (*)();
BackendRegisterFunc load_backend_register_func(const char *backend_name,
                                               const char *func_name) {
  void *symbol = load_backend_symbol(backend_name, func_name);
  return reinterpret_cast<BackendRegisterFunc>(symbol);
}

using DialectRegisterFunc = void (*)(mlir::DialectRegistry *);
DialectRegisterFunc load_dialect_register_func(const char *backend_name,
                                               const char *func_name) {
  void *symbol = load_backend_symbol(backend_name, func_name);
  return reinterpret_cast<DialectRegisterFunc>(symbol);
}

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();

  auto registerAllTritonPasses =
      load_backend_register_func("metax", "registerAllTritonPasses");
  registerAllTritonPasses();
  auto registerConvertTritonGPUToLLVMPass =
      load_backend_register_func("metax", "registerConvertTritonGPUToLLVMPass");
  registerConvertTritonGPUToLLVMPass();

  auto registerDialect = load_dialect_register_func("metax", "registerDialect");
  registerDialect(&registry);
}
