#ifndef FLAGTREE_PLUGIN_H
#define FLAGTREE_PLUGIN_H

#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#define DEFINE_LOAD_FUNC(symbol_name)                                          \
  static symbol_name##Func load_##symbol_name##_func(const char *backend_name, \
                                                     const char *func_name) {  \
    void *symbol = load_backend_symbol(backend_name, func_name);               \
    return reinterpret_cast<symbol_name##Func>(symbol);                        \
  }

#define DEFINE_CALL_LOAD_FUNC(backend_name, symbol_name)                       \
  auto func = load_##symbol_name##_func(#backend_name, #symbol_name);

#define XDEFINE_CALL_LOAD_FUNC(backend_name, symbol_name)                      \
  DEFINE_CALL_LOAD_FUNC(backend_name, symbol_name)

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

static std::optional<std::string> get_env(std::string_view key) {
  if (const char *p = std::getenv(std::string(key).c_str()))
    return std::string(p);
  return std::nullopt;
}

static void *load_backend_plugin(const char *backend_name) {
  const std::string lib_name = std::string(backend_name);
  std::string plugin_path = get_env("PLUGINSO_DIR").value_or(lib_name) +
                            get_env("FLAGTREE_BACKEND").value_or("") +
                            "TritonPlugin.so";
  void *handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load plugin: " << std::string(dlerror());
    std::cerr << "Please download ${FLAGTREE_BACKEND}TritonPlugin.so , place "
                 "it under ~/.flagtree/${FLAGTREE_BACKEND}, or specify another "
                 "directory manually, and set the PLUGINSO_DIR environment "
                 "variable to the directory containing the .so file.";
    assert(handle);
  }
  return handle;
}

static void *load_backend_symbol(const char *backend_name,
                                 const char *func_name) {
  void *handle = load_backend_plugin(backend_name);
  void *symbol = dlsym(handle, func_name);
  if (!symbol) {
    std::cerr << "Failed to load symbol: " << std::string(dlerror());

    assert(symbol);
  }
  return symbol;
}

static int load_backend_const_int(const char *backend_name,
                                  const char *const_name) {
  void *handle = load_backend_plugin(backend_name);
  void *symbol = dlsym(handle, const_name);
  if (!symbol) {
    std::cerr << "Failed to load symbol: " << std::string(dlerror());
    assert(symbol);
  }
  return *(const int *)symbol;
}

#endif
