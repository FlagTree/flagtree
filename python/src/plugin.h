#ifndef FLAGTREE_PLUGIN_H
#define FLAGTREE_PLUGIN_H

#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include<filesystem>

#define DEFINE_LOAD_FUNC(symbol_name)                                          \
  static symbol_name##Func load_##symbol_name##_func(const char *backend_name, \
                                                     const char *func_name) {  \
    void *symbol = load_backend_symbol(backend_name, func_name);               \
    return reinterpret_cast<symbol_name##Func>(symbol);                        \
  }

#define DEFINE_CALL_LOAD_FUNC(backend_name, symbol_name)                       \
  static auto func = load_##symbol_name##_func(#backend_name, #symbol_name);

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
  const std::string lib_name =  DEFAULT_PLUGIN_DIR +"/" + backend_name + "TritonPlugin.so";
  void *handle = dlopen(lib_name.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr<<"Warnings: can not find Plugin.so at default path"
              "We will get the Plugin.so from $ENV{FLAGTREE_BACKEND_PLUGIN_LIB_DIR}"
              " and help you copy it to default path";
    if(auto lib_dir = get_env("FLAGTREE_BACKEND_PLUGIN_LIB_DIR")){
      const std::string src_path=*lib_dir+"/"+backend_name+"TritonPlugin.so";
      std::filesystem::copy_file(src_path,lib_name);
      try
      {
         std::filesystem::copy_file(src_path,lib_name);
      }
      catch (std::filesystem::filesystem_error& e)
      {
        std::cout << "Could not copy the share library: " << e.what() << '\n';
      }
      handle=dlopen(lib_name.c_str(), RTLD_LAZY);
    }
  }
  if (!handle) {
    std::cerr << "Failed to load plugin: " << std::string(dlerror());
    std::cerr << "We could not find your shared library in the default directory, nor could we find the environment variable. "
                 "If you haven't downloaded the shared library yet, please download it from the following URLs:\n"
                 "{\"backend\":\"iluvatar\",\"urls\":\"https://github.com/"
                 "FlagTree/flagtree/releases/download/"
                 "v0.3.0-build-deps/"
                 "iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-"c
                 "cxxabi1.3.12-ubuntu-x86_64.tar.gz\"} and\n"
                 "{\"backend\":\"mtheads\",\"urls\":\"https://github.com/"
                 "FlagTree/flagtree/releases/download/v0.3.0-build-deps/"
                 "mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-"
                 "cxxabi1.3.13-ubuntu-x86_64_v0.3.0.tar.gz\"},\n"
                 "then set $ENV{FLAGTREE_BACKEND_PLUGIN_LIB_DIR} to the directory where the shared library is located.\n"
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
