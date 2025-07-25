#ifndef TRITON_TOOLS_SYS_GETENV_HPP
#define TRITON_TOOLS_SYS_GETENV_HPP

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <optional>
#include <set>
#include <sstream>
#include <string>

#ifdef __ILUVATAR__
#include <dlfcn.h>
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace mlir::triton {

inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    "AMDGCN_ENABLE_DUMP",
    "DISABLE_FAST_REDUCTION",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_ENABLE_TIMING",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE",
    "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_LLVM_DEBUG_ONLY",
    "USE_TTGIR_LOC",
    "NVPTX_ENABLE_DUMP",
#ifdef __ILUVATAR__
    "ILUIR_ENABLE_DUMP",
#endif
    // clang-format on
};

inline const std::set<std::string> CACHE_NEUTRAL_ENV_VARS = {
    "TRITON_REPRODUCER_PATH",
};

namespace tools {

inline void assertIsRecognized(const std::string &env) {
  bool is_invalidating = CACHE_INVALIDATING_ENV_VARS.find(env.c_str()) !=
                         CACHE_INVALIDATING_ENV_VARS.end();
  bool is_neutral =
      CACHE_NEUTRAL_ENV_VARS.find(env.c_str()) != CACHE_NEUTRAL_ENV_VARS.end();
  std::string errmsg = env + "is not recognized. "
                             "Please add it to triton/tools/sys/getenv.hpp";
  assert((is_invalidating || is_neutral) && errmsg.c_str());
}

inline std::string getStrEnv(const std::string &env) {
  assertIsRecognized(env);
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

// return value of a cache-invalidating boolean environment variable
inline bool getBoolEnv(const std::string &env) {
  assertIsRecognized(env);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

inline std::optional<bool> isEnvValueBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "on" || str == "true" || str == "1")
    return true;
  if (str == "off" || str == "false" || str == "0")
    return false;
  return std::nullopt;
}

#ifdef __ILUVATAR__
static fs::path &getCudaPath(void) {
  static fs::path cuda_path = [] {
    void *handle = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!handle) {
      std::fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
    }
    void *pfunc = dlsym(handle, "nvrtcCompileProgram");
    Dl_info info;
    if (dladdr(pfunc, &info) == 0) {
      std::fprintf(stderr, "Failed to get symbol information: %s\n", dlerror());
      exit(EXIT_FAILURE);
    }
    return fs::path(info.dli_fname).parent_path().parent_path();
  }();
  return cuda_path;
}

static fs::path &getLinkerPath(void) {
  static fs::path linker_path = [] {
    fs::path cuda_path = getCudaPath();
    fs::path linker_path1 = cuda_path / "bin/ld.lld";
    fs::path linker_path2 = cuda_path / "../bin/ld.lld";
    if (!fs::exists(linker_path1)) {
      if (fs::exists(linker_path2)) {
        linker_path1 = linker_path2;
      } else {
        fprintf(stderr, "iluvatar linker not found in %s and %s\n",
                linker_path1.c_str(), linker_path2.c_str());
        exit(EXIT_FAILURE);
      }
    }
    return linker_path1;
  }();
  return linker_path;
}
#endif

} // namespace tools
} // namespace mlir::triton

#endif
