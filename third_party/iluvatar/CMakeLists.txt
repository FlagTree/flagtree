add_subdirectory(include)
add_subdirectory(lib)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
set (ILUVATAR_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB iluvatarTritonPlugin
    )
  else()
    find_library(iluvatarTritonPluginLib
      NAMES
        iluvatarTritonPlugin.so
      PATHS
      ${ILUVATAR_PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB ${iluvatarTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
