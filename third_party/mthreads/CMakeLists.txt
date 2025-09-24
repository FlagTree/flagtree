add_subdirectory(include)
add_subdirectory(lib)

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
set (MTHREADS_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB mthreadsTritonPlugin
    )
  else()
    find_library(mthreadsTritonPluginLib
      NAMES
        mthreadsTritonPlugin.so
      PATHS
        ${MTHREADS_PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB ${mthreadsTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
