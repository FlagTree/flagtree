add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMetax
      SHARED_LIB metaxTritonPlugin
    )
  else()
    find_library(metaxTritonPluginLib
      NAMES
        metaxTritonPlugin.so
      PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMetax
      SHARED_LIB ${metaxTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
