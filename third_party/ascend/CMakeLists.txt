add_subdirectory(triton-adapter triton-adapter)

add_triton_plugin(TritonHUAWEI ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cpp)

# Copy triton-adapter-opt to python files
add_custom_target(COPY_TRITON_ADAPTER_OPT)
add_custom_command(TARGET COPY_TRITON_ADAPTER_OPT POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                        $<TARGET_FILE:triton-adapter-opt>
                        ${TRITON_ROOT_DIR}/python/triton/backends/ascend/triton-adapter-opt
                    DEPENDS triton-adapter-opt)
add_dependencies(TritonHUAWEI COPY_TRITON_ADAPTER_OPT)
