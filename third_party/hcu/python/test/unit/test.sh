pytest -v -s language/test_annotations.py
pytest -v -s language/test_block_pointer.py
pytest -v -s language/test_compile_errors.py
pytest -v -s language/test_random.py
pytest -v -s language/test_subprocess.py
pytest -v -s language/test_line_info.py
pytest -v -s language/test_decorator.py
pytest -v -s language/test_reproducer.py
pytest -v -s language/test_pipeliner.py
pytest -v -s language/test_standard.py
pytest -v -s runtime/test_autotuner.py
pytest -v -s runtime/test_bindings.py
pytest -v -s runtime/test_cache.py
pytest -v -s runtime/test_cublas.py
pytest -v -s runtime/test_driver.py
pytest -v -s runtime/test_jit.py
pytest -v -s runtime/test_launch.py
pytest -v -s runtime/test_subproc.py
pytest -v -s tools/test_disasm.py
pytest -v -s instrumentation/test_gpuhello.py
pytest -v -s language/test_core.py
pytest -v -s language/test_conversions.py
python gemm/gemm.py
pytest -v -s sglang/test_int4kv_asym.py
pytest -v -s fused_moe/test_moe.py
python ../../examples/W4A8/W4A8_fusemoe_kernel_tunning2ntsz.py
python ../../examples/W4A8/W4A8_GEMM_splitk_tune.py
pytest -v -s ../../examples/fused_moe/test_moe.py
python ../../examples/fused_moe_w8a8/benchmark_moe_int8.py
pytest -v -s ../../examples/MLAttention/test_triton_decode_attention.py
python ../../examples/streamK/test_streamK_Gemm.py
python ../../examples/W4A16/benchmark_moe_int4_sz_dpsk.py
