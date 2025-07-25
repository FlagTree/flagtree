import functools


# Add these stub function to prevent import failures from external repositories.
@functools.lru_cache()
def get_clock_rate_in_khz():
    raise NotImplementedError


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    raise NotImplementedError


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    raise NotImplementedError


def get_tflops(device, num_ctas, num_warps, dtype):
    raise NotImplementedError


def estimate_matmul_time(
        # backend, device,
        num_warps, num_stages,  #
        A, B, C,  #
        M, N, K,  #
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,  #
        debug=False, **kwargs  #
):
    raise NotImplementedError


def early_config_prune(configs, named_args, **kwargs):
    raise NotImplementedError
