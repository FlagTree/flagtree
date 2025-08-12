def sequence_parallel_mma_v3_dq(ds, k):
    import triton.language as tl
    dq = tl.dot(ds, k)
    return dq


def hardware_config(capability):
    from .triton.runtime.build import is_corex
    if is_corex():
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 1
    else:
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_M = 128
        BLOCK_N = 64
        num_stages = 4
    return BLOCK_M, BLOCK_N, num_stages


def get_num_stages(num_stages):
    return num_stages


def get_block_and_warps(context):
    from .triton.runtime.build import is_corex
    # otherwise shared memory out of resource
    BLOCK = 128 if not is_corex() else 64  # FIXME: currently BLOCK=128 has issues, BLOCK=64 works for common cases.
    num_warps = 16 if is_corex() and context.BLOCK_DMODEL > 64 else 8
    return BLOCK, num_warps


def get_num_warps(num_warps):
    return num_warps