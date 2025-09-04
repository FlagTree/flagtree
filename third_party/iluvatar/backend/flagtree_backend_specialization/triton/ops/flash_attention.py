def is_iluvatar():
    return True


def attention_forward_config():
    from ..runtime.build import is_corex
    if is_corex():
        BLOCK_M = 64
        BLOCK_N = 64
        num_stages = 1
        return (BLOCK_M, BLOCK_N, num_stages)
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        num_stages = 4
        return (BLOCK_M, BLOCK_N, num_stages)


def attention_backward_config(BLOCK_DMODEL):
    from ..runtime.build import is_corex
    # otherwise shared memory out of resource
    BLOCK = 128 if not is_corex() else 64  # FIXME: currently BLOCK=128 has issues, BLOCK=64 works for common cases.
    num_warps = 16 if is_corex() and BLOCK_DMODEL > 64 else 8
    return (BLOCK, num_warps)
