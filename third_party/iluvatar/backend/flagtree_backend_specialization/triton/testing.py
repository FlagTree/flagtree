def corex_cmd(attrs):
    from .triton.runtime.build import is_corex
    if is_corex():
        cmd = ['ixsmi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']
        return cmd
    else:
        return None


def get_mem_clock_khz():
    import torch
    capability = torch.cuda.get_device_capability()
    if capability[0] == 8:
        mem_clock_khz = 1800000
        return mem_clock_khz
    else:
        return None


def dtype_and_corex_assert(dtype):
    import torch
    from .triton.runtime.build import is_corex
    assert dtype == torch.float16 or is_corex()