def init_corexLoad(corexLoad):
    if corexLoad is None:
        return dict()
    return None


def to_dict_corexLoad(divisible_by_16, equal_to_1, corexLoad):
    return {
            'divisible_by_16': list(divisible_by_16), 'equal_to_1': list(equal_to_1), 'corexLoad': list(corexLoad.items())
        }


def from_dict_corexLoad(data):
    from triton.compiler.compiler import AttrsDescriptor
    return AttrsDescriptor(divisible_by_16=set(data.get('divisible_by_16', [])),
                            equal_to_1=set(data.get('equal_to_1', [])), corexLoad=dict(data.get('corexLoad', [])))


def hash_AttrsDescriptor(attrsDescriptor):
    key = str(
        [sorted(x) if isinstance(x, tuple) or isinstance(x, set) else x.values() for x in attrsDescriptor.__dict__.values()])
    return key


def src_fn_hash_cache_file(ir_source, src, hash):
    from triton.runtime.jit import JITFunction
    if not ir_source and isinstance(src.fn, JITFunction):
        src.fn.hash_cache_file = hash


def src_fn_so_path(ir_source, src):
    from triton.runtime.driver import driver
    if not ir_source:
        src.fn.so_path = driver.active.get_cache_path()


def init_handles_n_threads(compiledKernel, device):
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.driver import driver
    compiledKernel.module, compiledKernel.function, compiledKernel.n_regs, compiledKernel.n_spills, compiledKernel.n_threads = \
        driver.active.utils.load_binary(compiledKernel.name, compiledKernel.kernel, compiledKernel.metadata.shared, device)
    if compiledKernel.metadata.num_warps * 64 > compiledKernel.n_threads:
        compiledKernel.module = None
        raise OutOfResources(compiledKernel.metadata.num_warps * 64, compiledKernel.n_threads, "threads")