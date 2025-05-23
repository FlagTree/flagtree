from triton.language import core
from triton.runtime.jit import jit


@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along (currently only final dimension supported)
    :type dim: int
    """
    core.static_print("tl.flip is unsupported for now. Use libdevice.flip instead.")
    core.static_assert(False)
    return x
