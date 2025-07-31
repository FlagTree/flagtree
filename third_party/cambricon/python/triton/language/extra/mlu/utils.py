from triton.language import core
from triton._C.libtriton import mlu, ir


@core.extern
def perf_begin(_builder=None):
    context = ir.context()
    location = _builder.get_loc()
    pt = _builder.get_insertion_point()
    mlu.create_readperf_begin(context, location, pt)


@core.extern
def perf_end(_builder=None):
    context = ir.context()
    location = _builder.get_loc()
    pt = _builder.get_insertion_point()
    mlu.create_readperf_end(context, location, pt)
