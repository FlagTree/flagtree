# flagtree tle
import triton.language.core as tl
from triton.language.core import tensor
# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@tl.builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, is_async=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    x = tl.load(pointer, mask=mask, other=other, boundary_check=boundary_check, padding_option=padding_option,
                cache_modifier=cache_modifier, eviction_policy=eviction_policy, volatile=volatile, _semantic=_semantic)
    x.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(is_async))
    return x

@tl.builtin
def make_partition_view(base: tensor, shape, strides, tile, padding_value="zero", _semantic=None) -> "tl.tensor_view":
    """Create a tiled partition view over a scalar base pointer.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.
    """
    return _semantic.make_partition_view(base, shape, strides, tile, padding_value)


@tl.builtin
def make_strided_view(base: tensor, shape, strides, tile, traversal_strides,
                      padding_value="zero", _semantic=None) -> "tl.tensor_view":
    """Create a tiled pointer view whose origins advance by ``traversal_strides``.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.
    """
    return _semantic.make_strided_view(base, shape, strides, tile, traversal_strides, padding_value)


@tl.builtin
def make_gather_scatter_view(base: tensor, shape, strides, tile, sparse_dim,
                             padding_value="zero", _semantic=None) -> "tl.tensor_view":
    """Create a pointer view with tensor-valued indices in selected dimensions.

    ``padding_value`` must be one of ``"zero"``, ``"nan"``, ``"inf"``, or ``"-inf"``.
    """
    return _semantic.make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value)
