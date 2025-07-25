from triton.language import core, semantic
from triton._C.libtriton import mlu, ir

from typing import Optional

# ===----------------------------------------------------------------------===
#                               Gather
# ===----------------------------------------------------------------------===


@core.extern
def gather(src: core.tensor, offset: core.tensor, mask: Optional[core.tensor], *window_shape, other=None,
           _builder=None) -> core.tensor:
    """
    Perform a gather operation from `src` tensor using `offset`, optionally masked.

    Parameters:
    - src (core.tensor): The source tensor to gather data from.
    - offset (core.tensor): Tensor specifying the indices to gather.
                          Shape should end with the index depth (e.g., [..., K]).
    - mask (Optional[core.tensor]): A boolean mask tensor indicating valid entries
                                  in `offset`. If None, all entries are considered valid.
    - *window_shape: The shape of the gathered window (is a tuple). example: (4, ) or (4, 5)
    - other: A tensor or scalar to fill in positions where `mask` is False.
                                   Must be provided if `mask` is used.
                                   If it's a scalar, it will be cast and broadcast to the appropriate shape.
                                   If `mask` is None, this must be None as well.
    - _builder: Internal builder object used for IR construction.

    Returns:
    - core.tensor: A new tensor containing gathered values from `src`, shaped according
                to batch dimensions from `offset` and `window_shape`.

    Usage example:
        >>> output = gather(src_tensor, offset_tensor, mask_tensor, (4, 4), other=0.)
    """
    src_rank = len(src.shape)
    if src_rank == 1 and len(offset.shape) == 1:
        offset = semantic.expand_dims(offset, 1, _builder)

    offset_shape = [core._constexpr_to_value(x) for x in offset.shape]
    batch = offset_shape[:-1]
    dst_shape = [x for x in batch]
    window_shape_value = ()
    for i in window_shape:
        window_shape_value = core._constexpr_to_value(i)
    if isinstance(window_shape_value, int):
        dst_shape.append(window_shape_value)
    else:
        dst_shape.extend(window_shape_value)
    other = core._constexpr_to_value(other)

    # Check `mask` and `other` arguments
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if other is not None:
        other = semantic.to_tensor(other, _builder)
        src_ty = src.type.scalar
        other = semantic.cast(other, src_ty, _builder)
        other = semantic.broadcast_impl_shape(other, dst_shape, _builder)
    location = _builder.get_loc()
    pt = _builder.get_insertion_point()
    gather_ext = mlu.gather(location, pt, src.handle, offset.handle, mask.handle if mask else None,
                            other.handle if other else None, dst_shape)
    gather_ext = semantic.wrap_tensor(gather_ext, src.type.scalar, dst_shape)
    return gather_ext


# ===----------------------------------------------------------------------===
#                               Scatter
# ===----------------------------------------------------------------------===


@core.extern
def scatter(dst: core.tensor, src: core.tensor, offset: core.tensor, mask: Optional[core.tensor],
            _builder=None) -> core.tensor:
    """
    Scatter `src` tensor into `dst` tensor at locations specified by `offset`.

    Parameters:
    - dst: Destination tensor to write into.
    - src: Source tensor containing the values to be scattered. The leading (batch) dimensions of this tensor must either:
        1. Exactly match the corresponding batch dimensions of the offset tensor, or
        2. Be broadcast-compatible with the offset's batch dimensions (following standard broadcasting rules).
        The remaining dimensions specify the values to be written at each scattered location.
    - offset: Tensor specifying the indices where values in `src` will be written to `dst`.
              The last dimension of `offset` represents index positions in `dst`.
    - mask: Optional boolean tensor that masks which elements of `src` are written.
    - _builder: IR builder used to emit the underlying operations.

    Returns:
    - A new tensor that is the result of scattering `src` into `dst`.

    Usage example:
        >>> output = scatter(dst_tensor, src_tensor, offset_tensor, mask_tensor)
    """

    dst_shape = dst.shape
    src_shape = src.shape
    offset_shape = offset.shape

    # Determine batch dimensions (everything except last dim of offset)
    batch_dims = len(offset_shape) - 1
    offset_batch_shape = offset_shape[:batch_dims]
    src_batch_shape = src_shape[:batch_dims]

    # Check broadcasting compatibility and collect broadcasted dims
    broadcasted = []
    is_broadcasted = False
    for i in range(1, batch_dims + 1):
        src_dim = src_batch_shape[-i] if i <= len(src_batch_shape) else 1
        offset_dim = offset_batch_shape[-i]
        if src_dim != offset_dim:
            if src_dim == 1:
                broadcasted.insert(0, batch_dims - i)
                is_broadcasted = True
            else:
                raise ValueError(f"Invalid broadcast: src_dim {src_dim} != 1")
                break

    location = _builder.get_loc()
    pt = _builder.get_insertion_point()
    scatter_ext = mlu.scatter(location, pt, dst.handle, src.handle, offset.handle, mask.handle if mask else None,
                              broadcasted if is_broadcasted else [])
    scatter_ext = semantic.wrap_tensor(scatter_ext, src.type.scalar, dst.type.shape)
    return scatter_ext
