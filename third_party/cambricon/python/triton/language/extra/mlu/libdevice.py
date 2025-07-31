from triton.language import core, semantic


def is_block_arg(arg):
    arg_is_block = True
    if isinstance(arg, core.constexpr):
        arg_is_block = False
    else:
        if not arg.type.is_block():
            arg_is_block = False
    return arg_is_block


def is_cycle_args(arg0, arg1):
    arg0_shape = arg0.shape
    arg1_shape = arg1.shape
    need_broadcast = False
    for i in reversed(range(0, len(arg1_shape))):
        if not need_broadcast:
            if arg1_shape[i] == arg0_shape[i]:
                continue
            need_broadcast = True
        if arg0_shape[i] % arg1_shape[i] != 0:
            return False
    return True


@core.extern
def bitwise_and(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_and_bool", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_and_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_and_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_and_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_and_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_and_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_and_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_and_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_and_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int1"),
                    core.dtype("int1"),
                ): ("__cn_scalar_and_bool", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_and_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_and_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_and_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_and_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_and_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_and_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_and_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_and_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def bitwise_or(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_or_bool", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_or_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_or_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_or_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_or_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_or_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_or_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_or_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_or_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int1"),
                    core.dtype("int1"),
                ): ("__cn_scalar_or_bool", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_or_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_or_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_or_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_or_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_or_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_or_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_or_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_or_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def bitwise_not(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_not_bool", core.dtype("int8")),
                (core.dtype("int8"), ): ("__cn_vector_not_s8", core.dtype("int8")),
                (core.dtype("uint8"), ): ("__cn_vector_not_u8", core.dtype("uint8")),
                (core.dtype("int16"), ): ("__cn_vector_not_s16", core.dtype("int16")),
                (core.dtype("uint16"), ): ("__cn_vector_not_u16", core.dtype("uint16")),
                (core.dtype("int32"), ): ("__cn_vector_not_s32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__cn_vector_not_u32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__cn_vector_not_s64", core.dtype("int64")),
                (core.dtype("uint64"), ): ("__cn_vector_not_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int1"), ): ("__cn_scalar_not_bool", core.dtype("int1")),
                (core.dtype("int8"), ): ("__cn_scalar_not_s8", core.dtype("int8")),
                (core.dtype("uint8"), ): ("__cn_scalar_not_u8", core.dtype("uint8")),
                (core.dtype("int16"), ): ("__cn_scalar_not_s16", core.dtype("int16")),
                (core.dtype("uint16"), ): ("__cn_scalar_not_u16", core.dtype("uint16")),
                (core.dtype("int32"), ): ("__cn_scalar_not_s32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__cn_scalar_not_u32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__cn_scalar_not_s64", core.dtype("int64")),
                (core.dtype("uint64"), ): ("__cn_scalar_not_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def xor(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_xor_bool", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_xor_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_xor_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_xor_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_xor_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_xor_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_xor_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_xor_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_xor_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int1"),
                    core.dtype("int1"),
                ): ("__cn_scalar_xor_bool", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_xor_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_xor_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_xor_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_xor_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_xor_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_xor_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_xor_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_xor_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_isinf_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_vector_isinf_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_vector_isinf_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_isinf_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_scalar_isinf_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_scalar_isinf_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_isnan_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_vector_isnan_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_vector_isnan_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_isnan_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_scalar_isnan_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_scalar_isnan_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def isfinited(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_isfinite_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_vector_isfinite_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_vector_isfinite_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_isfinite_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_scalar_isfinite_f16", core.dtype("int16")),
                (core.dtype("bf16"), ): ("__cn_scalar_isfinite_bf16", core.dtype("int16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rd(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_rcp_f32_dn", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rcp_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rcp_f32_dn", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rcp_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_div_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_div_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_div_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_div_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rcp_ru(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_rcp_f32_up", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rcp_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rcp_f32_up", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rcp_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_div_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_div_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_div_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_div_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_rcp_f32_tz", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rcp_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rcp_f32_tz", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rcp_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_div_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_div_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_div_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_div_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("bf16"), ): ("__cn_vector_rcp_bf16_rn", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_vector_rcp_f32_rn", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rcp_f16_rn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rcp_f32_rn", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rcp_f16_rn", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_rcp_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_div_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_div_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_div_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_div_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_div_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_div_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def float2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rd(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_add_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_add_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_add_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_add_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def add_rn(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_add_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_add_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_add_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_add_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_add_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_add_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def add_ru(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_add_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_add_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_add_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_add_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def add_rz(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_add_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_add_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_add_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_add_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def add(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_add_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_add_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_add_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_add_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_add_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_add_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_add_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_add_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_add_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_add_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_add_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_add_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_add_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_add_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_add_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_add_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sub_rd(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_sub_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_sub_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_sub_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_sub_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sub_ru(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_sub_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_sub_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_sub_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_sub_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sub_rz(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_sub_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_sub_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_sub_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_sub_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sub_rn(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_sub_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_sub_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_sub_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_sub_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_sub_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_sub_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sub(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_sub_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_sub_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_sub_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_sub_s16", core.dtype("int16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_sub_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_sub_u32", core.dtype("uint32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_sub_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_sub_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_sub_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_sub_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_sub_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_sub_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_sub_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_sub_u32", core.dtype("uint32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_sub_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_sub_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul_rd(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mul_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mul_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_mul_f32_dn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_mul_f16_dn", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul_ru(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mul_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mul_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_mul_f32_up", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_mul_f16_up", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul_rz(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mul_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mul_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_mul_f32_tz", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_mul_f16_tz", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul_rn(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mul_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mul_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_mul_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_mul_f32_rn", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_mul_f16_rn", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_mul_bf16_rn", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_mul_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_mul_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_mul_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_mul_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_mul_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_mul_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_mul_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_mul_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_mul_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_mul_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_mul_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_mul_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_mul_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_mul_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_mul_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_mul_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mul_complex(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mul_complex_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mul_complex_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_mul_complex_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def mulhi(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_mulh_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_mulh_u32", core.dtype("uint32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_mulh_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_mulh_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_mulh_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_mulh_u32", core.dtype("uint32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_mulh_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_mulh_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def div(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_div_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_div_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_div_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_div_s16", core.dtype("int16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_div_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_div_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_div_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_div_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_div_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_div_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_div_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_div_s16", core.dtype("int16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_div_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_div_u32", core.dtype("uint32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_div_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_div_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_mod_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_mod_f16", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_mod_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_mod_f16", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def mod(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_mod_bf16", core.dtype("bf16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_mod_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_mod_u32", core.dtype("uint32")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_mod_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_mod_u16", core.dtype("uint16")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_mod_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_mod_u8", core.dtype("uint8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_mod_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_mod_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_mod_bf16", core.dtype("bf16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_mod_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_mod_u32", core.dtype("uint32")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_mod_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_mod_u16", core.dtype("uint16")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_mod_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_mod_u8", core.dtype("uint8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_mod_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_mod_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sqrt(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_sqrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_sqrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_sqrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_sqrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_sqrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_sqrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def float2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def accurate_sqrt(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_accurate_sqrt_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_accurate_sqrt_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def clz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_clz_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_vector_clz_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_vector_clz_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_vector_clz_s64", core.dtype("int64")),
                (core.dtype("uint8"), ): ("__cn_vector_clz_u8", core.dtype("uint8")),
                (core.dtype("uint16"), ): ("__cn_vector_clz_u16", core.dtype("uint16")),
                (core.dtype("uint32"), ): ("__cn_vector_clz_u32", core.dtype("uint32")),
                (core.dtype("uint64"), ): ("__cn_vector_clz_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("uint32"), ): ("__cn_scalar_clz_u32", core.dtype("uint32")),
                (core.dtype("int8"), ): ("__cn_scalar_clz_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_scalar_clz_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_scalar_clz_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_scalar_clz_s64", core.dtype("int64")),
                (core.dtype("uint8"), ): ("__cn_scalar_clz_u8", core.dtype("uint8")),
                (core.dtype("uint16"), ): ("__cn_scalar_clz_u16", core.dtype("uint16")),
                (core.dtype("uint64"), ): ("__cn_scalar_clz_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_rsqrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rsqrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_rsqrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rsqrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rsqrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_rsqrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def cbrt(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_cbrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_cbrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_cbrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_cbrt_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_cbrt_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_cbrt_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def abs(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_abs_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_vector_abs_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_vector_abs_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_vector_abs_s64", core.dtype("int64")),
                (core.dtype("fp32"), ): ("__cn_vector_abs_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_abs_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_abs_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_abs_f32", core.dtype("fp32")),
                (core.dtype("int8"), ): ("__cn_scalar_abs_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_scalar_abs_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_scalar_abs_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_scalar_abs_s64", core.dtype("int64")),
                (core.dtype("fp16"), ): ("__cn_scalar_abs_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_abs_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_exp2_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_exp2_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_exp2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_exp2_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_exp2_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_exp2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def negate(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_negate_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_vector_negate_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_vector_negate_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_vector_negate_s64", core.dtype("int64")),
                (core.dtype("fp16"), ): ("__cn_vector_negate_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_negate_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_vector_negate_f32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_scalar_negate_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_scalar_negate_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_scalar_negate_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_scalar_negate_s64", core.dtype("int64")),
                (core.dtype("fp16"), ): ("__cn_scalar_negate_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_negate_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_scalar_negate_f32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)


@core.extern
def max(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_max_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_max_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_max_bf16", core.dtype("bf16")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_max_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_max_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_max_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_max_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_max_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_max_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_max_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_max_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_max_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_max_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_max_bf16", core.dtype("bf16")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_max_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_max_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_max_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_max_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_max_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_max_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_max_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_max_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_max(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_max_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_fast_max_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_fast_max_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def nan_max(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_nan_max_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_nan_max_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_nan_max_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_nan_max_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_nan_max_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_nan_max_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_nan_max(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_fast_nan_max_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_nan_max_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_fast_nan_max_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def min(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_min_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_min_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_min_bf16", core.dtype("bf16")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_min_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_min_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_min_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_min_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_min_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_min_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_min_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_min_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_min_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_min_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_min_bf16", core.dtype("bf16")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_min_u8", core.dtype("uint8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_min_s8", core.dtype("int8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_min_u16", core.dtype("uint16")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_min_s16", core.dtype("int16")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_min_u32", core.dtype("uint32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_min_s32", core.dtype("int32")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_min_u64", core.dtype("uint64")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_min_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_min(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_min_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_fast_min_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_fast_min_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def nan_min(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_nan_min_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_nan_min_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_nan_min_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_nan_min_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_nan_min_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_nan_min_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_nan_min(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_fast_nan_min_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_nan_min_f32", core.dtype("fp32")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_fast_nan_min_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def round(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_round_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_round_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_round_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_round_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_round_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_round_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_trunc_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_trunc_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_trunc_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_trunc_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_trunc_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_trunc_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_ceil_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_ceil_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_ceil_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_ceil_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_ceil_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_ceil_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def ctz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_ctz_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_vector_ctz_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_vector_ctz_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_vector_ctz_s64", core.dtype("int64")),
                (core.dtype("uint8"), ): ("__cn_vector_ctz_u8", core.dtype("uint8")),
                (core.dtype("uint16"), ): ("__cn_vector_ctz_u16", core.dtype("uint16")),
                (core.dtype("uint32"), ): ("__cn_vector_ctz_u32", core.dtype("uint32")),
                (core.dtype("uint64"), ): ("__cn_vector_ctz_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_scalar_ctz_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_scalar_ctz_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_scalar_ctz_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_scalar_ctz_s64", core.dtype("int64")),
                (core.dtype("uint8"), ): ("__cn_scalar_ctz_u8", core.dtype("uint8")),
                (core.dtype("uint16"), ): ("__cn_scalar_ctz_u16", core.dtype("uint16")),
                (core.dtype("uint32"), ): ("__cn_scalar_ctz_u32", core.dtype("uint32")),
                (core.dtype("uint64"), ): ("__cn_scalar_ctz_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_floor_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_floor_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_floor_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_floor_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_floor_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_floor_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def rint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_rint_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_rint_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_rint_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_rint_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_rint_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_rint_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def nearbyint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_nearbyint_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_nearbyint_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_nearbyint_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_nearbyint_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_nearbyint_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_nearbyint_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def erf(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_erf_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_erf_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_erf_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_erf_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_erf_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_erf_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_erf(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_fast_erf_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_fast_erf_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_fast_erf_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def erfc(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_erfc_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_erfc_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_erfc_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_erfc_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_erfc_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_erfc_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def erfcx(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_erfcx_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_erfcx_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_erfcx_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_erfcx_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_erfcx_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_erfcx_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_exp_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_exp_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_exp_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_exp_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_exp_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_exp_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def erfinv(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_erfinv_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_erfinv_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_erfinv_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_erfinv_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_erfinv_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_erfinv_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_log_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_log_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_log_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_log_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_log_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_log_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def erfcinv(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_erfcinv_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_erfcinv_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_erfcinv_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_erfcinv_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_erfcinv_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_erfcinv_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_log2_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_log2_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_log2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_log2_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_log2_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_log2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sincos(arg0, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__cn_vector_sincos_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_sincos_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_sin_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_sin_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_sin_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_sin_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_sin_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_sin_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_cos_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_cos_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_cos_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_cos_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_cos_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_cos_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_tan_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_tan_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_tan_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_tan_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_tan_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_tan_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_asin_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_asin_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_asin_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_asin_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_asin_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_asin_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_acos_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_acos_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_acos_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_acos_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_acos_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_acos_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_asinh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_asinh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_asinh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_asinh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_asinh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_asinh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_log1p_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_log1p_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_log1p_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_log1p_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_log1p_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_log1p_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_acosh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_acosh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_acosh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_acosh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_acosh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_acosh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_atanh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_atanh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_atanh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_atanh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_atanh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_atanh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_atan_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_atan_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_atan_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_atan_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_atan_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_atan_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def atan2(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_atan2_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_atan2_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_atan2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_atan2_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_atan2_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_atan2_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_sinh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_sinh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_sinh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_sinh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_sinh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_sinh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_cosh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_cosh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_cosh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_cosh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_cosh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_cosh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_tanh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_tanh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_tanh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_tanh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_tanh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_tanh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_tanh(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_fast_tanh_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_fast_tanh_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_fast_tanh_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_expf(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_exp_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def scalbn(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("int32"),
                ): ("__cn_vector_scalbn_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("int32"),
                ): ("__cn_vector_scalbn_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("int32"),
                ): ("__cn_vector_scalbn_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_scalbn_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("int32"),
                ): ("__cn_scalar_scalbn_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("int32"),
                ): ("__cn_scalar_scalbn_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def float2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp10(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_exp10_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_exp10_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_exp10_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_exp10_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_exp10_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_exp10_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_exp10f(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_exp10_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def expm1(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_expm1_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_expm1_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_expm1_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_expm1_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_expm1_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_expm1_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s32_rn", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s32_rn", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_hypot_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_hypot_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_hypot_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_hypot_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_hypot_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_hypot_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def lgamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_lgamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_lgamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s32_tz", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s32_tz", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fast_lgamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_lgamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_log(arg0, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_fast_log_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_log_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fast_log_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_log2f(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_log2_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def log10(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_log10_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_log10_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_log10_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_log10_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_log10_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_log10_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_log10(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_log10_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def normcdf(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_normcdf_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_normcdf_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_normcdf_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_normcdf_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_scalar_normcdf_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_normcdf_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1) or is_block_arg(arg2):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
                arg2,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_fma_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_fma_f16", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
            arg2,
        ], {
            (
                core.dtype("fp16"),
                core.dtype("fp16"),
                core.dtype("fp16"),
            ): ("__cn_scalar_fma_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_pow_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_pow_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_pow_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_pow_f32", core.dtype("fp32")),
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_pow_f16", core.dtype("fp16")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_pow_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def fast_powf(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__cn_vector_fast_pow_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_powi(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("int32"),
            ): ("__cn_vector_fast_powi_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def signbit(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_signbit_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_vector_signbit_f16", core.dtype("int32")),
                (core.dtype("bf16"), ): ("__cn_vector_signbit_bf16", core.dtype("int32")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_scalar_signbit_f32", core.dtype("int32")),
                (core.dtype("fp16"), ): ("__cn_scalar_signbit_f16", core.dtype("int32")),
                (core.dtype("bf16"), ): ("__cn_scalar_signbit_bf16", core.dtype("int32")),
            }, is_pure=True, _builder=_builder)


@core.extern
def sign(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp16"), ): ("__cn_vector_sign_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_sign_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_vector_sign_f32", core.dtype("fp32")),
                (core.dtype("int8"), ): ("__cn_vector_sign_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_vector_sign_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_vector_sign_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_vector_sign_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp16"), ): ("__cn_scalar_sign_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_sign_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_scalar_sign_f32", core.dtype("fp32")),
                (core.dtype("int8"), ): ("__cn_scalar_sign_s8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__cn_scalar_sign_s16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__cn_scalar_sign_s32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__cn_scalar_sign_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def nan_sign(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp16"), ): ("__cn_vector_nan_sign_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_nan_sign_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_vector_nan_sign_f32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp16"), ): ("__cn_scalar_nan_sign_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_scalar_nan_sign_bf16", core.dtype("bf16")),
                (core.dtype("fp32"), ): ("__cn_scalar_nan_sign_f32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)


@core.extern
def shift_left(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_shift_left_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_shift_left_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_shift_left_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_shift_left_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_shift_left_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_shift_left_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_shift_left_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_shift_left_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_shift_left_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_shift_left_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_shift_left_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_shift_left_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_shift_left_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_shift_left_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_shift_left_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_shift_left_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def shift_right_logical(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_shift_right_logical_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_shift_right_logical_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_shift_right_logical_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_shift_right_logical_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_shift_right_logical_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_shift_right_logical_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_shift_right_logical_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_shift_right_logical_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_shift_right_logical_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_shift_right_logical_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_shift_right_logical_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_shift_right_logical_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_shift_right_logical_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_shift_right_logical_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_shift_right_logical_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_shift_right_logical_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def shift_right_arithmetic(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_shift_right_arithmetic_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_shift_right_arithmetic_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_shift_right_arithmetic_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_shift_right_arithmetic_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_shift_right_arithmetic_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_shift_right_arithmetic_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_shift_right_arithmetic_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_shift_right_arithmetic_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_shift_right_arithmetic_s8", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_shift_right_arithmetic_u8", core.dtype("uint8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_shift_right_arithmetic_s16", core.dtype("int16")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_shift_right_arithmetic_u16", core.dtype("uint16")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_shift_right_arithmetic_s32", core.dtype("int32")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_shift_right_arithmetic_u32", core.dtype("uint32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_shift_right_arithmetic_s64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_shift_right_arithmetic_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def float2half_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_f16_tz", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_f16_tz", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2double(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def double2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp64"), ): ("__cn_vector_cast_f64_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def double2float_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp64"), ): ("__cn_vector_cast_f64_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def double2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp64"), ): ("__cn_vector_cast_f64_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def double2ll_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp64"), ): ("__cn_vector_cast_f64_to_s64_tz", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def float2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2byte_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s8_tz", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s8_tz", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2short_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s16_tz", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s16_tz", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2int_ru(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s32_up", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s32_up", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rd(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s32_dn", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s32_dn", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ubyte_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u8_tz", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u8_tz", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u32_tz", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u32_tz", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ushort_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u16_tz", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u16_tz", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u32_rn", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u32_rn", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_f32_tz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_f32_tz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f32_tz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_f32_tz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2double(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f64", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ll2double_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f64_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ull2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_f32_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s64_tz", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s64_tz", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_s64_rn", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_s64_rn", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_llrint_f32", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_llrint_f32", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def lrint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_lrint_f32", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_lrint_f32", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_cast_f32_to_u64_tz", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_cast_f32_to_u64_tz", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2byte_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s8_tz", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s8_tz", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2short_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s16_tz", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s16_tz", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2int_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s32_tz", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s32_tz", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ll_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_s64_tz", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_s64_tz", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ubyte_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u8_tz", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u8_tz", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ushort_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u16_tz", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u16_tz", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2uint_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u32_tz", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u32_tz", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2ull_rz(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_u64_tz", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_u64_tz", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2half_rn(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_f16_rn", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162byte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_s8", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162ubyte(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_u8", core.dtype("uint8")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162short(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_s16", core.dtype("int16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162ushort(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_u16", core.dtype("uint16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162int(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_s32", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162uint(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_u32", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162ll(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_s64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162ull(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_u64", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162half(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_f16", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def bfloat162float(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_vector_cast_bf16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("bf16"), ): ("__cn_scalar_cast_bf16_to_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_vector_cast_s8_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int8"), ): ("__cn_scalar_cast_s8_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ubyte2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_vector_cast_u8_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint8"), ): ("__cn_scalar_cast_u8_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def short2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_vector_cast_s16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int16"), ): ("__cn_scalar_cast_s16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ushort2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_vector_cast_u16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint16"), ): ("__cn_scalar_cast_u16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def int2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_vector_cast_s32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__cn_scalar_cast_s32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def uint2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_vector_cast_u32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint32"), ): ("__cn_scalar_cast_u32_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ll2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_vector_cast_s64_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("int64"), ): ("__cn_scalar_cast_s64_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ull2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_vector_cast_u64_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("uint64"), ): ("__cn_scalar_cast_u64_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def half2bfloat16(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_vector_cast_f16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__cn_scalar_cast_f16_to_bf16", core.dtype("bf16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def eq(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_eq_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_eq_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_eq_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_eq_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_eq_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_eq_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_eq_s64", core.dtype("int8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_eq_u64", core.dtype("uint8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_eq_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_eq_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_eq_u32", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_eq_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_eq_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_eq_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_eq_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_eq_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_eq_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_eq_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_eq_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_eq_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_eq_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_eq_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def eq_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_eq_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_eq_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_eq_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def eq_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_eq_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_eq_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_eq_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def eq_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_eq_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_eq_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ne(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ne_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ne_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ne_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_ne_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_ne_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_ne_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_ne_s64", core.dtype("int8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_ne_u64", core.dtype("uint8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_ne_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_ne_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_ne_u32", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_ne_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_ne_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_ne_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_ne_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_ne_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_ne_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_ne_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_ne_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_ne_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_ne_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_ne_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def ne_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ne_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ne_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ne_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ne_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ne_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ne_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ne_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ne_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_ne_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_ne_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def lt(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_lt_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_lt_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_lt_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_lt_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_lt_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_lt_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_lt_s64", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_lt_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_lt_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_lt_u32", core.dtype("uint8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_lt_u64", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_lt_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_lt_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_lt_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_lt_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_lt_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_lt_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_lt_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_lt_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_lt_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_lt_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_lt_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def lt_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_lt_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_lt_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_lt_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def lt_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_lt_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_lt_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_lt_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def lt_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_lt_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_lt_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def le(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_le_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_le_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_le_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_le_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_le_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_le_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_le_s64", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_le_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_le_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_le_u32", core.dtype("uint8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_le_u64", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_le_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_le_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_le_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_le_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_le_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_le_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_le_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_le_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_le_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_le_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_le_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def le_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_le_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_le_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_le_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def le_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_le_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_le_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_le_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def le_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_le_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_le_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def gt(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_gt_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_gt_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_gt_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_gt_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_gt_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_gt_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_gt_s64", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_gt_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_gt_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_gt_u32", core.dtype("uint8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_gt_u64", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_gt_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_gt_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_gt_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_gt_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_gt_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_gt_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_gt_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_gt_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_gt_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_gt_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_gt_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def gt_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_gt_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_gt_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_gt_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def gt_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_gt_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_gt_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_gt_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def gt_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_gt_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_gt_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ge(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ge_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ge_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ge_bf16", core.dtype("int8")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_vector_ge_s8", core.dtype("int8")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_vector_ge_s16", core.dtype("int8")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_ge_s32", core.dtype("int8")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_ge_s64", core.dtype("int8")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_vector_ge_u8", core.dtype("uint8")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_vector_ge_u16", core.dtype("uint8")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_vector_ge_u32", core.dtype("uint8")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_ge_u64", core.dtype("uint8")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_scalar_ge_f16", core.dtype("int1")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_scalar_ge_f32", core.dtype("int1")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_scalar_ge_bf16", core.dtype("int1")),
                (
                    core.dtype("int8"),
                    core.dtype("int8"),
                ): ("__cn_scalar_ge_s8", core.dtype("int1")),
                (
                    core.dtype("int16"),
                    core.dtype("int16"),
                ): ("__cn_scalar_ge_s16", core.dtype("int1")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_scalar_ge_s32", core.dtype("int1")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_scalar_ge_s64", core.dtype("int1")),
                (
                    core.dtype("uint8"),
                    core.dtype("uint8"),
                ): ("__cn_scalar_ge_u8", core.dtype("uint1")),
                (
                    core.dtype("uint16"),
                    core.dtype("uint16"),
                ): ("__cn_scalar_ge_u16", core.dtype("uint1")),
                (
                    core.dtype("uint32"),
                    core.dtype("uint32"),
                ): ("__cn_scalar_ge_u32", core.dtype("uint1")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_scalar_ge_u64", core.dtype("uint1")),
            }, is_pure=True, _builder=_builder)


@core.extern
def ge_order(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ge_order_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ge_order_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ge_order_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ge_unorder(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_ge_unorder_f16", core.dtype("int8")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_ge_unorder_f32", core.dtype("int8")),
                (
                    core.dtype("bf16"),
                    core.dtype("bf16"),
                ): ("__cn_vector_ge_unorder_bf16", core.dtype("int8")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ge_out(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_ge_s64_outs64", core.dtype("int64")),
                (
                    core.dtype("uint64"),
                    core.dtype("uint64"),
                ): ("__cn_vector_ge_u64_outu64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def popc(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_vector_popcnt_s8", core.dtype("int8")),
                (core.dtype("uint8"), ): ("__cn_vector_popcnt_u8", core.dtype("uint8")),
                (core.dtype("int16"), ): ("__cn_vector_popcnt_s16", core.dtype("int16")),
                (core.dtype("uint16"), ): ("__cn_vector_popcnt_u16", core.dtype("uint16")),
                (core.dtype("int32"), ): ("__cn_vector_popcnt_s32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__cn_vector_popcnt_u32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__cn_vector_popcnt_s64", core.dtype("int64")),
                (core.dtype("uint64"), ): ("__cn_vector_popcnt_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("int8"), ): ("__cn_scalar_popcnt_s8", core.dtype("int8")),
                (core.dtype("uint8"), ): ("__cn_scalar_popcnt_u8", core.dtype("uint8")),
                (core.dtype("int16"), ): ("__cn_scalar_popcnt_s16", core.dtype("int16")),
                (core.dtype("uint16"), ): ("__cn_scalar_popcnt_u16", core.dtype("uint16")),
                (core.dtype("int32"), ): ("__cn_scalar_popcnt_s32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__cn_scalar_popcnt_u32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__cn_scalar_popcnt_s64", core.dtype("int64")),
                (core.dtype("uint64"), ): ("__cn_scalar_popcnt_u64", core.dtype("uint64")),
            }, is_pure=True, _builder=_builder)


@core.extern
def zeta(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__cn_vector_zeta_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__cn_scalar_zeta_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def trigamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_trigamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_trigamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fast_trigamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_trigamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def digamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_digamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_scalar_digamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fast_digamma(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_digamma_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def iota(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return core.extern_elementwise(
            "", "", [
                arg0,
                arg1,
            ], {
                (
                    core.dtype("fp16"),
                    core.dtype("fp16"),
                ): ("__cn_vector_iota_f16", core.dtype("fp16")),
                (
                    core.dtype("fp32"),
                    core.dtype("fp32"),
                ): ("__cn_vector_iota_f32", core.dtype("fp32")),
                (
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("__cn_vector_iota_s32", core.dtype("int32")),
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("__cn_vector_iota_s64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def frexp(arg0, arg1, _builder=None):
    if is_block_arg(arg0) or is_block_arg(arg1):
        return None
    else:
        return core.extern_elementwise("", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp32"),
                core.dtype("int32"),
            ): ("__cn_scalar_frexp_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ultra_gelu(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_ultra_gelu_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_silu(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_silu_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ultra_silu(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_ultra_silu_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_ultra_silu_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_ultra_silu_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ultra_silubp(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise(
            "", "", [
                arg0,
            ], {
                (core.dtype("fp32"), ): ("__cn_vector_ultra_silubp_f32", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__cn_vector_ultra_silubp_f16", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__cn_vector_ultra_silubp_bf16", core.dtype("bf16")),
            }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def fast_sigmoid(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_fast_sigmoid_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


@core.extern
def ultra_sigmoid(arg0, _builder=None):
    if is_block_arg(arg0):
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__cn_vector_ultra_sigmoid_f32", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)
    else:
        return None


# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo, subsequenceHi, innerRounds
@core.extern
def philox(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, _builder=None):
    ret_type = core.block_type(core.dtype("uint32"), [arg0.value, 4])
    assert (not is_block_arg(arg0) and not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3)
            and not is_block_arg(arg4) and not is_block_arg(arg5) and not is_block_arg(arg6)
            and not is_block_arg(arg7)), 'philox: all args must be scalars'
    args = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_philox_u32", args, ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo
@core.extern
def philox_v2(arg0, arg1, arg2, arg3, arg4, arg5, _builder=None):
    assert (not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3) and not is_block_arg(arg4)
            and not is_block_arg(arg5)), 'philox_v2: all args must be scalars'
    assert (isinstance(arg0, core.constexpr)), 'philox_v2: outGroups(arg0) must be constexpr'
    assert (arg0.value % 128 == 0), 'philox_v2: outGroups(arg0) must be divided by 128'
    ret_type = core.block_type(core.dtype("uint32"), [arg0.value, 4])
    args = [arg0, arg1, arg2, arg3, arg4, arg5]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_philox_v2_u32", args, ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


# ARGS: outGroups, seedLo, seedHi, offsetLo, offsetHi, subsequenceLo, innerRounds, subsequenceLimit
@core.extern
def philox_v3(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, _builder=None):
    assert (not is_block_arg(arg1) and not is_block_arg(arg2) and not is_block_arg(arg3)
            and not is_block_arg(arg4)), 'philox_v3: all args must be scalars'
    assert (isinstance(arg0, core.constexpr)), 'philox_v3: outGroups(arg0) must be constexpr'
    assert (isinstance(arg5, core.constexpr)), 'philox_v3: subsequenceLo(arg5) must be constexpr'
    assert (isinstance(arg6, core.constexpr)), 'philox_v3: innerRounds(arg6) must be constexpr'
    assert (isinstance(arg7, core.constexpr)), 'philox_v3: subsequenceLimit(arg7) must be constexpr'
    assert (arg0.value % 128 == 0), 'philox_v3: outGroups(arg0) must be divided by 128'
    assert (arg5.value % 128 == 0), 'philox_v3: subsequenceLo(arg5) must be divided by 128'
    assert (arg6.value in [2, 4, 6, 8, 10]), 'philox_v3: innerRounds(arg0) must be divided by 128'
    assert (arg7.value % 128 == 0), 'philox_v3: subsequenceLimit(arg7) must be divided by 128'
    ret_type = core.block_type(core.dtype("uint32"), [arg0.value, 4])
    args = [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_philox_v3_u32", args, ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silu_float2half(arg0, _builder=None):
    ret_type = core.block_type(core.float16, arg0.shape)
    args = [arg0]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silu_f32_outf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silu_float2bfloat16(arg0, _builder=None):
    ret_type = core.block_type(core.bfloat16, arg0.shape)
    args = [arg0]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silu_f32_outbf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silubp_float2half(arg0, _builder=None):
    ret_type = core.block_type(core.float16, arg0.shape)
    args = [arg0]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silubp_f32_outf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silubp_float2bfloat16(arg0, _builder=None):
    ret_type = core.block_type(core.bfloat16, arg0.shape)
    args = [arg0]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silubp_f32_outbf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silu_mul_float2half(arg0, arg1, _builder=None):
    ret_type = core.block_type(core.float16, arg0.shape)
    assert (not is_block_arg(arg1)), 'ultra_silu_mul_float2half: arg1 must be a scalar'
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silu_mul_scalar_f32_outf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silu_mul_float2bfloat16(arg0, arg1, _builder=None):
    ret_type = core.block_type(core.bfloat16, arg0.shape)
    assert (not is_block_arg(arg1)), 'ultra_silu_mul_float2bfloat16: arg1 must be a scalar'
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silu_mul_scalar_f32_outbf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silubp_mul_float2half(arg0, arg1, _builder=None):
    ret_type = core.block_type(core.float16, arg0.shape)
    assert (not is_block_arg(arg1)), 'ultra_silubp_mul_float2half: arg1 must be a scalar'
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silubp_mul_scalar_f32_outf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


@core.extern
def ultra_silubp_mul_float2bfloat16(arg0, arg1, _builder=None):
    ret_type = core.block_type(core.bfloat16, arg0.shape)
    assert (not is_block_arg(arg1)), 'ultra_silubp_mul_float2bfloat16: arg1 must be a scalar'
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_silubp_mul_scalar_f32_outbf16", args,
                                             ret_type.to_ir(_builder), True)
    return core.tensor(ret, ret_type)


# result = exp2((arg0 - arg1) * arg2)
@core.extern
def cycle_sub_mul_exp(arg0, arg1, arg2, _builder=None):
    ret_shape = arg0.shape
    sub_val = arg1
    n = arg0.type.numel
    if not is_block_arg(arg1):
        assert n // 64 >= 16, "arg0 element num must be more than 64 * 16"
        sub_val = core.full([64], arg1, dtype=arg0.dtype, _builder=_builder)
    else:
        assert is_cycle_args(arg0, arg1)
        assert arg0.shape[1:] == arg1.shape[1:], 'arg0 and arg1 shape must equal except the hightest dim'
        n_short = sub_val.type.numel
        sub_val = core.reshape(sub_val, (n_short), can_reorder=True, _builder=_builder)
        assert (
            (n % n_short == 0) and (n // n_short >= 16)
        ), "arg0 element num must be divisible by arg1 element num, and the ratio between the two must be greater than 16."
    arg0 = core.reshape(arg0, (n), can_reorder=True, _builder=_builder)
    ret_type = core.block_type(arg0.dtype, arg0.shape)
    args = [arg0, sub_val, arg2]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = _builder.create_extern_elementwise("", "", "__cn_vector_cycle_sub_exp_f32", args, ret_type.to_ir(_builder),
                                             True)
    ret_tensor = core.tensor(ret, ret_type)
    return core.reshape(ret_tensor, ret_shape, can_reorder=True, _builder=_builder)


@core.extern
def fast_dividef(arg0, arg1, _builder=None):
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = None
    if is_block_arg(arg0) and is_block_arg(arg1):
        ret_type = core.block_type(arg0.dtype, arg0.shape)
        if arg0.dtype.is_fp16():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_fast_div_f16_rn", args,
                                                     ret_type.to_ir(_builder), True)
        if arg0.dtype.is_bf16():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_fast_div_bf16_rn", args,
                                                     ret_type.to_ir(_builder), True)
        if arg0.dtype.is_fp32():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_fast_div_f32_rn", args,
                                                     ret_type.to_ir(_builder), True)
    elif is_block_arg(arg0) and not is_block_arg(arg1):
        ret_type = core.block_type(arg0.dtype, arg0.shape)
        if arg0.dtype.is_fp16():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_fast_div_scalar_f16_rn", args,
                                                     ret_type.to_ir(_builder), True)
        if arg0.dtype.is_bf16():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_fast_div_scalar_bf16_rn", args,
                                                     ret_type.to_ir(_builder), True)
    elif not is_block_arg(arg0) and is_block_arg(arg1):
        ret_type = core.block_type(arg1.dtype, arg1.shape)
        if arg0.dtype.is_fp16():
            ret = _builder.create_extern_elementwise("", "", "__cn_scalar_fast_div_vector_f16_rn", args,
                                                     ret_type.to_ir(_builder), True)
        if arg0.dtype.is_bf16():
            ret = _builder.create_extern_elementwise("", "", "__cn_scalar_fast_div_vector_bf16_rn", args,
                                                     ret_type.to_ir(_builder), True)
    if ret is not None:
        return core.tensor(ret, ret_type)
    else:
        return None


@core.extern
def ultra_pow(arg0, arg1, _builder=None):
    args = [arg0, arg1]
    args = [semantic.to_tensor(arg, _builder) for arg in args]
    args = [arg.handle for arg in args]
    ret = None
    if is_block_arg(arg0) and is_block_arg(arg1):
        ret_type = core.block_type(arg0.dtype, arg0.shape)
        if arg0.dtype.is_fp32():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_pow_f32", args,
                                                     ret_type.to_ir(_builder), True)
    elif is_block_arg(arg0) and not is_block_arg(arg1):
        ret_type = core.block_type(arg0.dtype, arg0.shape)
        if arg0.dtype.is_fp32():
            ret = _builder.create_extern_elementwise("", "", "__cn_vector_ultra_pow_scalar_f32", args,
                                                     ret_type.to_ir(_builder), True)
    elif not is_block_arg(arg0) and is_block_arg(arg1):
        ret_type = core.block_type(arg1.dtype, arg1.shape)
        if arg0.dtype.is_fp32():
            ret = _builder.create_extern_elementwise("", "", "__cn_scalar_ultra_pow_vector_f32", args,
                                                     ret_type.to_ir(_builder), True)
    if ret is not None:
        return core.tensor(ret, ret_type)
    else:
        return None


popcnt = popc
fast_div_rn = fast_dividef
finitef = isfinited
