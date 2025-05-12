
//===---------------------- Tx81Ops.td ------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Definition of TsingMicro's Tx81 ML accelerator operations.
//
// Data format supported by Tx81 ML accelerator are:
//  f16,fp16,tf32,fp32
//
// For Tx81 accelerator unsupported data type, we can either convert it by
// using `TsmConvert`, or lower the operations to run on RISC-V controller
// instead.
//
// NOTE: CHANGING THE ARGUMENTS AND RETURNS OF ANY OPS RESULT IN THE CHANGE OF
// THEIR RUNTIME INTERFACE AND IMPLEMENTATION IN crt/Target/Tx81.
//
//===----------------------------------------------------------------------===//

#ifndef TSINGMICRO_TX81_OPS
#define TSINGMICRO_TX81_OPS

include "tsingmicro-tx81/Dialect/IR/Tx81AttrDefs.td"
include "tsingmicro-tx81/Dialect/IR/Tx81Types.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/IR/OpBase.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

class Tx81Op<string mnemonic, list<Trait> traits = []> :
  Op<Tx81Dialect, mnemonic,
     !listconcat(traits, [/*TensorSizeTrait, VerifyTensorLayoutsTrait*/])> {
}

def MemRefOrInt
  : AnyTypeOf<[AnyMemRef, AnySignlessIntegerOrIndex],
              "MemRef or Int as address type.", "::mlir::Type">;

// =============================================================================
// 4.8/4.9 DDR and SPM transfer ops
// =============================================================================

def RdmaOp : Tx81Op<"rdma", [
    AttrSizedOperandSegments,
    PredOpTrait<"Constrain shape to 4d.",
      CPred<"cast<tx::RdmaOp>($_op).getShape().size() == 4">>,
    PredOpTrait<"Constrain strides to 3d.",
      CPred<"cast<tx::RdmaOp>($_op).getStrides().size() == 3">>
      ]> {

  let summary = "Copy data from global memory DDR(dram) to per thread local SPM(sram)";

  let description = [{
    Copy data from global memory DDR(dram) to per thread local SPM(sram).
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,                 // The source address in DDR
    MemRefOrInt:$target,                 // The target address in SPM
    Variadic<Index>:$shape,      // HHWC shape
    Variadic<Index>:$strides,    // 3 dim strides
    I32Attr:$fmt
  );

  let builders = [
    OpBuilder<(ins "MemRefType":$resultType,
      "Value":$source, "Value":$target,
      "ArrayRef<Value>":$shape,
      "ArrayRef<Value>":$strides,
      "IntegerAttr":$fmt
    )>
  ];

  let results = (outs I64:$dst); // The dest address in SPM
}

def WdmaOp : Tx81Op<"wdma", [
    AttrSizedOperandSegments,
    PredOpTrait<"Constrain shape to 4d.",
      CPred<"cast<tx::WdmaOp>($_op).getShape().size() == 4">>,
    PredOpTrait<"Constrain strides to 3d.",
      CPred<"cast<tx::WdmaOp>($_op).getStrides().size() == 3">>
      ]> {
  let summary = "Copy data from per thread local SPM(sram) to global memory DDR(dram)";

  let description = [{
    Copy data from per thread local SPM(sram) to global memory DDR(dram).
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,                 // The source address in DDR
    MemRefOrInt:$target,                 // The target address in SPM
    Variadic<Index>:$shape,      // HHWC shape
    Variadic<Index>:$strides,    // 3 dim strides
    I32Attr:$fmt
  );

  let builders = [
    OpBuilder<(ins "MemRefType":$resultType,
      "Value":$source, "Value":$target,
      "ArrayRef<Value>":$shape,
      "ArrayRef<Value>":$strides,
      "IntegerAttr":$fmt
    )>
  ];

  let results = (outs I64:$dst); // The dest address in DDR
}

// =============================================================================
// 4.4~6 TsmConv, TsmDepthwiseConv, TsmBackwardConv
// =============================================================================

def ConvOp : Tx81Op<"conv", [Pure]> {
  let summary = "Convolution engine intrinsic runtime API";

  let description = [{
    A common convolution op for TsmConv, TsmDepthwiseConv, TsmBackwardConv.
    This TsmConv is not a 1 to 1 map to TsingMicro's TsmConv intrinsic, it is
    the wrap of all APIs related to TsmConv. This op wraps the following APIs:
      TsmNewConv, TsmDeleteConv, AddInput, AddWeight, AddBias, AddOutput,
      SetOpType, SetNegativeAxisScale, SetPositiveAxisScale, SetSparse, SetPsum,
      SetPads, SetUnPads, SetKernelStrides, SetDilations, EnableRelu,
      EnableLeakyRelu, DisableRelu, DisableLeakyRelu, SetQuant.
  }];

  let arguments = (
    ins
    I64Attr:$op_type,         // 0: conv, 1: depthwise conv, 2: backward conv,
                              // 3: gemm
    MemRefOrInt:$src_activation,      // Input activation addr in SPM
    I32ArrayAttr:$src_dims,   // dims of src activation in NHWC format
    MemRefOrInt:$weight,              // Input weight addr in SPM
    I16Attr:$weight_dims,     // dims of weight(conv kernel) in Kx, Ky, Sx, Sy
                              // Where K and S is short for size(K) and step(S)
    BoolAttr:$en_bias,        // Enable bias add
    MemRefOrInt:$src_bias,           // The address of bias in SPM
    BoolAttr:$en_neg_scale,   // Enable negative axis scale
    MemRefOrInt:$src_neg_scale,      // The address of negative scale data in SPM
    BoolAttr:$en_pos_scale,   // Enable positive axis scale
    MemRefOrInt:$src_pos_scale,      // The address of positive scale data in SPM
    BoolAttr:$en_sparse,      // Enable sparse
    MemRefOrInt:$src_sparse,         // The sparse matrix addr in SPM
    BoolAttr:$en_psum,        // Enable psum? TODO: Production sum?
    MemRefOrInt:$src_psum,           // psum addr in SPM?
    I32ArrayAttr:$pads,      // Pad in top, bottom, left, right order
    I32ArrayAttr:$unpads,    // Unpad in top, bottom, left, right order
    I32ArrayAttr:$strides,   // Kernel strids in Kx, Ky, Sx, Sy
    I32ArrayAttr:$dilations, // dialation d0, d1 for conv/backwardconv
    BoolAttr:$en_leaky_relu,  // Enable LeakyRelu or normal Relu
    I32ArrayAttr:$out_dims,  // dims of output in NHWC format
    I64Attr:$src_fmt,         // Data format of src activation
    I64Attr:$weight_fmt,      // Data format of weight
    I64Attr:$out_fmt          // Data format of output
    // The param of SetQuant() is unused
  );

  // Output matrix C addr in SPM
  let results = (outs I64:$dst);
}

// =============================================================================
// 4.7. TsmGemm
// =============================================================================

def GemmOp : Tx81Op<"gemm", []> {
  let summary = "Gemm engine intrinsic runtime API";

  let description = [{
    This TsmGemm is not a 1 to 1 map to TsingMicro's TsmGemm intrinsic, it is
    the wrap of all APIs related to TsmGemm. This op wraps the following APIs:
      TsmNewGemm, TsmDeleteGemm, AddInput, ConfigMKN, AddOutput, SetPsum,
      SetTransflag, SetQuant, ConfigBatch, EnableRelu, EnableLeakyRelu,
      DisableRelu, DisableLeakyRelu, AddBias, SetNegativeAxisScale,
      SetPositiveAxisScale.
  }];

  let arguments = (
    ins
    MemRefOrInt:$src_a,          // Input matrix A addr in SPM
    MemRefOrInt:$src_b,          // Input matrix B addr in SPM
    MemRefOrInt:$src_bias,       // The address of bias in SPM
    // Zeroes buffer which can be used to fill $dst
    // FIXME: Whether need add side effect to source operands?
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$zeroes,
    I32ArrayAttr:$dims,   // The dimensions of M, K, N
    BoolAttr:$en_psum,        // Enable psum? TODO: Production sum?
    MemRefOrInt:$psum_addr,      // The address of psum in SPM, TODO: psum?
    BoolAttr:$trans_src_a,  // Should matrix A be transposed
    BoolAttr:$trans_src_b,  // Should matrix B be transposed
    I32Attr:$batch_src_a,  // The batch of matrix A
    I32Attr:$batch_src_b,  // The batch of matrix B
    BoolAttr:$en_leaky_relu,// Enable LeakyRelu or normal Relu
    BoolAttr:$en_bias,      // Enable bias add
    BoolAttr:$en_neg_scale, // Enable negative axis scale
    MemRefOrInt:$src_neg_scale,    // The address of negative scale data in SPM
    BoolAttr:$en_pos_scale, // Enable positive axis scale
    MemRefOrInt:$src_pos_scale,    // The address of positive scale data in SPM
    I32Attr:$src_fmt,       // Input matrix data format
    I32Attr:$dst_fmt        // Output matrix data format
    // The param of SetQuant() is unused
  );

  // Output matrix C addr in SPM
  let results = (outs Variadic<I64>:$dst);
}

// =============================================================================
// 4.10. TsmArith
// =============================================================================

def AbsVVOp : Tx81Op<"absvv", [Pure, Elementwise]> {}
def RecipVVOp : Tx81Op<"recipvv", [Pure, Elementwise]> {}
def SquareVVOp : Tx81Op<"squarevv", [Pure, Elementwise]> {}

class BinaryVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    MemRefOrInt:$input1,              // Second vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$rnd_mode,        // round mode
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AddVVOp : BinaryVVOp<"addvv"> {
  let summary = "Add two vectors element-wise";
}
def SubVVOp : BinaryVVOp<"subvv">;
def MulVVOp : BinaryVVOp<"mulvv">;
def DivVVOp : BinaryVVOp<"divvv">;

class BinaryVSOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    I32:$value,                       // Const value
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,  // Number of input elements
    I16Attr:$rnd_mode,        // round mode
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AddVSOp : BinaryVSOp<"addvs"> {
  let summary = "Add input vector and constant value";
}
def SubVSOp : BinaryVSOp<"subvs">;
def MulVSOp : BinaryVSOp<"mulvs">;
def DivVSOp : BinaryVSOp<"divvs">;

// ...

// =============================================================================
// 4.13. TsmTranscendental
// =============================================================================

class TranscendentalOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$src,              // Input vector address
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );
  let results = (outs I64:$dst);
}

def Log2 : TranscendentalOp<"log2", []> {
  let summary = "Logarithm based 2";
}
def Ln : TranscendentalOp<"ln", []> {
  let summary = "Logarithm based e";
}
def Pow2 : TranscendentalOp<"pow2", []> {
  let summary = "2 ** x";
}
def Exp : TranscendentalOp<"exp", []> {
  let summary = "Exponential with high precision";
}
def Explp : TranscendentalOp<"explp", []> {
  let summary = "Exponential with low precision";
}
def Sin : TranscendentalOp<"sin", []> {
  let summary = "Sine";
}
def Cos : TranscendentalOp<"cos", []> {
  let summary = "Cosine";
}

// =============================================================================
// 4.13. TsmActivation
// =============================================================================

class ActivationOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$src,              // Input vector address
    UI32Attr:$elem_count,   // Number of input elements
    UI16Attr:$fmt           // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Tanh : ActivationOp<"tanh", []> {
  let summary = "Hyperbolic tangent";
}
def Sigmoid : ActivationOp<"sigmoid", []> {
  let summary = "Logistic sigmoid";
}
def Relu : ActivationOp<"relu", []> {
  let summary = "Rectified linear unit";
}
def Satrelu : ActivationOp<"satrelu", []> {
  let summary = "Saturated ReLU";
}
def Leakyrelu : ActivationOp<"leakyrelu", []> {
  let summary = "Leaky rectified linear unit";
}
def Softplus : ActivationOp<"softplus", []> {
  let summary = "Smooth approximation of ReLU";
}

// =============================================================================
// 4.15. TsmReduce
// =============================================================================

class Reduce<string mnemonic> : Tx81Op<mnemonic, [Pure]> {
  let summary = "Reduction engine intrinsic runtime API";

  let description = [{
    Includes ReduceSum, ReduceAvg, ReduceMin and ReduceMax interfaces.
    Mapping between `dim` and NCHW:
      Reduction on C: dim=0
      Reduction on W: dim=1
      Reduction on H: dim=2
      Reduction on HW: dim=4
  }];

  let arguments = (
    ins
    AnyType:$src,           // Input tensor address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,           // Output tensor address in SPM
    UI32Attr:$dim,      // Which dimension to be reduced
    I64ArrayAttr:$shape,  // The shape info of src
    I16Attr:$fmt        // The data format of src & dst
  );

  // Output tensor address in SPM
  let results = (outs Variadic<AnyType>);
}

def ReduceSumOp : Reduce<"reduce_sum">;
def ReduceAvgOp : Reduce<"reduce_avg">;
def ReduceMaxOp : Reduce<"reduce_max">;
def ReduceMinOp : Reduce<"reduce_min">;

// =============================================================================
// 4.15. TsmMaskDataMove
// =============================================================================

def MaskMoveOp : Tx81Op<"mask_move", []> {
  let summary = "Mask data move engine intrinsic runtime API";

  let description = [{ When mask is 1, extract the data from src and write it to dst.
When mask=0, the corresponding elements of dst remain unchanged.
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,            // The source address in SPM
    // The target address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$target,
    AnySignlessIntegerOrIndex:$elem_count,    // Number of elements to be copied
    I32ArrayAttr:$mask,     // 3 dim masks
    I32Attr:$fmt
  );

  // The dst address is not used, use target in arguments instead.
  let results = (outs Variadic<I64>:$dst);
}

// =============================================================================
// 4.19. TsmConvert instructions
// =============================================================================

class ZeroPointConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$src,
    UI32Attr:$zp,
    UI32Attr:$elem_count
  );
  let results = (outs UI64:$dst);
}

def INT8ToFP16Op : ZeroPointConvertOp<"int8_fp16", []> {
  let summary = "Data format from int8 to fp16";
}
def INT8ToBF16Op : ZeroPointConvertOp<"int8_bf16", []> {
  let summary = "Data format from int8 to bf16";
}
def INT8ToFP32Op : ZeroPointConvertOp<"int8_fp32", []> {
  let summary = "Data format from int8 to fp32";
}
def INT8ToTF32Op : ZeroPointConvertOp<"int8_tf32", []> {
  let summary = "Data format from int8 to tf32";
}

class RoundConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$input,
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$output,
    AnySignlessIntegerOrIndex:$elem_count,
    I16Attr:$rnd_mode
  );
  let results = (outs I64:$dst);
}

def INT16ToBF16Op : RoundConvertOp<"int16_bf16", []> {
  let summary = "Data format from int16 to bf16";
}
def INT16ToFP32Op : RoundConvertOp<"int16_fp32", []> {
  let summary = "Data format from int16 to fp32";
}
def INT16ToTF32Op : RoundConvertOp<"int16_tf32", []> {
  let summary = "Data format from int16 to tf32";
}
def INT32ToFP16Op : RoundConvertOp<"int32_fp16", []> {
  let summary = "Data format from int32 to fp16";
}
def INT32ToBF16Op : RoundConvertOp<"int32_bf16", []> {
  let summary = "Data format from int32 to bf16";
}
def INT32ToFP32Op : RoundConvertOp<"int32_fp32", []> {
  let summary = "Data format from int32 to fp32";
}
def INT32ToTF32Op : RoundConvertOp<"int32_tf32", []> {
  let summary = "Data format from int32 to tf32";
}
def BF16ToINT16Op : RoundConvertOp<"bf16_int16", []> {
  let summary = "Data format from bf16 to int16";
}
def BF16ToINT32Op : RoundConvertOp<"bf16_int32", []> {
  let summary = "Data format from bf16 to int32";
}
def FP16ToINT8Op : RoundConvertOp<"fp16_int8", []> {
  let summary = "Data format from fp16 to int8";
}
def FP16ToINT16Op : RoundConvertOp<"fp16_int16", []> {
  let summary = "Data format from fp16 to int16";
}
def FP16ToINT32Op : RoundConvertOp<"fp16_int32", []> {
  let summary = "Data format from fp16 to int32";
}
def FP16ToBF16Op : RoundConvertOp<"fp16_bf16", []> {
  let summary = "Data format from fp16 to bf16";
}
def FP32ToINT8Op : RoundConvertOp<"fp32_int8", []> {
  let summary = "Data format from fp32 to int8";
}
def FP32ToINT16Op : RoundConvertOp<"fp32_int16", []> {
  let summary = "Data format from fp32 to int16";
}
def FP32ToINT32Op : RoundConvertOp<"fp32_int32", []> {
  let summary = "Data format from fp32 to int32";
}
def FP32ToFP16Op : RoundConvertOp<"fp32_fp16", []> {
  let summary = "Data format from fp32 to fp16";
}
def FP32ToBF16Op : RoundConvertOp<"fp32_bf16", []> {
  let summary = "Data format from fp32 to bf16";
}
def FP32ToTF32Op : RoundConvertOp<"fp32_tf32", []> {
  let summary = "Data format from fp32 to tf32";
}
def TF32ToINT8Op : RoundConvertOp<"tf32_int8", []> {
  let summary = "Data format from tf32 to int8";
}
def TF32ToINT16Op : RoundConvertOp<"tf32_int16", []> {
  let summary = "Data format from tf32 to int16";
}
def TF32ToINT32Op : RoundConvertOp<"tf32_int32", []> {
  let summary = "Data format from tf32 to int32";
}
def TF32ToFP32Op : RoundConvertOp<"tf32_fp32", []> {
  let summary = "Data format from tf32 to fp32";
}

class NormalConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$input,
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$output,
    AnySignlessIntegerOrIndex:$elem_count
  );
  let results = (outs I64:$dst);
}

def INT16ToFP16Op : NormalConvertOp<"int16_fp16", []> {
  let summary = "Data format from int16 to fp16";
}
def BF16ToINT8Op : NormalConvertOp<"bf16_int8", []> {
  let summary = "Data format from bf16 to int8";
}
def BF16ToFP16Op : NormalConvertOp<"bf16_fp16", []> {
  let summary = "Data format from bf16 to fp16";
}
def BF16ToFP32Op : NormalConvertOp<"bf16_fp32", []> {
  let summary = "Data format from bf16 to fp32";
}
def BF16ToTF32Op : NormalConvertOp<"bf16_tf32", []> {
  let summary = "Data format from bf16 to tf32";
}
def FP16ToFP32Op : NormalConvertOp<"fp16_fp32", []> {
  let summary = "Data format from fp16 to fp32";
}
def FP16ToTF32Op : NormalConvertOp<"fp16_tf32", []> {
  let summary = "Data format from fp16 to tf32";
}
def TF32ToFP16Op : NormalConvertOp<"tf32_fp16", []> {
  let summary = "Data format from tf32 to fp16";
}
def TF32ToBF16Op : NormalConvertOp<"tf32_bf16", []> {
  let summary = "Data format from tf32 to bf16";
}

// =============================================================================
// 4.20. TsmPeripheral instructions
// =============================================================================

def CountOp : Tx81Op<"count", [Pure]> {
  let summary = "Count the non-zero elements from given tensor";

  let arguments = (
    ins
    MemRefOrInt:$src,               // Input tensor address in SPM
    I32Attr:$elem_count,     // TODO: Ask TsingMicro for explain.
    //I64Attr:$p_wb_data0,   // TODO: Ask TsingMicro for explain.
    //I64Attr:$p_wb_data1,   // TODO: Ask TsingMicro for explain.
    I16Attr:$fmt
  );

  // The output tensor address in SPM
  let results = (outs MemRefOrInt:$dst);
}

def MemsetOp : Tx81Op<"memset", []> {
  let summary = "Write given `value` to range of address on SPM(sram)";

  let arguments = (
    ins
    MemRefOrInt:$src,           // SPM address to be memset
    I32:$value,         // Value to be written
    AnySignlessIntegerOrIndex:$elem_count,
    I32ArrayAttr:$strides,
    I32ArrayAttr:$iterations,
    I16Attr:$fmt
  );

  // The address updated by memset in SPM
  let results = (outs MemRefOrInt:$dst);
}

def Bit2FpOp : Tx81Op<"bit2fp", []> {
  let summary = "Convert a vector of the bitwise into the fp vector";

  let arguments = (ins
    UI64:$src,             // Input tensor
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def ArgMaxOp : Tx81Op<"argmax", []> {
  let summary = "Return a max value inner a vector and its corresponding index";

  let arguments = (ins
    UI64:$src,             // Input vector
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );

  let results = (outs
    AnyType:$max,          // Max value inner a vector
    UI64:$index            // Corresponding index
  );
}

def ArgMinOp : Tx81Op<"argmin", []> {
  let summary = "Return a min value inner a vector and its corresponding index";

  let arguments = (ins
    UI64:$src,             // Input vector
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );

  let results = (outs
    AnyType:$min,          // Min value inner a vector
    UI64:$index            // Corresponding index
  );
}

def BilinearOp : Tx81Op<"bilinear", []> {
  let summary = "Bilinear interpolation";

  let arguments = (ins
    UI64:$src,                 // Input tensor with the NHWC format
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    F32:$scale_w,              // Input tensor "w" divided by output tensor "w"
    F32:$scale_h,              // Input tensor "h" divided by output tensor "h"
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Lut16Op : Tx81Op<"lut16", []> {
  let summary = "16-bit lookup table";

  let arguments = (ins
    // FIXME: AnyVector is not defined
    // AnyVector:$src,            // Vector offset with respect to LUT
    UI64:$lut16,
    I32Attr:$src_elem_count,   // Number of elements in vector offset
    I32Attr:$lut_elem_count    // Number of elements in LUT
  );

  let results = (outs UI64:$dst);
}

def Lut32Op : Tx81Op<"lut32", []> {
  let summary = "32-bit lookup table";

  let arguments = (ins
    // FIXME: AnyVector is not defined
    // AnyVector:$src,            // Vector offset with respect to LUT
    UI64:$lut32,
    I32Attr:$src_elem_count,   // Number of elements in vector offset
    I32Attr:$lut_elem_count    // Number of elements in LUT
  );

  let results = (outs UI64:$dst);
}

def RandGenOp : Tx81Op<"randgen", []> {
  let summary = "Generate random numbers using two 64-bit seeds";

  let arguments = (ins
    UI64:$src0,          // The first random seed
    UI64:$src1,          // The second random seed
    UI64:$dst0,          // Store the first random seed
    UI64:$dst1,          // Store the second random seed
    UI64:$dst2,          // Random value
    I32Attr:$elem_num,   // Number of random values
    I16Attr:$fmt         // The date format of random value
  );
}


//
// 4.21. TsmDataMove
//

class TransformOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    UI64:$src,                 // Input matrix or tensor address
    I32ArrayAttr:$src_shape,   // Input shape
    I32ArrayAttr:$dst_shape,   // Output shape
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Mirror : TransformOp<"mirror", []> {
  let summary = "Horizontal mirror to a matrix";
}
def Transpose : TransformOp<"transpose", []> {
  let summary = "Transpose a matrix";
}
def Rotate90 : TransformOp<"rotate90", []> {
  let summary = "Rotate a matrix 90 degree clockwise";
}
def Rotate180 : TransformOp<"rotate180", []> {
  let summary = "Rotate a matrix 180 degree clockwise";
}
def Rotate270 : TransformOp<"rotate270", []> {
  let summary = "Rotate a matrix 270 degree clockwise";
}
def Nchw2nhwc : TransformOp<"nchw2nhwc", []> {
  let summary = "Tranform a tensor from nchw to nhwc";
}
def Nhwc2nchw : TransformOp<"nhwc2nchw", []> {
  let summary = "Tranform a tensor from nhwc to nchw";
}
def TensorNorm : TransformOp<"tensornorm", []> {
  let summary = "Make continuous tensor align std format in ch direction";
}

def Concat : Tx81Op<"concat", []> {
  let summary = "Concatenation based on the dim";

  let arguments = (ins
    UI64:$src1,                 // The first input tensor
    I32ArrayAttr:$src1_shape,   // The first input tensor shape
    UI64:$src2,                 // The second input tensor
    I32ArrayAttr:$src2_shape,   // The second input tensor shape
    I32ArrayAttr:$dst_shape,    // Ouput tensor shape
    I16Attr:$dim,               // Represent the concat direction, such as:
                                //   0 is channel, 1 is width, and 2 is height
    I16Attr:$fmt                // The data format of input & output tensor
  );
  let results = (outs UI64:$dst);
}

def Pad : Tx81Op<"pad", []> {
  let summary = "Tensor padding";

  let arguments = (ins
    UI64:$src,                 // Input tensor
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    I16Attr:$pad,              // Padding mode: top, bottom, left, and right
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Img2col : Tx81Op<"img2col", []> {
  let summary = "Transform a feature-map tensor into a matrix";

  let arguments = (ins
    UI64:$src,                 // Input tensor
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    I32Attr:$src_elem_num,     // Number of elements in input tensor
    I32Attr:$dst_elem_num,     // Number of elements in output tensor
    I32ArrayAttr:$swr,         // Horizontal stride of convolution
    I32ArrayAttr:$pdr,         // Vertical stride of convolution
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def GatherScatter : Tx81Op<"gatherscatter", []> {
  let summary = "Transfer data in strides and iterations";

  let arguments = (ins
    UI64:$src,                      // Input tensor
    I32Attr:$size,                  // Transfer data size in bytes
    I32ArrayAttr:$src_strides,      // 3 dim strides for input
    I32ArrayAttr:$src_iterations,   // 3 dim iterations for input
    I32ArrayAttr:$dst_strides,      // 3 dim strides for output
    I32ArrayAttr:$dst_iterations    // 3 dim iterations for output
  );
  let results = (outs UI64:$dst);
}

#endif // TSINGMICRO_TX81_OPS
