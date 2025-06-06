#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Triton_Structured_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::tts";

  let summary = "Structured Triton operations";

  let description = [{
    Triton Structured Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect"
  ];

  let usePropertiesForAttributes = 1;
}

//
// Op Base
//
class TTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Structured_Dialect, mnemonic, traits> {
}

def TTS_MakeTensorPtrOp
  : TTS_Op<"make_tptr", [AttrSizedOperandSegments, Pure]> {
  let summary = "create a pointer that points to a tensor in memory";

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // order:   The order of the block, which means how the block is laid out in memory.
  //          It contains the same info as order in tt.make_tensor_ptr.
  // shape:   If order is present, this field signifies the shape of the parent tensor in
  //          memory; if order is not present, it signifies the boundary by which addresses
  //          wraps around (constant zero indicates no wrap-around in the corresponding dimension).
  // offsets: Offset of the block along each dimension from base.
  // result:  If order is present, this op produces a pointer to a tensor; otherwise,
  //          it produces a tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_shape,
                       DenseI32ArrayAttr:$order);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    `` `,` `order` `` `:` $order
    attr-dict `:` type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      "ArrayRef<int32_t>":$order)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    bool isBlockPtr() {
      return !getOrder().empty();
    }
    bool isStructuredPtr() {
      return !isBlockPtr() &&
             llvm::all_of(getStaticShape(), [](auto shape) { return shape == 0; });
    }
    bool isSplitPtr() {
      return !isBlockPtr() &&
             !isStructuredPtr();
    }
  }];

  // TODO
  //let hasVerifier = 1;
  //let hasCanonicalizer = 1;
}

// SameVariadicResultSize
// AttrSizedResultSegments
def TTS_GetStructuredStateOp : TTS_Op<"get_structured_state", [AttrSizedResultSegments, Pure]> {
  let summary = "Placeholder for the structured pointer states computed during PtrAnalysis.";
  let description = "Used to pass the offsets and strides to scf.for op to simplify IR rewrites.";

  let arguments = (ins AnyTypeOf<[TT_PtrLike, I32Tensor]>:$input);
  let results = (outs AnyTypeOf<[TT_PtrLike, I32Tensor]>:$structured, Variadic<Index>:$offsets, Variadic<Index>:$strides);

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];

  let extraClassDeclaration = [{
    static std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
      getOffsetAndStrideTypes(MLIRContext *context, Type ptrLikeType);

    static std::optional<std::pair<int32_t, int32_t>>
      getOffsetAndStrideSegmentSizes(Type ptrLikeType);
  }];

  let hasFolder = 0;
  let hasVerifier = 1;
}

def TTS_LoadOp : TTS_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "ArrayRef<OpFoldResult>":$mask_dims, "Value":$other)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getStaticMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

def TTS_StoreOp : TTS_Op<"store", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<OpFoldResult>":$dims)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getStaticMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

#endif // TRITON_STRUCTURED_DIALECT
