include "TritonGPUAttrDefs.td"

def DotOperandEncodingAttr : BaseDotOperandEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth,
    "unsigned":$useSme
  );

  let builders = [
    // Specially for MMAV1(Volta)
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
      if (!parentAttr || !parentAttr.isAmpere())
        return $_get(context, opIdx, parent, 0, 0);
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned MMAv2kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, MMAv2kWidth, 0);
    }]>,

    // Specially for MR/BI150
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy,
                     "unsigned":$useSme), [{
      IluvatarMmaEncodingAttr parentAttr = mlir::dyn_cast<IluvatarMmaEncodingAttr>(parent);
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, kWidth, useSme);
    }]>
  ];
}
