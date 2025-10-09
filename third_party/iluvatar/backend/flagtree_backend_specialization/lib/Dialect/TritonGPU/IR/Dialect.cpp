#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

bool isMma(Attribute layout) {
  if (auto mmaLayout = layout.dyn_cast<IluvatarMmaEncodingAttr>())
    return mmaLayout.isVolta();
  return false;
}

bool isMmaOrSliceMma(Attribute layout) {
  if (auto mmaLayout = layout.dyn_cast<IluvatarMmaEncodingAttr>())
    return mmaLayout.isVolta();
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
    return isMmaOrSliceMma(sliceLayout.getParent());
  return false;
}

bool isSliceMmaWithDim(Attribute layout, int targetDim) {
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    if (auto mmaLayout = parentLayout.dyn_cast<IluvatarMmaEncodingAttr>()) {
      return mmaLayout.isVolta() && (sliceLayout.getDim() == targetDim);
    }
  }
  return false;
}

bool isMmaConvertLayout(Operation *op) {
  // Match cvt(#mma(version_minor = 0) -> #mma(version_minor > 0))
  // The later is for storing dot result.
  if (auto convertOp = dyn_cast<ConvertLayoutOp>(op)) {
    auto srcType = convertOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convertOp.getResult().getType().cast<RankedTensorType>();
    if (!srcType || !dstType)
      return false;
    auto srcMmaEnc = dyn_cast<IluvatarMmaEncodingAttr>(srcType.getEncoding());
    auto dstMmaEnc = dyn_cast<IluvatarMmaEncodingAttr>(dstType.getEncoding());
    if (!srcMmaEnc || !dstMmaEnc)
      return false;
    return srcMmaEnc.getVersionMinor() == 0 && dstMmaEnc.getVersionMinor() > 0;
  }
  return false;
}

bool isSliceMmaConvertLayout(Operation *op, bool srcNoWarpReduce,
                             bool dstNoWarpReduce) {
  // Match cvt(slice<{parent=#mma, noWarpReduce=srcNoWarpReduce}>
  // -> slice<{parent=#mma, noWarpReduce=dstNoWarpReduce}>)
  if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    auto srcType = convertOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convertOp.getResult().getType().cast<RankedTensorType>();
    if (!srcType || !dstType)
      return false;
    auto srcLayout =
        dyn_cast<triton::gpu::SliceEncodingAttr>(srcType.getEncoding());
    auto dstLayout =
        dyn_cast<triton::gpu::SliceEncodingAttr>(dstType.getEncoding());
    if (!srcLayout || !dstLayout)
      return false;
    auto srcMmaLayout =
        srcLayout.getParent().dyn_cast<triton::gpu::IluvatarMmaEncodingAttr>();
    auto dstMmaLayout =
        dstLayout.getParent().dyn_cast<triton::gpu::IluvatarMmaEncodingAttr>();
    if (!srcMmaLayout || !dstMmaLayout)
      return false;
    return srcLayout.getNoWarpReduce() == srcNoWarpReduce &&
           dstLayout.getNoWarpReduce() == dstNoWarpReduce;
  }
  return false;
}

} // namespace gpu
} // namespace triton
} // namespace mlir
