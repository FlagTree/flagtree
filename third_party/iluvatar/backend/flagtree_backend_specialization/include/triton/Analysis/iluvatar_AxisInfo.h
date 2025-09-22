#ifndef ILUVATAR_TRITON_ANALYSIS_AXISINFO_H
#define ILUVATAR_TRITON_ANALYSIS_AXISINFO_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton {

#define FLAGTREE_SPEC_AxisInfo_initPessimisticStateFromFunc iluvatar_initPessimisticStateFromFunc
  template <class T> void
  AxisInfo::iluvatar_initPessimisticStateFromFunc(int argNumber, T funcOp, AxisInfo::DimVectorT *contiguity,
                                                  AxisInfo::DimVectorT *divisibility, AxisInfo::DimVectorT *constancy,
                                                  AxisInfo::DimVectorT *corex_stride);

} // namespace mlir::triton

#endif