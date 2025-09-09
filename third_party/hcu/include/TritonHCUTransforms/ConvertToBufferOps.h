//incluede file
#ifndef TRITON_DIALECT_TRITONHCUGPU_TRANSFORMS_COVERTTOBUFFEROPS_H_
#define TRITON_DIALECT_TRITONHCUGPU_TRANSFORMS_COVERTTOBUFFEROPS_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {

bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions);

}

#endif