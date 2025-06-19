#ifndef UNIFIED_HARDWARE_AIPU_H
#define UNIFIED_HARDWARE_AIPU_H

#include <optional> 

#include "flagtree/Common/UnifiedHardwareBase.h"

namespace mlir {
namespace aipu {

class UnifiedHardwareAIPU final : public mlir::flagtree::UnifiedHardware {
    
    //DMA
    std::optional<int> getAllocSpaceForDMATag() const override{
        return std::optional<int>(11);
    }
};

} // namespace aipu
} // namespace mlir

#endif // UNIFIED_HARDWARE_AIPU_H