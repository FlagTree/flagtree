#ifndef UNIFIED_HARDWARE_BASE_H
#define UNIFIED_HARDWARE_BASE_H

#include <optional> 

namespace mlir {

//this is the unified hardware abstraction for hardware
//to determined if these abstraction is specified, using std::optional is needed
//using in passes: if(uh_flagtree->xxx()){...}

class UnifiedHardware{

public:
    virtual ~UnifiedHardware() = default;

    //DMA
    virtual std::optional<int> getAllocSpaceForDMATag() const {
        return std::nullopt;
    }

};


} // namespace mlir

#endif // UNIFIED_HARDWARE_BASE_H
