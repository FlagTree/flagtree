#include "flagtree/Common/UnifiedHardware.h"
#include <memory>
namespace mlir {
namespace flagtree {

std::unique_ptr<UnifiedHardware> createUnifiedHardwareManager() {
  return std::make_unique<UnifiedHardware>();
}

} // namespace flagtree
} // namespace mlir
