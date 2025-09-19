#include "flagtree/Common/UnifiedHardware.h"
#include <memory>
namespace mlir {
namespace flagtree {

bool UnifiedHardware::isRegistered() {
#ifdef FLAGTREE_BACKEND
  return true;
#else
  return false;
#endif
}

int UnifiedHardware::getDMATag() { return 0; }

int UnifiedHardware::getSharedMemoryTag() { return 0; }

std::string UnifiedHardware::getFlagTreeBackend() { return "default"; }

__attribute__((weak)) std::unique_ptr<UnifiedHardware>
createUnifiedHardwareManager() {
  return std::make_unique<UnifiedHardware>();
}

} // namespace flagtree
} // namespace mlir
