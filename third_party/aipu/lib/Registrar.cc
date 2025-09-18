#include "flagtree/Common/UnifiedHardware.h"

class AipuUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
  int getDMATag() override;
};

int AipuUnifiedHardware::getDMATag() { return 11; }

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<AipuUnifiedHardware>();
}
