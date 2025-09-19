#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace flagtree {
// this is the unified hardware abstraction for hardware
// to determined if these abstraction is specified, using std::optional is
// needed using in passes: if(uh_flagtree->xxx()){...}

class UnifiedHardware {

public:
  UnifiedHardware() = default;
  virtual ~UnifiedHardware() = default;
  virtual bool isRegistered();
  virtual int getDMATag();
  virtual int getSharedMemoryTag();
  virtual std::string getFlagTreeBackend();
};

std::unique_ptr<UnifiedHardware> createUnifiedHardwareManager();

} // namespace flagtree
} // namespace mlir
