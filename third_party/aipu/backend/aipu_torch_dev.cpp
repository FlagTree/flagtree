#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/InferSize.h>

#include <standard_api.h>
#include <internal/internal_api.h>

#include <mutex>
#include <unordered_map>

static c10::DeviceIndex aipu_device_index = 0;

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
}}


#define AIPU_DRIVER_HANDLE_ERROR(status)                            \
  do {                                                              \
    if (status != AIPU_STATUS_SUCCESS) {                            \
      const char* error_message = nullptr;                          \
      aipu_get_error_message(aipu_ctx_, status, &error_message);         \
      std::cout << error_message; \
    }                                                               \
  } while (false)


/*! \brief Return whether a string starts with the given prefix. */
inline bool StrStartsWith(const std::string& str, const std::string& prefix) {
  if (prefix.size() > str.size()) return false;
  return std::equal(str.c_str(), str.c_str() + prefix.size(), prefix.c_str());
}


class Context final {
  public:
    aipu_ctx_handle_t* process_ctx = nullptr;
    std::mutex inst_lock;
    Context() {
      if (process_ctx == nullptr) {
        std::lock_guard<std::mutex> lock(inst_lock);
        if (process_ctx == nullptr) {
          aipu_status_t status = aipu_init_context(&process_ctx);
          if (status != AIPU_STATUS_SUCCESS) {
            //
          }
        }
      }
    };
    ~Context() {
      if (process_ctx != nullptr) {
        std::lock_guard<std::mutex> lock(inst_lock);
        if (process_ctx != nullptr) {
          aipu_status_t status = aipu_deinit_context(process_ctx);
          if (status != AIPU_STATUS_SUCCESS) {
            //
          }
          process_ctx = nullptr;
        }
      }
    };
};

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      return new Context();
    } catch (...) {
    }
    return nullptr;
  }());

  return context.get();
}


struct AIPUAllocator final : at::Allocator {
  AIPUAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    status_ = aipu_malloc(aipu_ctx_, nbytes, 32, 0, &data);
    AIPU_DRIVER_HANDLE_ERROR(status_);

    std::cerr << "alloc with aipu allocator for " << nbytes << " bytes with ptr " << (uint64_t)data << std::endl;
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, aipu_device_index)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    status_ = aipu_free(aipu_ctx_, &ptr);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  static aipu_ctx_handle_t* aipu_ctx_;
  static aipu_status_t status_;
};

// Register our dummy allocator
aipu_ctx_handle_t* AIPUAllocator::aipu_ctx_ = context()->process_ctx;
aipu_status_t AIPUAllocator::status_ = AIPU_STATUS_SUCCESS;
static AIPUAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);


at::Tensor custom_empty_symint(c10::IntArrayRef size,
                               std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout,
                               std::optional<at::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size,
    &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor custom_empty_strided(c10::IntArrayRef size,
                                c10::IntArrayRef stride,
                                std::optional<at::ScalarType> dtype_opt,
                                std::optional<at::Layout> layout_opt,
                                std::optional<at::Device> device_opt,
                                std::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

at::Tensor aipu_view(const at::Tensor& self, c10::IntArrayRef size) {
  at::IntArrayRef self_sizes = self.sizes();
  at::IntArrayRef self_strides = self.strides();
  at::DimVector inferred_size = at::infer_size_dv(self_sizes, self.numel());
  std::optional<at::DimVector> stride =
      at::detail::computeStride(self_sizes, self_strides, inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");

  at::Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
  self_.unsafeGetTensorImpl()->set_sizes_and_strides(inferred_size, *stride);
  self_.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
  return self_;
}

at::Tensor aipu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking=false) {
  auto kind = AIPU_MEMCPY_HOST_TO_DEVICE;
  if (StrStartsWith(self.device().str(), "aipu")) {
    kind = AIPU_MEMCPY_DEVICE_TO_HOST;
  }
  aipu_memcpy(
    AIPUAllocator::aipu_ctx_,
    dst.data_ptr(),
    self.data_ptr(),
    self.nbytes(),
    kind);
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_symint);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("aten::view", &aipu_view);
  m.impl("aten::uniform_", at::native::uniform_);
  m.impl("aten::_copy_from", &aipu_copy_from);
}

namespace aipu {

bool is_available() {
  return context() != nullptr;
}


int device_count() {
  return is_available()? 1  : 0;
}


int current_device() {
  return 0;
}
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("device_count", &aipu::device_count, "aipu device count");
    m.def("is_available", &aipu::is_available, "aipu is available");
    m.def("current_device", &aipu::current_device, "aipu current device");
}
