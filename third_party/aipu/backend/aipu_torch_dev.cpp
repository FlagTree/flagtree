#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>

#include <internal/internal_api.h>
#include <standard_api.h>

#include <mutex>
#include <unordered_map>

static c10::DeviceIndex aipu_device_index = 0;

namespace c10 {
namespace impl {

struct C10_API AIPUGuardImpl final : public DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1;
  inline static int8_t current_device = 0;
  inline static int64_t current_stream = 0;

  DeviceType type() const override { return static_type; }

  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Device must be PrivateUse1 type");
    current_device = d.index();
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    current_device = d.index();
  }

  Device getDevice() const override {
    return Device(DeviceType::PrivateUse1, current_device);
  }

  Device exchangeDevice(Device d) const override {
    Device old_device = getDevice();
    setDevice(d);
    return old_device;
  }

  Stream getStream(Device d) const noexcept override {
    int64_t stream_id = d.index();
    return Stream(Stream::UNSAFE, d, stream_id);
  }

  Stream exchangeStream(Stream s) const noexcept override {
    auto old_stream = getStream(s.device());
    current_stream = s.id();
    return old_stream;
  }

  DeviceIndex deviceCount() const noexcept override { return 1; }
};

} // namespace impl
} // namespace c10

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::AIPUGuardImpl);
}
} // namespace at

#define AIPU_DRIVER_HANDLE_ERROR(status)                                       \
  do {                                                                         \
    if (status != AIPU_STATUS_SUCCESS) {                                       \
      const char *error_message = nullptr;                                     \
      aipu_get_error_message(aipu_ctx_, status, &error_message);               \
      std::cout << error_message;                                              \
    }                                                                          \
  } while (false)

/*! \brief Return whether a string starts with the given prefix. */
inline bool StrStartsWith(const std::string &str, const std::string &prefix) {
  if (prefix.size() > str.size())
    return false;
  return std::equal(str.c_str(), str.c_str() + prefix.size(), prefix.c_str());
}

class Context final {
public:
  aipu_ctx_handle_t *process_ctx = nullptr;
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

Context *context() {
  static const std::unique_ptr<Context> context([]() -> Context * {
    try {
      return new Context();
    } catch (...) {
    }
    return nullptr;
  }());

  return context.get();
}

using namespace at;

struct AIPUAllocator final : Allocator {
  AIPUAllocator() = default;

  DataPtr allocate(size_t nbytes) override {
    void *data = nullptr;
    status_ = aipu_malloc(aipu_ctx_, nbytes, 32, 0, &data);
    AIPU_DRIVER_HANDLE_ERROR(status_);

    return {data, data, &ReportAndDelete,
            Device(DeviceType::PrivateUse1, aipu_device_index)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    status_ = aipu_free(aipu_ctx_, &ptr);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }

  DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }

  void copy_data(void *dest, const void *src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  static aipu_ctx_handle_t *aipu_ctx_;
  static aipu_status_t status_;
};

// Register our dummy allocator
aipu_ctx_handle_t *AIPUAllocator::aipu_ctx_ = context()->process_ctx;
aipu_status_t AIPUAllocator::status_ = AIPU_STATUS_SUCCESS;
static AIPUAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

Tensor custom_empty_symint(c10::IntArrayRef size,
                           std::optional<ScalarType> dtype,
                           std::optional<Layout> layout,
                           std::optional<Device> device,
                           std::optional<bool> pin_memory,
                           std::optional<MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks,
                                   c10::dtype_or_default(dtype), memory_format);
}

Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                            std::optional<ScalarType> dtype_opt,
                            std::optional<Layout> layout_opt,
                            std::optional<Device> device_opt,
                            std::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return at::detail::empty_strided_generic(size, stride, &global_custom_alloc,
                                           private_use_ks, dtype);
}

Tensor aipu_view(const Tensor &self, c10::IntArrayRef size) {
  IntArrayRef self_sizes = self.sizes();
  IntArrayRef self_strides = self.strides();
  DimVector inferred_size = infer_size_dv(self_sizes, self.numel());
  std::optional<DimVector> stride =
      at::detail::computeStride(self_sizes, self_strides, inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");

  Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  self_.unsafeGetTensorImpl()->set_sizes_and_strides(inferred_size, *stride);
  self_.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
  return self_;
}

Tensor aipu_copy_from(const Tensor &self, const Tensor &dst,
                      bool non_blocking = false) {
  auto kind = AIPU_MEMCPY_HOST_TO_DEVICE;
  if (StrStartsWith(self.device().str(), "aipu")) {
    kind = AIPU_MEMCPY_DEVICE_TO_HOST;
    if (StrStartsWith(dst.device().str(), "aipu")) {
      kind = AIPU_MEMCPY_DEVICE_TO_DEVICE;
    }
  }

  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  auto status = aipu_memcpy(aipu_ctx_, dst.data_ptr(), self.data_ptr(),
                            self.nbytes(), kind);
  AIPU_DRIVER_HANDLE_ERROR(status);
  return self;
}

template <template <typename> class RND>
Tensor &random_kernel(Tensor &self, double cond1, double cond2,
                      c10::optional<Generator> gen) {
  CPUGeneratorImpl *generator = get_generator_or_default<CPUGeneratorImpl>(
      gen, at::detail::getDefaultCPUGenerator());
  int64_t numel = self.numel();

  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  char *data_ptr = nullptr;
  auto status = aipu_get_va(aipu_ctx_, self.data_ptr(), &data_ptr);
  AIPU_DRIVER_HANDLE_ERROR(status);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "random_kernel_aipu", [&]() {
        RND<double> distribution(cond1, cond2);

        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        for (int i = 0; i < numel; ++i) {
          data[i] = static_cast<scalar_t>(distribution(generator));
        }
      });
  return self;
}

template <template <typename> class RND>
Tensor &random_from_to_kernel(Tensor &self, int64_t from,
                              c10::optional<int64_t> to_opt,
                              c10::optional<Generator> gen) {
  CPUGeneratorImpl *generator = get_generator_or_default<CPUGeneratorImpl>(
      gen, at::detail::getDefaultCPUGenerator());
  int64_t numel = self.numel();
  uint64_t range = static_cast<int64_t>(*to_opt) - static_cast<int64_t>(from);

  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  char *data_ptr = nullptr;
  auto status = aipu_get_va(aipu_ctx_, self.data_ptr(), &data_ptr);
  AIPU_DRIVER_HANDLE_ERROR(status);

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "random_from_to_kernel_aipu", [&]() {
        RND<scalar_t> distribution(range, from);

        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        for (int i = 0; i < numel; ++i) {
          data[i] = static_cast<scalar_t>(distribution(generator));
        }
      });
  return self;
}

Scalar _local_scalar_dense_aipu(const Tensor &self) {
  Scalar r;
  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  char *data_ptr = nullptr;
  auto status = aipu_get_va(aipu_ctx_, self.data_ptr(), &data_ptr);
  AIPU_DRIVER_HANDLE_ERROR(status);

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "_local_scalar_dense_aipu", [&]() {
        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        scalar_t value = static_cast<scalar_t>(*data);
        r = Scalar(value);
      });
  return r;
}

Tensor &fill_scalar_aipu(Tensor &self, const Scalar &value) {
  int64_t numel = self.numel();
  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  char *data_ptr = nullptr;
  auto status = aipu_get_va(aipu_ctx_, self.data_ptr(), &data_ptr);
  AIPU_DRIVER_HANDLE_ERROR(status);

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "fill_scalar_aipu", [&]() {
        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        std::fill(data, data + numel, value.to<scalar_t>());
      });
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_symint);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("as_strided", native::as_strided_tensorimpl);
  m.impl("aten::view", &aipu_view);
  m.impl("aten::uniform_", &random_kernel<uniform_real_distribution>);
  m.impl("aten::normal_", &random_kernel<normal_distribution>);
  m.impl("aten::_copy_from", &aipu_copy_from);
  m.impl("aten::random_.from",
         &random_from_to_kernel<uniform_int_from_to_distribution>);
  m.impl("aten::_local_scalar_dense", &_local_scalar_dense_aipu);
  m.impl("aten::fill_.Scalar", &fill_scalar_aipu);
}

namespace aipu {

bool is_available() { return context() != nullptr; }

int device_count() { return is_available() ? 1 : 0; }

int current_device() { return 0; }
} // namespace aipu

struct _DeviceGuard {
  _DeviceGuard(int index, int prev_index) : idx(index), prev_idx(prev_index) {}

  int idx = 0;
  int prev_idx = -1;
};

struct _Device {
  _Device(c10::Device device) { idx = device.index(); }

  int idx = 0;
  int prev_idx = -1;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_count", &aipu::device_count, "aipu device count");
  m.def("is_available", &aipu::is_available, "aipu is available");
  m.def("current_device", &aipu::current_device, "aipu current device");
  m.def("_is_in_bad_fork", []() { return py::bool_(false); });
  m.def("manual_seed_all", [](int seed) { std::srand(seed); });

  py::class_<_DeviceGuard>(m, "_DeviceGuard", py::module_local())
      .def(py::init(
          [](int index) { return std::make_unique<_DeviceGuard>(index, -1); }))
      .def("__enter__", [](_DeviceGuard &self) { ; })
      .def("__exit__",
           [](_DeviceGuard &self, pybind11::object type, pybind11::object value,
              pybind11::object traceback) { return py::bool_(false); });

  py::class_<_Device>(m, "device", py::module_local())
      .def(py::init(
          [](c10::Device device) { return std::make_unique<_Device>(device); }))
      .def("__enter__", [](_Device &self) { ; })
      .def("__exit__",
           [](_Device &self, pybind11::object type, pybind11::object value,
              pybind11::object traceback) { return py::bool_(false); });
}
