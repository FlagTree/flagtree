#include <c10/core/Allocator.h>
#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}
} // namespace at

struct TXDADeviceAllocator final : at::Allocator {
  TXDADeviceAllocator() {}

  at::DataPtr allocate(size_t nbytes) override {
    void *data = c10::alloc_cpu(nbytes);
    return {data, nullptr, &ReportAndDelete,
            at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    // TsmDeviceFree((uint64_t)ptr)
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }
  void copy_data(void *dest, const void *src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

// register device allocator
static TXDADeviceAllocator global_txda_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_txda_alloc);

// to.Device
at::Tensor txda_to_device(const at::Tensor &self, at::Device device,
                          at::ScalarType dtype, bool non_blocking, bool copy,
                          c10::optional<at::MemoryFormat> memory_format) {
  // TsmMemcpyH2D();

  TORCH_CHECK(self.is_cpu() ||
                  self.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  // Some dummy asserts for the basic use case: inputs are the same size /
  // dtype, all contiguous.
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  if (device != at::DeviceType::CPU) {
    return at::empty(self.sizes(), self.options());
  }

  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device,
                       false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  return out;
}

// _copy_from
at::Tensor txda__copy_from(const at::Tensor &self, const at::Tensor &dst,
                           bool non_blocking) {
  // TsmMemcpyD2H();

  TORCH_CHECK(self.is_cpu() ||
                  self.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  TORCH_CHECK(dst.is_cpu() ||
                  dst.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");

  // Some dummy asserts for the basic use case: inputs are the same size /
  // dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(),
              self.storage().nbytes());
  return dst;
}

at::Tensor txda_empty_memory_format(
    at::IntArrayRef size, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_txda_alloc, private_use_ks,
                                   c10::dtype_or_default(dtype), memory_format);
}

at::Tensor txda_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype_opt,
                              std::optional<at::Layout> layout_opt,
                              std::optional<at::Device> device_opt,
                              std::optional<bool> pin_memory_opt) {

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return at::detail::empty_strided_generic(size, stride, &global_txda_alloc,
                                           private_use_ks, dtype);
}

at::Tensor txda_as_strided(const at::Tensor &input, at::IntArrayRef size,
                           at::IntArrayRef stride,
                           c10::optional<int64_t> storage_offset) {
  return at::cpu::as_strided(input, size, stride, storage_offset);
}

at::Tensor &txda_fill__scalar(at::Tensor &self, const at::Scalar &value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "only support txda");
  TORCH_CHECK(self.is_contiguous());
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  auto _data = static_cast<float *>(self.mutable_data_ptr());
  for (size_t idx = 0; idx < self.numel(); idx++) {
    _data[idx] = value.toFloat();
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("to.Device", &txda_to_device);
  m.impl("fill_.Scalar", &txda_fill__scalar);
  m.impl("_copy_from", &txda__copy_from);
  m.impl("empty.memory_format", &txda_empty_memory_format);
  m.impl("empty_strided", &txda_empty_strided);
  m.impl("as_strided", &txda_as_strided);
}

bool init_device() {
  // return init_txda_runtime();
  return true;
}

bool cleanup_device() {
  // cleanup_txda_runtime();
  return true;
}

int current_device() { return 0; }

int current_stream(int id) { return 0; }

void set_device(int id) {}

c10::Device get_txda_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("current_device", &current_device, "get current tx device");
  m.def("current_stream", &current_stream, "get current tx stream");
  m.def("set_device", &set_device, "set tx device");
  m.def("get_txda_device", &get_txda_device, "get tx device");
  m.def("init_device", &init_device, "initialize tx device");
  m.def("cleanup_device", &cleanup_device, "cleanup tx device");
}
