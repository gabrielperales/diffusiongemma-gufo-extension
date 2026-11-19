#ifndef DIFFUSION_GEMMA_EXT_DEVICE_BUFFER_HPP_
#define DIFFUSION_GEMMA_EXT_DEVICE_BUFFER_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <utility>

#include "src/hip_check.hpp"

namespace gufo::models::diffusion_gemma {

/// Owning device allocation, zero-initialised.
class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0)
      return;
    HipCheck(hipMalloc(&data_, bytes), "allocation");
    // The memset runs on the null stream, which the engine's non-blocking
    // stream does not wait for: finish it before the buffer is used.
    HipCheck(hipMemset(data_, 0, bytes), "memset");
    HipCheck(hipStreamSynchronize(nullptr), "memset");
  }
  ~Buffer() {
    if (data_)
      (void)hipFree(data_);
  }
  Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }
  template<typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_DEVICE_BUFFER_HPP_
