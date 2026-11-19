// A GGUF written by tools/convert_nvfp4_gguf.py must hold NVIDIA's NVFP4
// decoder tensors unchanged: every tensor of the checkpoint the loader can
// use, with its dtype, element count and bytes. Needs DIFFUSION_GEMMA_MODEL
// (the safetensors checkpoint) and DIFFUSION_GEMMA_NVFP4_GGUF; skips (77)
// without them.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/weights.hpp"

namespace dg = gufo::models::diffusion_gemma;

namespace {

bool SameBytes(const dg::TensorEntry& a, const dg::TensorEntry& b) {
  std::ifstream fa(a.file, std::ios::binary), fb(b.file, std::ios::binary);
  fa.seekg(static_cast<std::streamoff>(a.offset));
  fb.seekg(static_cast<std::streamoff>(b.offset));
  std::vector<char> x(1 << 20), y(1 << 20);
  for (std::uint64_t left = a.bytes; left > 0;) {
    const auto n = static_cast<std::streamsize>(
        std::min<std::uint64_t>(left, x.size()));
    if (!fa.read(x.data(), n) || !fb.read(y.data(), n) ||
        !std::equal(x.begin(), x.begin() + n, y.begin()))
      return false;
    left -= static_cast<std::uint64_t>(n);
  }
  return true;
}

std::uint64_t Elements(const dg::TensorEntry& t) {
  std::uint64_t n = 1;
  for (const auto d : t.shape)
    n *= static_cast<std::uint64_t>(d);
  return n;
}

}  // namespace

int main() {
  const char* root = std::getenv("DIFFUSION_GEMMA_MODEL");
  const char* gguf = std::getenv("DIFFUSION_GEMMA_NVFP4_GGUF");
  if (root == nullptr || gguf == nullptr) {
    std::puts("skipped: set DIFFUSION_GEMMA_MODEL and DIFFUSION_GEMMA_NVFP4_GGUF");
    return 77;
  }
  const auto checkpoint = dg::ReadSafetensorsIndex(root);
  const auto packed = dg::ReadNvfp4GgufIndex(gguf);
  int failures = 0, compared = 0;
  for (const auto& [name, source] : checkpoint) {
    const bool kept =
        (name.starts_with("model.decoder.") ||
         name.starts_with("model.encoder.language_model.")) &&
        !name.ends_with(".input_scale");
    const auto found = packed.find(name);
    if (!kept) {
      if (found != packed.end()) {
        std::fprintf(stderr, "FAIL: %s should not be packed\n", name.c_str());
        ++failures;
      }
      continue;
    }
    if (found == packed.end()) {
      std::fprintf(stderr, "FAIL: %s is missing\n", name.c_str());
      ++failures;
      continue;
    }
    const auto& copy = found->second;
    if (copy.dtype != source.dtype || Elements(copy) != Elements(source) ||
        copy.bytes != source.bytes || !SameBytes(source, copy)) {
      std::fprintf(stderr, "FAIL: %s differs\n", name.c_str());
      ++failures;
    }
    ++compared;
  }
  if (!packed.contains("rope_freqs.weight")) {
    std::fprintf(stderr, "FAIL: rope_freqs.weight is missing\n");
    ++failures;
  }
  if (failures == 0)
    std::printf("diffusion_gemma nvfp4 gguf: ok (%d tensors identical)\n",
                compared);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
