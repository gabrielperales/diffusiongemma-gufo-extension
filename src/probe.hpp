#ifndef DIFFUSION_GEMMA_EXT_PROBE_HPP_
#define DIFFUSION_GEMMA_EXT_PROBE_HPP_

#include <string_view>

namespace gufo::models::diffusion_gemma {

/// True when `model_root` is a DiffusionGemma checkpoint directory whose
/// config.json declares `model_type: diffusion_gemma` and the one shape the
/// kernels are written for. Reads only config.json; never touches the GPU.
[[nodiscard]] bool ProbeDiffusionGemma(std::string_view model_root);

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_PROBE_HPP_
