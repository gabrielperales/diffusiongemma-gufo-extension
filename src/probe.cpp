#include "src/probe.hpp"

#include <exception>
#include <filesystem>

#include "src/config.hpp"

namespace gufo::models::diffusion_gemma {

bool ProbeDiffusionGemma(std::string_view model_root) {
  const std::filesystem::path root{model_root};
  if (!IsDiffusionGemmaDirectory(root))
    return false;
  // A DiffusionGemma directory with another shape is not ours to run; the
  // loader would reject it, so the probe declines rather than claim it.
  try {
    (void)LoadConfig(root);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace gufo::models::diffusion_gemma
