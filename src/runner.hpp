#ifndef DIFFUSION_GEMMA_EXT_RUNNER_HPP_
#define DIFFUSION_GEMMA_EXT_RUNNER_HPP_

#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

namespace gufo::models::diffusion_gemma {

/// The TextModelRunner factory of the diffusiongemma-26b-a4b provider.
std::shared_ptr<ext::TextModelRunner> CreateDiffusionGemmaRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_RUNNER_HPP_
