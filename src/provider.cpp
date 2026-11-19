#include <gufo/ext/generated/diffusion_gemma_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/probe.hpp"

// Phase 1: the provider recognises the checkpoint but cannot run it yet.
// Phase 4 adds the TextModelRunner factory and the prompt/chat/bench drivers.
void diffusion_gemma_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_diffusion_gemma::
                           kProvider_diffusiongemma_26b_a4b);
  registry.AddTextProvider({
      .id = "diffusiongemma-26b-a4b",
      .probe = &gufo::models::diffusion_gemma::ProbeDiffusionGemma,
      .create = nullptr,
  });
}
