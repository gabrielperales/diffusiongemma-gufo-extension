#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/probe.hpp"
#include "src/tool_dialect.hpp"

namespace gufo::models::diffusion_gemma {

#if !defined(ENGINE_ENABLE_HIP)
// CPU-only builds still recognise the checkpoint, so diagnostics report the
// provider as present but unable to run.
void RegisterDiffusionGemmaCpu(ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "diffusiongemma-26b-a4b",
      .probe = &ProbeDiffusionGemma,
      .probe_gguf = &ProbeDiffusionGemmaGguf,
      .create = nullptr,
      .accepts_unsized_tensors = true,
  });
}
#endif

}  // namespace gufo::models::diffusion_gemma

void diffusion_gemma_register(gufo::ext::Registry& registry) {
  // Before any runner exists, as the dialect registry requires.
  gufo::models::diffusion_gemma::RegisterToolDialect();
#if !defined(ENGINE_ENABLE_HIP)
  gufo::models::diffusion_gemma::RegisterDiffusionGemmaCpu(registry);
#else
  (void)registry;
#endif
}
