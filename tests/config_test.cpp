// Configuration and probe checks on a synthetic config.json with the shape of
// google/diffusiongemma-26B-A4B-it. No weights or GPU needed.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "src/config.hpp"
#include "src/probe.hpp"

namespace dg = gufo::models::diffusion_gemma;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

std::string LayerTypes(int layers) {
  std::string out = "[";
  for (int i = 0; i < layers; ++i) {
    if (i)
      out += ",";
    // Every sixth layer is global, as in the released checkpoint.
    out += (i % 6 == 5) ? "\"full_attention\"" : "\"sliding_attention\"";
  }
  return out + "]";
}

std::string Config(const std::string& model_type, int canvas) {
  return R"({"model_type":")" + model_type + R"(","canvas_length":)" +
         std::to_string(canvas) + R"(,"architectures":["DiffusionGemmaForBlockDiffusion"],
  "text_config":{"hidden_size":2816,"num_hidden_layers":30,
  "num_attention_heads":16,"head_dim":256,"num_key_value_heads":8,
  "global_head_dim":512,"num_global_key_value_heads":2,
  "intermediate_size":2112,"num_experts":128,"top_k_experts":8,
  "moe_intermediate_size":704,"vocab_size":262144,"sliding_window":1024,
  "max_position_embeddings":262144,"rms_norm_eps":1e-6,
  "final_logit_softcapping":30.0,"pad_token_id":0,"bos_token_id":2,
  "layer_types":)" + LayerTypes(30) + R"(,
  "rope_parameters":{
    "sliding_attention":{"rope_type":"default","rope_theta":10000.0},
    "full_attention":{"rope_type":"proportional","rope_theta":1000000.0,
                      "partial_rotary_factor":0.25}}}})";
}

void Write(const fs::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary) << text;
}

fs::path MakeDir(const char* name) {
  auto dir = fs::temp_directory_path() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

}  // namespace

int main() {
  const auto good = MakeDir("dg_config_good");
  Write(good / "config.json", Config("diffusion_gemma", 256));
  Write(good / "generation_config.json",
        R"({"eos_token_id":[1,106,50],"max_denoising_steps":48,"t_min":0.4,
            "t_max":0.8,"stability_threshold":1,"confidence_threshold":0.005,
            "sampler_config":{"entropy_bound":0.1}})");

  Check(dg::ProbeDiffusionGemma(good.string()), "probe accepts checkpoint");
  const auto config = dg::LoadConfig(good);
  Check(config.layers == 30 && config.canvas == 256, "layers and canvas");
  Check(config.is_global.size() == 30 && config.is_global[5] &&
            !config.is_global[0],
        "layer types");
  Check(config.eos_ids.size() == 3 && config.eos_ids[2] == 50,
        "eos ids from generation_config.json");
  const auto defaults = dg::LoadDiffusionDefaults(good);
  Check(defaults.max_denoising_steps == 48 && defaults.entropy_bound == 0.1F,
        "diffusion defaults");

  const auto other = MakeDir("dg_config_other");
  Write(other / "config.json", Config("gemma4", 256));
  Check(!dg::ProbeDiffusionGemma(other.string()), "probe rejects gemma4");

  const auto shape = MakeDir("dg_config_shape");
  Write(shape / "config.json", Config("diffusion_gemma", 512));
  Check(!dg::ProbeDiffusionGemma(shape.string()),
        "probe rejects an unsupported canvas");

  Check(!dg::ProbeDiffusionGemma((good / "missing").string()),
        "probe rejects a missing directory");

  if (failures == 0)
    std::puts("diffusion_gemma config: ok");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
