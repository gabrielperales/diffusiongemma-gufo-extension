// DiffusionGemma architecture from a llama.cpp GGUF (architecture
// "diffusion-gemma"), as unsloth/diffusiongemma-26B-A4B-it-GGUF writes it.
#include <cmath>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "src/config.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

const std::string kArch = kGgufArchitecture;

[[noreturn]] void Missing(const std::string& key) {
  throw std::runtime_error("DiffusionGemma GGUF is missing " + key);
}

std::uint64_t Uint(const core::GgufReader& r, const std::string& key) {
  if (const auto value = r.GetMetadataUint64(key))
    return *value;
  if (const auto value = r.GetMetadataUint32(key))
    return *value;
  Missing(key);
}

float Float(const core::GgufReader& r, const std::string& key) {
  if (const auto value = r.GetMetadataFloat32(key))
    return *value;
  Missing(key);
}

/// A numeric or boolean array, as unsigned values.
std::vector<std::uint64_t> UintArray(const core::GgufReader& r,
                                     const std::string& key) {
  const auto* value = r.FindMetadata(key);
  if (value == nullptr)
    Missing(key);
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&value->value))
    return *u;
  if (const auto* i = std::get_if<std::vector<std::int64_t>>(&value->value))
    return {i->begin(), i->end()};
  throw std::runtime_error("DiffusionGemma GGUF: " + key +
                           " is not an integer array");
}

int TokenId(const std::vector<std::string_view>& tokens,
            std::string_view piece) {
  for (std::size_t i = 0; i < tokens.size(); ++i)
    if (tokens[i] == piece)
      return static_cast<int>(i);
  return -1;
}

}  // namespace

bool IsDiffusionGemmaGguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
         kGgufArchitecture;
}

Config LoadConfigGguf(const core::GgufReader& r) {
  if (!IsDiffusionGemmaGguf(r))
    throw std::runtime_error("not a diffusion-gemma GGUF");
  Config c;
  c.hidden = static_cast<int>(Uint(r, kArch + ".embedding_length"));
  c.layers = static_cast<int>(Uint(r, kArch + ".block_count"));
  c.heads = static_cast<int>(Uint(r, kArch + ".attention.head_count"));
  c.sliding_head_dim =
      static_cast<int>(Uint(r, kArch + ".attention.key_length_swa"));
  c.global_head_dim =
      static_cast<int>(Uint(r, kArch + ".attention.key_length"));
  c.intermediate = static_cast<int>(Uint(r, kArch + ".feed_forward_length"));
  c.experts = static_cast<int>(Uint(r, kArch + ".expert_count"));
  c.top_k = static_cast<int>(Uint(r, kArch + ".expert_used_count"));
  c.expert_intermediate =
      static_cast<int>(Uint(r, kArch + ".expert_feed_forward_length"));
  c.sliding_window =
      static_cast<int>(Uint(r, kArch + ".attention.sliding_window"));
  c.max_positions = static_cast<int>(Uint(r, kArch + ".context_length"));
  c.canvas = static_cast<int>(Uint(r, "diffusion.canvas_length"));
  c.rms_eps = Float(r, kArch + ".attention.layer_norm_rms_epsilon");
  c.final_softcap = Float(r, kArch + ".final_logit_softcapping");
  c.sliding_theta = Float(r, kArch + ".rope.freq_base_swa");
  c.global_theta = Float(r, kArch + ".rope.freq_base");

  // Per layer: sliding (true) or full attention, and its KV heads.
  const auto pattern = UintArray(r, kArch + ".attention.sliding_window_pattern");
  const auto kv_heads = UintArray(r, kArch + ".attention.head_count_kv");
  if (static_cast<int>(pattern.size()) != c.layers ||
      kv_heads.size() != pattern.size())
    throw std::runtime_error("DiffusionGemma GGUF: per-layer arrays do not "
                             "match block_count");
  c.sliding_kv_heads = c.global_kv_heads = 0;
  for (int l = 0; l < c.layers; ++l) {
    const bool global = pattern[l] == 0;
    c.is_global.push_back(global);
    int& heads = global ? c.global_kv_heads : c.sliding_kv_heads;
    if (heads != 0 && heads != static_cast<int>(kv_heads[l]))
      throw std::runtime_error("DiffusionGemma GGUF: KV heads differ between "
                               "layers of one kind");
    heads = static_cast<int>(kv_heads[l]);
  }

  // Proportional RoPE on the global layers: rope_freqs holds a factor per
  // frequency pair, 1 for the rotated pairs and ~1e30 (no rotation) for the
  // rest.
  const auto* freqs = r.FindTensor("rope_freqs.weight");
  if (freqs == nullptr || freqs->type != core::GgmlType::kF32 ||
      freqs->ElementCount() != static_cast<std::uint64_t>(c.global_head_dim) / 2)
    throw std::runtime_error("DiffusionGemma GGUF: invalid rope_freqs.weight");
  int rotated = 0;
  for (std::uint64_t i = 0; i < freqs->ElementCount(); ++i) {
    float factor;
    std::memcpy(&factor, static_cast<const char*>(freqs->data) + 4 * i, 4);
    if (factor == 1.0F) {
      if (static_cast<int>(i) != rotated)
        throw std::runtime_error(
            "DiffusionGemma GGUF: rotated RoPE pairs are not leading");
      ++rotated;
    }
  }
  c.global_rotary_fraction =
      static_cast<float>(rotated) / static_cast<float>(freqs->ElementCount());

  const auto tokens = r.GetMetadataStringArray("tokenizer.ggml.tokens");
  c.vocab = static_cast<int>(tokens.size());
  c.pad_id = static_cast<int>(Uint(r, "tokenizer.ggml.padding_token_id"));
  c.bos_id = static_cast<int>(Uint(r, "tokenizer.ggml.bos_token_id"));
  c.eos_ids = {static_cast<int>(Uint(r, "tokenizer.ggml.eos_token_id"))};
  for (const auto* piece : {"<turn|>", "<|tool_response>"})
    if (const int id = TokenId(tokens, piece); id >= 0)
      c.eos_ids.push_back(id);

  ValidateConfig(c);
  if (std::abs(c.global_rotary_fraction - 0.25F) > 1e-6F)
    throw std::runtime_error(
        "unsupported DiffusionGemma config: global rotary fraction");
  return c;
}

DiffusionDefaults LoadDiffusionDefaultsGguf(const core::GgufReader& r) {
  DiffusionDefaults d;
  const auto integer = [&](const char* key, int& out) {
    if (const auto v = r.GetMetadataUint32(key))
      out = static_cast<int>(*v);
    else if (const auto v64 = r.GetMetadataUint64(key))
      out = static_cast<int>(*v64);
  };
  const auto real = [&](const char* key, float& out) {
    if (const auto v = r.GetMetadataFloat32(key))
      out = *v;
  };
  integer("diffusion.eb_max_steps", d.max_denoising_steps);
  real("diffusion.eb_t_min", d.t_min);
  real("diffusion.eb_t_max", d.t_max);
  real("diffusion.eb_entropy_bound", d.entropy_bound);
  integer("diffusion.eb_stability_threshold", d.stability_threshold);
  real("diffusion.eb_confidence_threshold", d.confidence_threshold);
  if (d.max_denoising_steps <= 0 || d.t_min <= 0.0F || d.t_max < d.t_min ||
      d.entropy_bound < 0.0F || d.stability_threshold < 0 ||
      d.confidence_threshold < 0.0F)
    throw std::runtime_error("DiffusionGemma GGUF: invalid sampler settings");
  return d;
}

}  // namespace gufo::models::diffusion_gemma
