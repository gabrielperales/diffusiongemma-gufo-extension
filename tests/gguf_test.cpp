// The GGUF path must equal the checkpoint path: the configuration read from
// a llama.cpp DiffusionGemma GGUF against config.json, and the tokenizer
// built from its metadata against tokenizer.json. Needs
// DIFFUSION_GEMMA_GGUF (a .gguf file) and DIFFUSION_GEMMA_MODEL (a
// safetensors checkpoint directory); skips (77) without them.
#include <cstdio>
#include <cstdlib>
#include <gufo/ext/gguf_reader.hpp>
#include <string>
#include <vector>

#include "src/chat.hpp"
#include "src/config.hpp"
#include "src/tokenizer.hpp"

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  const char* gguf = std::getenv("DIFFUSION_GEMMA_GGUF");
  const char* root = std::getenv("DIFFUSION_GEMMA_MODEL");
  if (gguf == nullptr || root == nullptr) {
    std::puts("skipped: set DIFFUSION_GEMMA_GGUF and DIFFUSION_GEMMA_MODEL");
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s: %s\n", gguf, error.c_str());
    return 1;
  }
  Check(dg::IsDiffusionGemmaGguf(*reader), "architecture");

  const auto a = dg::LoadConfigGguf(*reader);
  const auto b = dg::LoadConfig(root);
  Check(a.hidden == b.hidden && a.layers == b.layers && a.heads == b.heads,
        "hidden, layers, heads");
  Check(a.sliding_head_dim == b.sliding_head_dim &&
            a.global_head_dim == b.global_head_dim &&
            a.sliding_kv_heads == b.sliding_kv_heads &&
            a.global_kv_heads == b.global_kv_heads,
        "attention shape");
  Check(a.intermediate == b.intermediate && a.experts == b.experts &&
            a.top_k == b.top_k && a.expert_intermediate == b.expert_intermediate,
        "feed-forward shape");
  Check(a.vocab == b.vocab && a.sliding_window == b.sliding_window &&
            a.canvas == b.canvas && a.max_positions == b.max_positions,
        "vocabulary, window, canvas, positions");
  Check(a.rms_eps == b.rms_eps && a.final_softcap == b.final_softcap &&
            a.sliding_theta == b.sliding_theta &&
            a.global_theta == b.global_theta &&
            a.global_rotary_fraction == b.global_rotary_fraction,
        "norm, softcap and RoPE");
  Check(a.is_global == b.is_global, "layer types");
  Check(a.eos_ids == b.eos_ids, "end tokens");
  Check(a.pad_id == b.pad_id && a.bos_id == b.bos_id, "pad and bos");

  const dg::Tokenizer from_gguf(*reader);
  const dg::Tokenizer from_json{std::filesystem::path(root)};
  Check(from_gguf.vocab_size() == from_json.vocab_size(), "vocabulary size");

  std::vector<std::string> corpus = {
      "Hello, world!",
      "  leading and   inner   spaces\tand\ttabs\n\nnewlines\n",
      "Explica en dos frases qué es la difusión por bloques. ¿Sí? ¡Ñandú!",
      "日本語のテキストと中文混合 текст на русском",
      "emoji 🦉🧪👩‍👩‍👧 and invalid \xFF\xFE bytes",
      "code: for (int i = 0; i < n; ++i) { x[i] = y[i] * 2; }",
      "<s>literal tags</s> <bos> <eos> <|turn>user\nhi<turn|>",
      "<|tool_call>call:get_weather{city:<|\"|>Madrid<|\"|>}<tool_call|>",
      "<|channel>thought\nreasoning<channel|>answer",
      std::string(300, 'a') + std::string(70, ' ') + "end",
  };
  std::vector<dg::Message> messages(2);
  messages[0].role = "system";
  messages[0].content = "Eres un asistente útil.";
  messages[1].content = "¿Qué tiempo hace en Madrid?";
  corpus.push_back(dg::RenderChat(messages, {}, true));

  for (const auto& text : corpus) {
    const auto x = from_gguf.Encode(text);
    const auto y = from_json.Encode(text);
    Check(x == y, "encode: " + text.substr(0, 40));
    Check(from_gguf.Decode(x, false) == from_json.Decode(y, false),
          "decode: " + text.substr(0, 40));
    Check(from_gguf.Decode(x, true) == from_json.Decode(y, true),
          "decode skipping specials: " + text.substr(0, 40));
  }
  for (std::int32_t id = 0; id < static_cast<std::int32_t>(from_json.vocab_size());
       ++id)
    if (from_gguf.IsSpecial(id) != from_json.IsSpecial(id)) {
      Check(false, "special flag of token " + std::to_string(id));
      break;
    }

  if (failures == 0)
    std::puts("diffusion_gemma gguf: ok");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
