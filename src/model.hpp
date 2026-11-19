#ifndef DIFFUSION_GEMMA_EXT_MODEL_HPP_
#define DIFFUSION_GEMMA_EXT_MODEL_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "src/cache_state.hpp"
#include "src/config.hpp"
#include "src/device_buffer.hpp"
#include "src/image_prompt.hpp"
#include "src/kernels.hpp"
#include "src/tokenizer.hpp"

namespace gufo::models::diffusion_gemma {

class Profiler;

struct ModelOptions {
  /// Largest prompt plus generated length; sizes the global-layer KV cache.
  int max_context{131072};
  /// Prompt tokens per encoder pass (rounded up to 64, at least 256).
  int prefill_chunk{8192};
  /// Canvases one forward pass may denoise together.
  int max_batch{1};
  /// Sliding positions kept beyond the attention window, so that a prompt
  /// diverging from the cache shortly before its end still reuses it
  /// (CacheState::Reuse). Gufo's continuation cache restores prefixes
  /// itself, so its runner keeps none.
  int reuse_history{4096};
  /// Run every prompt chunk through the shared stage caches, so that each
  /// session's sliding caches hold the kept positions plus one canvas only.
  /// Implied when max_batch > 1.
  bool stage_prompts{false};
  /// Print per-phase GPU time to stderr.
  bool profile{false};
};

/// Told before a forward pass writes a session's KV rows [first, end): the
/// continuation cache must stop sharing them first.
class SessionHooks {
public:
  virtual ~SessionHooks() = default;
  virtual void BeforeWrite(int first, int end) = 0;
};

/// One sequence's device state. KV caches per layer are bf16
/// [kv heads][capacity][head dim]: global layers hold position p at slot p,
/// sliding layers the newest `cache.sliding_valid()` positions at slots
/// [0, sliding_valid). A canvas being denoised occupies the slots just past
/// them. `probs` is the canvas's self-conditioning distribution.
struct Session {
  explicit Session(CacheState state) : cache(std::move(state)) {}
  CacheState cache;
  std::vector<Buffer> k_cache, v_cache;
  Buffer probs;
  Buffer pinned;
  SessionHooks* hooks{nullptr};
  // Bookkeeping of the standalone Engine's scheduler; unused by the runner.
  bool busy{false};
  std::uint64_t last_used{0};
};

struct ScoreHost {
  std::vector<std::int32_t> top_ids;
  std::vector<float> top_logprobs;
  std::vector<float> label_logprobs;
  std::vector<float> logsumexp;
  std::vector<std::int32_t> argmax;
};

/// One canvas of a batched denoising step: its session, tokens, sampler
/// settings, and where its per-row results go.
struct CanvasStep {
  Session* session{nullptr};
  const std::int32_t* tokens{nullptr};
  bool self_condition{false};
  bool make_probabilities{true};
  bool has_pins{false};
  float temperature{1.0F};
  std::uint64_t seed{0};
  kernels::SuppressedTokens suppressed;
  float* entropy{nullptr};
  std::int32_t* argmax{nullptr};
  std::int32_t* sample{nullptr};
  ScoreHost* scores{nullptr};
  const std::int32_t* label_ids{nullptr};
  int label_count{0};
};

/// Causal rows appended to a session's cache: a prompt chunk or a
/// committed canvas.
struct EncodeWork {
  Session* session{nullptr};
  const std::int32_t* tokens{nullptr};
  int rows{0};
  const ImageInput* image{nullptr};
};

/// DiffusionGemma weights, scratch and kernels on gfx1151. One forward pass
/// runs at a time; callers serialise passes (gpu_mutex) and own sessions.
class Model {
public:
  Model(const std::filesystem::path& root, const ModelOptions& options);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] const Config& config() const;
  [[nodiscard]] const DiffusionDefaults& defaults() const;
  [[nodiscard]] const Tokenizer& tokenizer() const;
  /// The options after normalisation (chunk rounding, context clamping).
  [[nodiscard]] const ModelOptions& options() const;
  [[nodiscard]] std::size_t weight_bytes() const;
  /// Scratch rows of one pass: max(prefill chunk, max_batch x canvas).
  [[nodiscard]] int rows() const;
  [[nodiscard]] int global_capacity() const;
  [[nodiscard]] int sliding_capacity() const;
  /// Device bytes one session holds.
  [[nodiscard]] std::size_t session_bytes() const;

  [[nodiscard]] std::unique_ptr<Session> CreateSession() const;

  /// One forward pass over the stacked rows of `canvases` (a denoising step
  /// each, after its own session's cache) and then `encodes` (rows appended
  /// to their sessions' caches; at most one may need the stage caches).
  /// Leaves each canvas's per-row entropy, argmax and sample in its host
  /// arrays and its self-conditioning probabilities in its session.
  void Forward(std::span<const CanvasStep> canvases,
               std::span<const EncodeWork> encodes);
  /// Encodes `tokens` into `session` in prompt chunks; returns milliseconds.
  double Encode(Session& session, std::span<const std::int32_t> tokens);

  /// Device logits [canvases][canvas][vocab] of the last pass.
  [[nodiscard]] const float* last_logits() const;
  [[nodiscard]] hipStream_t stream() const;
  [[nodiscard]] Profiler& profiler() const;
  /// Prints the profile's active-expert count (profile mode only).
  void ReportExperts() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_MODEL_HPP_
