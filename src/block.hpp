#ifndef DIFFUSION_GEMMA_EXT_BLOCK_HPP_
#define DIFFUSION_GEMMA_EXT_BLOCK_HPP_

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

#include "src/config.hpp"

namespace gufo::models::diffusion_gemma {

/// One reply being generated, a canvas (block) at a time, by the
/// block-diffusion sampler. Host state only: the caller owns the session and
/// runs the forward passes, so several replies can share each pass.
///
/// Seeds, the random stream and every per-step decision follow Bubo's
/// Engine, so a reply generated here with a seed matches Engine::Generate
/// with that seed (up to floating-point differences in the kernels).
struct BlockGeneration {
  static constexpr int kMaxSuppressed = 4;

  DiffusionDefaults diffusion;
  std::mt19937_64 rng;
  std::uint64_t seed{0};
  /// End-of-sequence IDs never sampled (fixed-length generation).
  std::int32_t suppressed[kMaxSuppressed]{};
  int suppressed_count{0};

  /// Blocks finished so far; the next block draws its seed from it.
  int block{0};
  /// A block is being denoised.
  bool active{false};
  /// The last block's canvas belongs wholly to the reply and must be encoded
  /// into the cache before the next block starts.
  bool commit_pending{false};

  std::uint64_t block_seed{0};
  int steps{0};
  int total_steps{0};
  std::vector<std::int32_t> current, argmax_canvas, previous_argmax;
  std::vector<float> row_entropy;
  std::vector<std::int32_t> row_argmax, row_sample;
};

/// Starts a reply: seeds the random stream and, for fixed-length generation,
/// suppresses the end-of-sequence tokens. Throws if there are too many.
void BeginGeneration(BlockGeneration& generation, const Config& config,
                     const DiffusionDefaults& diffusion, std::uint64_t seed,
                     bool suppress_end);

/// One canvas's denoising step, as the forward pass needs it. The pass
/// writes `entropy`, `argmax` and `sample` (canvas rows each).
struct BlockStep {
  std::size_t generation{0};  ///< Index into RunBlocks' `generations`.
  const std::int32_t* tokens{nullptr};
  bool self_condition{false};
  float temperature{1.0F};
  std::uint64_t seed{0};
  const std::int32_t* suppressed{nullptr};
  int suppressed_count{0};
  float* entropy{nullptr};
  std::int32_t* argmax{nullptr};
  std::int32_t* sample{nullptr};
};

/// A finished canvas to encode into its generation's cache (canvas rows).
struct BlockCommit {
  std::size_t generation{0};
  const std::int32_t* tokens{nullptr};
};

/// Runs one forward pass: the steps, then the commits.
using BlockForward = std::function<void(std::span<const BlockStep>,
                                        std::span<const BlockCommit>)>;

struct BlockOutcome {
  /// The block's reply tokens: up to the first end-of-sequence token and at
  /// most the budget.
  std::vector<std::int32_t> tokens;
  bool end_of_sequence{false};
  /// No block fits the context: the reply cannot continue.
  bool context_full{false};
  bool cancelled{false};
  int steps{0};
};

/// Generates the next block of every generation, batching their denoising
/// steps (and any pending commits) into shared forward passes, until each
/// block has converged or reached the step limit. `cached[i]` is generation
/// i's cache length before its pending commit; `budgets[i]` the most tokens
/// it may still return. `cancelled(i)`, when set, is polled between passes.
std::vector<BlockOutcome> RunBlocks(
    const Config& config, int max_context,
    std::span<BlockGeneration* const> generations,
    std::span<const int> cached, std::span<const int> budgets,
    const BlockForward& forward,
    const std::function<bool(std::size_t)>& cancelled = {});

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_BLOCK_HPP_
