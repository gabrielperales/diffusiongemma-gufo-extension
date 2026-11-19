#include "src/block.hpp"

#include <algorithm>
#include <stdexcept>

#include "src/denoise.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

// Per-block seed offset: an odd constant, so blocks of one reply draw from
// distinct streams (as in Bubo's Engine).
constexpr std::uint64_t kBlockSeedStride = 0x1000193ULL;

/// Starts the generation's next canvas; false when it would not fit the
/// context.
bool StartBlock(BlockGeneration& g, const Config& c, int cached,
                int max_context) {
  if (cached + c.canvas > max_context)
    return false;
  g.block_seed = MixSeed(g.seed + kBlockSeedStride * (g.block + 1));
  // The canvas starts as random tokens; rows not accepted at a step are
  // drawn again.
  std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
  g.current.resize(c.canvas);
  for (auto& token : g.current)
    token = uniform(g.rng);
  g.row_entropy.resize(c.canvas);
  g.row_argmax.resize(c.canvas);
  g.row_sample.resize(c.canvas);
  g.previous_argmax.clear();
  g.steps = 0;
  g.active = true;
  return true;
}

}  // namespace

void BeginGeneration(BlockGeneration& g, const Config& c,
                     const DiffusionDefaults& diffusion, std::uint64_t seed,
                     bool suppress_end) {
  g = BlockGeneration{};
  g.diffusion = diffusion;
  g.seed = seed;
  g.rng.seed(MixSeed(seed));
  if (suppress_end) {
    if (c.eos_ids.size() > BlockGeneration::kMaxSuppressed)
      throw std::invalid_argument("too many end-of-sequence tokens");
    for (const int id : c.eos_ids)
      g.suppressed[g.suppressed_count++] = id;
  }
}

std::vector<BlockOutcome> RunBlocks(
    const Config& c, int max_context,
    std::span<BlockGeneration* const> generations, std::span<const int> cached,
    std::span<const int> budgets, const BlockForward& forward,
    const std::function<bool(std::size_t)>& cancelled) {
  const std::size_t n = generations.size();
  if (cached.size() != n || budgets.size() != n)
    throw std::invalid_argument("RunBlocks: mismatched generation inputs");
  std::vector<BlockOutcome> outcomes(n);
  std::vector<bool> done(n, false);
  std::vector<int> length(cached.begin(), cached.end());
  for (std::size_t i = 0; i < n; ++i)
    if (budgets[i] <= 0)
      done[i] = true;

  std::vector<BlockStep> steps;
  std::vector<BlockCommit> commits;
  for (;;) {
    steps.clear();
    commits.clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (done[i])
        continue;
      BlockGeneration& g = *generations[i];
      if (cancelled && cancelled(i)) {
        outcomes[i].cancelled = true;
        g.active = false;
        done[i] = true;
        continue;
      }
      // A finished canvas is encoded in its own pass row range; the next
      // block starts in the pass after.
      if (g.commit_pending) {
        commits.push_back({.generation = i, .tokens = g.argmax_canvas.data()});
        continue;
      }
      if (!g.active && !StartBlock(g, c, length[i], max_context)) {
        outcomes[i].context_full = true;
        done[i] = true;
        continue;
      }
      const DiffusionDefaults& d = g.diffusion;
      // Each step draws its Gumbel noise from its own stream.
      steps.push_back(BlockStep{
          .generation = i,
          .tokens = g.current.data(),
          .self_condition = g.steps > 0,
          .temperature = StepTemperature(d, d.max_denoising_steps - g.steps),
          .seed = MixSeed(g.block_seed ^ (static_cast<std::uint64_t>(g.steps)
                                          << 40)),
          .suppressed = g.suppressed,
          .suppressed_count = g.suppressed_count,
          .entropy = g.row_entropy.data(),
          .argmax = g.row_argmax.data(),
          .sample = g.row_sample.data()});
    }
    if (steps.empty() && commits.empty())
      break;
    forward(steps, commits);

    for (const auto& commit : commits) {
      BlockGeneration& g = *generations[commit.generation];
      g.commit_pending = false;
      length[commit.generation] += c.canvas;
    }
    std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
    for (const auto& step : steps) {
      const std::size_t i = step.generation;
      BlockGeneration& g = *generations[i];
      const DiffusionDefaults& d = g.diffusion;
      ++g.steps;
      const auto accepted = AcceptRows(g.row_entropy, d.entropy_bound);
      for (int r = 0; r < c.canvas; ++r)
        g.current[r] = accepted[r] ? g.row_sample[r] : uniform(g.rng);
      g.argmax_canvas = g.row_argmax;
      const bool converged =
          BlockConverged(d, g.previous_argmax, g.row_argmax, g.row_entropy);
      g.previous_argmax = g.row_argmax;
      if (!converged && g.steps < d.max_denoising_steps)
        continue;

      // The block is finished: its reply is the canvas up to the first end
      // token, within the budget.
      g.active = false;
      g.total_steps += g.steps;
      ++g.block;
      const auto reply = TakeReply(
          g.argmax_canvas,
          std::span<const int>(c.eos_ids.data(), c.eos_ids.size()),
          budgets[i]);
      BlockOutcome& out = outcomes[i];
      out.tokens.assign(g.argmax_canvas.begin(),
                        g.argmax_canvas.begin() + reply.tokens);
      out.end_of_sequence = reply.end_of_sequence;
      out.steps = g.steps;
      // Only a canvas that belongs wholly to the reply is cached: the next
      // block continues after it. Otherwise the reply ends here.
      g.commit_pending = !reply.end_of_sequence && reply.tokens == c.canvas;
      done[i] = true;
    }
  }
  return outcomes;
}

}  // namespace gufo::models::diffusion_gemma
