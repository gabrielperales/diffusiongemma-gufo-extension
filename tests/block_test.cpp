// Block-diffusion control flow (RunBlocks) against a fake forward pass:
// convergence, budgets, end of sequence, commits between blocks, context
// limits, cancellation, batching and seeded reproducibility.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/block.hpp"

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

dg::Config SmallConfig() {
  dg::Config c;
  c.canvas = 4;
  c.vocab = 16;
  c.eos_ids = {1};
  return c;
}

dg::DiffusionDefaults Defaults() {
  dg::DiffusionDefaults d;
  d.max_denoising_steps = 8;
  d.stability_threshold = 1;
  d.confidence_threshold = 0.5F;
  d.entropy_bound = 0.1F;
  return d;
}

/// A forward pass whose every canvas settles on `answer(generation)` at
/// zero entropy: a block converges on its second step.
struct FakeForward {
  std::vector<std::vector<std::int32_t>> answers;
  int passes{0};
  std::vector<std::size_t> committed;
  std::vector<std::size_t> batch_sizes;
  std::vector<std::uint64_t> seeds;

  void operator()(std::span<const dg::BlockStep> steps,
                  std::span<const dg::BlockCommit> commits) {
    ++passes;
    batch_sizes.push_back(steps.size());
    for (const auto& commit : commits)
      committed.push_back(commit.generation);
    for (const auto& step : steps) {
      const auto& answer = answers[step.generation];
      seeds.push_back(step.seed);
      for (std::size_t r = 0; r < answer.size(); ++r) {
        step.entropy[r] = 0.0F;
        step.argmax[r] = answer[r];
        step.sample[r] = answer[r];
      }
    }
  }
};

dg::BlockForward Bind(FakeForward& fake) {
  return [&fake](std::span<const dg::BlockStep> s,
                 std::span<const dg::BlockCommit> c) { fake(s, c); };
}

}  // namespace

int main() {
  const auto c = SmallConfig();
  const auto d = Defaults();

  // A whole-canvas block converges in two steps and leaves a pending commit.
  {
    dg::BlockGeneration g;
    dg::BeginGeneration(g, c, d, 7, false);
    FakeForward fake{.answers = {{5, 6, 7, 8}}};
    dg::BlockGeneration* gens[] = {&g};
    const int cached[] = {10}, budget[] = {100};
    auto out = dg::RunBlocks(c, 1000, gens, cached, budget, Bind(fake));
    Check(out[0].tokens == std::vector<std::int32_t>{5, 6, 7, 8},
          "full block tokens");
    Check(!out[0].end_of_sequence && out[0].steps == 2, "converges in 2");
    Check(g.commit_pending && g.block == 1, "commit pending after full block");
    Check(fake.passes == 2 && fake.committed.empty(), "no commit yet");

    // The next block first commits the previous canvas, then denoises.
    fake.answers = {{9, 1, 3, 3}};
    const int cached2[] = {10};
    out = dg::RunBlocks(c, 1000, gens, cached2, budget, Bind(fake));
    Check(fake.committed == std::vector<std::size_t>{0}, "commit first");
    Check(fake.batch_sizes[2] == 0, "commit pass has no steps");
    Check(out[0].tokens == std::vector<std::int32_t>{9} &&
              out[0].end_of_sequence,
          "stops at end token");
    Check(!g.commit_pending, "no commit after end of sequence");
  }

  // The budget cuts a block; a cut block is not committed.
  {
    dg::BlockGeneration g;
    dg::BeginGeneration(g, c, d, 7, false);
    FakeForward fake{.answers = {{5, 6, 7, 8}}};
    dg::BlockGeneration* gens[] = {&g};
    const int cached[] = {0}, budget[] = {3};
    const auto out = dg::RunBlocks(c, 1000, gens, cached, budget, Bind(fake));
    Check(out[0].tokens.size() == 3, "budget cuts the block");
    Check(!g.commit_pending, "cut block is not committed");
  }

  // A block that does not fit the context is refused, with no pass.
  {
    dg::BlockGeneration g;
    dg::BeginGeneration(g, c, d, 7, false);
    FakeForward fake{.answers = {{5, 6, 7, 8}}};
    dg::BlockGeneration* gens[] = {&g};
    const int cached[] = {98}, budget[] = {100};
    const auto out = dg::RunBlocks(c, 100, gens, cached, budget, Bind(fake));
    Check(out[0].context_full && fake.passes == 0, "context full");
  }

  // Two generations share passes; cancellation stops one between passes.
  {
    dg::BlockGeneration a, b;
    dg::BeginGeneration(a, c, d, 1, false);
    dg::BeginGeneration(b, c, d, 2, false);
    FakeForward fake{.answers = {{5, 6, 7, 8}, {2, 3, 4, 5}}};
    dg::BlockGeneration* gens[] = {&a, &b};
    const int cached[] = {0, 0}, budget[] = {100, 100};
    auto out = dg::RunBlocks(c, 1000, gens, cached, budget, Bind(fake));
    Check(fake.batch_sizes == std::vector<std::size_t>{2, 2}, "batched steps");
    Check(out[0].tokens.size() == 4 && out[1].tokens.size() == 4,
          "both finish");

    fake.batch_sizes.clear();
    int polls = 0;
    out = dg::RunBlocks(c, 1000, gens, cached, budget, Bind(fake),
                        [&polls](std::size_t i) { return i == 1 && ++polls > 1; });
    Check(out[1].cancelled && !out[0].cancelled, "cancel one generation");
  }

  // Equal seeds give equal step seeds and canvases; different seeds differ.
  {
    std::vector<std::uint64_t> runs[3];
    std::vector<std::int32_t> canvases[3];
    const std::uint64_t seeds[3] = {42, 42, 43};
    for (int r = 0; r < 3; ++r) {
      dg::BlockGeneration g;
      dg::BeginGeneration(g, c, d, seeds[r], false);
      FakeForward fake{.answers = {{5, 6, 7, 8}}};
      dg::BlockGeneration* gens[] = {&g};
      const int cached[] = {0}, budget[] = {100};
      // Capture the random starting canvas through the first step.
      dg::BlockForward capture = [&](std::span<const dg::BlockStep> s,
                                     std::span<const dg::BlockCommit> cm) {
        if (fake.passes == 0)
          canvases[r].assign(s[0].tokens, s[0].tokens + c.canvas);
        fake(s, cm);
      };
      (void)dg::RunBlocks(c, 1000, gens, cached, budget, capture);
      runs[r] = fake.seeds;
    }
    Check(runs[0] == runs[1] && canvases[0] == canvases[1], "seeded replay");
    Check(runs[0] != runs[2] && canvases[0] != canvases[2],
          "different seeds differ");
  }

  // Fixed-length generation suppresses every end token.
  {
    dg::BlockGeneration g;
    dg::BeginGeneration(g, c, d, 1, true);
    Check(g.suppressed_count == 1 && g.suppressed[0] == 1, "suppress end");
  }

  if (failures == 0)
    std::puts("diffusion_gemma block: ok");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
