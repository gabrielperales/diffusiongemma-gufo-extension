// DiffusionGemma KV-cache bookkeeping: slot arithmetic of the sliding and
// global layers and prompt prefix reuse (CPU only).
#include "src/cache_state.hpp"

#include <cstdio>
#include <numeric>
#include <vector>

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

std::vector<std::int32_t> Tokens(int first, int count) {
  std::vector<std::int32_t> tokens(count);
  std::iota(tokens.begin(), tokens.end(), first);
  return tokens;
}

// Window 8 (7 cached positions attended) and 4 positions of reuse history:
// the sliding caches keep 11 positions.
dg::CacheState Make() {
  return dg::CacheState(/*sliding_window=*/8, /*reuse_history=*/4);
}

void Appends() {
  auto cache = Make();
  Expect(cache.sliding_keep() == 11, "keeps window - 1 + history");
  auto shift = cache.Append(Tokens(0, 5));
  Expect(cache.length() == 5 && cache.sliding_valid() == 5,
         "short pass is kept whole");
  Expect(shift.from == 0 && shift.count == 5, "short pass needs no shift");
  auto pass = cache.SlidingPass();
  Expect(pass.first_slot == 5 && pass.skip == 0 && pass.first_position == 0,
         "sliding pass after 5 positions");

  shift = cache.Append(Tokens(5, 10));
  Expect(cache.length() == 15 && cache.sliding_valid() == 11,
         "sliding caches keep 11 of 15");
  Expect(shift.from == 4 && shift.count == 11, "oldest 4 slots dropped");
  pass = cache.SlidingPass();
  // Positions 8..14 (the newest 7) are attended to; slots 0..3 hold
  // positions 4..7 as reuse history.
  Expect(pass.first_slot == 11 && pass.skip == 4 && pass.first_position == 8,
         "sliding pass skips the reuse history");
  const auto global = cache.GlobalPass();
  Expect(
      global.first_slot == 15 && global.skip == 0 && global.first_position == 0,
      "global pass sees every position");
  Expect(cache.tokens().size() == 15 && cache.tokens()[14] == 14,
         "tokens recorded in order");
}

void Reuse() {
  {
    auto cache = Make();
    Expect(cache.Reuse(Tokens(0, 3)) == 0, "empty cache reuses nothing");
  }
  {
    auto cache = Make();
    (void)cache.Append(Tokens(0, 15));
    auto prompt = Tokens(0, 20);
    Expect(cache.Reuse(prompt) == 15, "full prefix is reused");
    Expect(cache.length() == 15 && cache.sliding_valid() == 11,
           "full reuse keeps the cache");
  }
  {
    // Divergence at 12: the window before it (positions 5..11) is held.
    auto cache = Make();
    (void)cache.Append(Tokens(0, 15));
    auto prompt = Tokens(0, 20);
    prompt[12] = 100;
    Expect(cache.Reuse(prompt) == 12, "reuse up to the divergence");
    Expect(cache.length() == 12 && cache.sliding_valid() == 8,
           "cache truncated to the prefix");
    Expect(cache.tokens().size() == 12, "tokens truncated");
    const auto pass = cache.SlidingPass();
    Expect(pass.skip == 1 && pass.first_position == 5 && pass.first_slot == 8,
           "next pass attends to positions 5..11");
  }
  {
    // Divergence at 10: its window starts at position 3, which the sliding
    // caches (positions 4..14) no longer hold.
    auto cache = Make();
    (void)cache.Append(Tokens(0, 15));
    auto prompt = Tokens(0, 20);
    prompt[10] = 100;
    Expect(cache.Reuse(prompt) == 0, "divergence beyond the history");
    Expect(cache.length() == 0 && cache.sliding_valid() == 0 &&
               cache.tokens().empty(),
           "cache cleared");
  }
  {
    // A prompt that is a prefix of the cache.
    auto cache = Make();
    (void)cache.Append(Tokens(0, 15));
    Expect(cache.Reuse(Tokens(0, 13)) == 13, "prompt within the cache");
  }
  {
    auto cache = Make();
    (void)cache.Append(Tokens(0, 15));
    auto prompt = Tokens(0, 20);
    prompt[0] = 100;
    Expect(cache.Reuse(prompt) == 0, "different first token");
  }
}

// ReusableLength reports what Reuse would keep and leaves the cache alone.
void ReusableLength() {
  auto cache = Make();
  (void)cache.Append(Tokens(0, 15));
  auto within = Tokens(0, 20);
  within[12] = 100;
  auto beyond = Tokens(0, 20);
  beyond[10] = 100;
  Expect(cache.ReusableLength(Tokens(0, 20)) == 15, "full prefix reusable");
  Expect(cache.ReusableLength(within) == 12, "reusable up to the divergence");
  Expect(cache.ReusableLength(beyond) == 0, "nothing beyond the history");
  Expect(cache.length() == 15 && cache.sliding_valid() == 11 &&
             cache.tokens().size() == 15,
         "query leaves the cache unchanged");
  Expect(cache.Reuse(within) == 12, "Reuse agrees with the query");
}

// A restored frontier: the sliding layers hold the newest kept positions
// and continue exactly as after encoding that many tokens.
void RestoreLength() {
  auto restored = Make();
  restored.RestoreLength(20);
  auto encoded = Make();
  encoded.Append(Tokens(0, 20));
  Expect(restored.length() == 20 && restored.sliding_valid() == 11,
         "restored length and sliding rows");
  Expect(restored.SlidingPass().first_slot == encoded.SlidingPass().first_slot &&
             restored.SlidingPass().skip == encoded.SlidingPass().skip &&
             restored.SlidingPass().first_position ==
                 encoded.SlidingPass().first_position,
         "restored sliding pass");
  Expect(restored.GlobalPass().first_slot == 20, "restored global pass");
  Expect(restored.ReusableLength(Tokens(0, 25)) == 0,
         "restored tokens are unknown");
  auto short_one = Make();
  short_one.RestoreLength(5);
  Expect(short_one.sliding_valid() == 5, "short restore keeps every row");
}

}  // namespace

int main() {
  Appends();
  Reuse();
  ReusableLength();
  RestoreLength();
  if (failures == 0)
    std::printf("diffusion_gemma_cache_state_test: OK\n");
  return failures == 0 ? 0 : 1;
}
