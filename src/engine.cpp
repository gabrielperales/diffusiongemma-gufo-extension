#include "src/engine.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/denoise.hpp"
#include "src/hip_check.hpp"
#include "src/kernels.hpp"
#include "src/model.hpp"
#include "src/profiler.hpp"
#include "src/read.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

ModelOptions ToModelOptions(const EngineOptions& o) {
  return {.max_context = o.max_context,
          .prefill_chunk = o.prefill_chunk,
          .max_batch = std::max(1, o.sessions),
          .reuse_history = 4096,
          .stage_prompts = false,
          .profile = o.profile};
}

}  // namespace

/// Bubo's request scheduler over the shared Model: a worker thread admits
/// requests to free sessions and runs their prompt chunks and denoising steps
/// as batched forward passes. Kept for the standalone runner, which compares
/// this port with Bubo; Gufo's server uses the TextModelRunner instead.
struct Engine::Impl {
  // Per-block seed offset: an odd constant, so blocks of one request draw
  // from distinct streams.
  static constexpr std::uint64_t kBlockSeedStride = 0x1000193ULL;

  Model model;
  const Config& config;
  const DiffusionDefaults& defaults;
  EngineOptions options;
  std::vector<std::unique_ptr<Session>> sessions;
  std::uint64_t session_clock{0};

  Impl(const std::filesystem::path& root, const EngineOptions& o)
      : model(root, ToModelOptions(o)),
        config(model.config()),
        defaults(model.defaults()),
        options(o) {
    options.sessions = std::max(1, o.sessions);
    options.prefill_chunk = model.options().prefill_chunk;
    options.max_context = model.options().max_context;
    for (int i = 0; i < options.sessions; ++i)
      sessions.push_back(model.CreateSession());
    StartWorker();
  }

  ~Impl() { StopWorker(); }


  /// One request: the caller's inputs, the worker's progress through it,
  /// and the finished canvases waiting for the caller.
  struct Job {
    std::vector<std::int32_t> prompt;
    std::vector<ImageInput> images;
    std::vector<ImageSpan> image_identities;
    GenerationOptions options;  // `cancelled` stays with the caller.
    std::optional<ReadOptions> read;
    ReadResult read_result;
    GenerationStats stats;
    Clock::time_point start;
    std::atomic<bool> cancel{false};
    // Shared with the caller.
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<std::int32_t>> blocks;
    bool finished{false};
    std::exception_ptr error;
    // Worker only.
    enum class Phase { kPrefill, kDenoise, kDone } phase{Phase::kPrefill};
    bool commit_pending{false};  // A finished canvas waits to be encoded.
    Session* session{nullptr};
    std::size_t encoded{0};  // Prompt tokens in the session's cache.
    std::mt19937_64 rng;
    kernels::SuppressedTokens suppressed;
    int max_blocks{0};
    int block{0};
    // The canvas being denoised.
    std::uint64_t block_seed{0};
    int steps{0};
    Clock::time_point block_start;
    std::vector<std::int32_t> current, argmax_canvas, previous_argmax;
    std::vector<float> row_entropy;
    std::vector<std::int32_t> row_argmax, row_sample;
    std::vector<std::int32_t> read_seed;
    std::vector<std::uint8_t> pins;
    ScoreHost scores;
  };

  /// Prompt chunk size while other requests are denoising, so that they
  /// wait for at most one such chunk between steps.
  static constexpr int kInterleavedChunk = 2048;

  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<std::shared_ptr<Job>> queue;
  bool stopping{false};
  std::vector<std::shared_ptr<Job>> active;  // Worker only.
  // Held by the worker around GPU work and by the benchmark primitives.
  std::mutex gpu_mutex;
  std::thread worker;

  void CancelAndWait(const std::shared_ptr<Job>& job) {
    job->cancel = true;
    queue_ready.notify_all();
    std::unique_lock<std::mutex> lock(job->mutex);
    job->ready.wait(lock, [&] { return job->finished; });
  }

  void StartWorker() {
    worker = std::thread([this] { Work(); });
  }

  void StopWorker() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    queue_ready.notify_all();
    if (worker.joinable())
      worker.join();
  }

  /// The free session for `job`: the one that reuses the longest prompt
  /// prefix, else the least recently used; null when all are busy.
  Session* PickSession(const Job& job) {
    Session* best = nullptr;
    int best_reuse = -1;
    for (auto& owned : sessions) {
      Session& session = *owned;
      if (session.busy)
        continue;
      const int reuse =
          job.options.reuse_prompt
              ? session.cache.ReusableLength(job.prompt, job.image_identities)
              : 0;
      if (reuse > best_reuse ||
          (reuse == best_reuse && session.last_used < best->last_used)) {
        best = &session;
        best_reuse = reuse;
      }
    }
    return best;
  }

  void Admit(Job& job, Session& session) {
    job.session = &session;
    session.busy = true;
    session.last_used = ++session_clock;
    if (job.options.reuse_prompt) {
      job.stats.reused_tokens =
          session.cache.Reuse(job.prompt, job.image_identities);
    } else {
      session.cache.Clear();
      job.stats.reused_tokens = 0;
    }
    job.encoded = static_cast<std::size_t>(job.stats.reused_tokens);
  }

  void Work() {
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_ready.wait(lock, [&] {
          return stopping || !active.empty() ||
                 (!queue.empty() && PickSession(*queue.front()) != nullptr);
        });
        if (stopping)
          break;
        while (!queue.empty()) {
          Session* session = PickSession(*queue.front());
          if (session == nullptr)
            break;
          auto job = std::move(queue.front());
          queue.pop_front();
          Admit(*job, *session);
          active.push_back(std::move(job));
        }
      }
      std::lock_guard<std::mutex> gpu(gpu_mutex);
      try {
        Advance();
      } catch (...) {
        const auto error = std::current_exception();
        for (auto& job : active) {
          job->error = error;
          Finish(*job);
        }
      }
      std::erase_if(active, [](const auto& job) {
        return job->phase == Job::Phase::kDone;
      });
    }
    // Shutting down: nothing more runs.
    std::lock_guard<std::mutex> lock(queue_mutex);
    const auto error = std::make_exception_ptr(
        std::runtime_error("DiffusionGemma engine stopped"));
    for (auto& job : active) {
      job->error = error;
      Finish(*job);
    }
    for (auto& job : queue) {
      job->error = error;
      Finish(*job);
    }
  }

  /// One forward pass over every request's pending work: the canvases being
  /// denoised (one step each), then finished canvases to commit and prompt
  /// chunks (oldest request first), within the pass's row budget.
  void Advance() {
    for (auto& job : active)
      if (job->cancel.load() && job->phase != Job::Phase::kDone) {
        job->stats.cancelled = true;
        Finish(*job);
      }
    const Config& c = config;
    std::vector<Job*> stepping, committing, prefilling;
    std::vector<int> prefill_rows;
    for (auto& job : active)
      if (job->phase == Job::Phase::kDenoise && !job->commit_pending)
        stepping.push_back(job.get());
    // Prompt rows this pass may add: a whole chunk on its own, else
    // at most kInterleavedChunk so that denoising requests are not held up.
    int budget = stepping.empty()
                     ? options.prefill_chunk
                     : std::min(options.prefill_chunk, kInterleavedChunk);
    budget =
        std::min(budget, model.rows() - c.canvas * static_cast<int>(stepping.size()));
    std::vector<EncodeWork> encodes;
    for (auto& job : active)
      if (job->phase == Job::Phase::kDenoise && job->commit_pending &&
          budget >= c.canvas) {
        encodes.push_back(EncodeWork{.session = job->session,
                                     .tokens = job->argmax_canvas.data(),
                                     .rows = c.canvas});
        committing.push_back(job.get());
        budget -= c.canvas;
      }
    bool staged = false;
    for (auto& job : active) {
      if (job->phase != Job::Phase::kPrefill)
        continue;
      const int remaining = static_cast<int>(job->prompt.size() - job->encoded);
      const PromptChunk chunk =
          job->images.empty()
              ? PromptChunk{std::min(remaining, budget), -1}
              : PlanPromptChunk(job->prompt.size(), job->encoded, budget,
                                options.prefill_chunk, options.max_context,
                                job->image_identities);
      const int take = chunk.rows;
      if (remaining > 0 && take <= 0)
        break;
      // Only one chunk per pass can use the stage caches.
      const bool needs_stage =
          job->session->cache.sliding_valid() + take > model.sliding_capacity();
      if (take > 0 && needs_stage && staged)
        continue;
      staged = staged || (take > 0 && needs_stage);
      if (take > 0)
        encodes.push_back(EncodeWork{
            .session = job->session,
            .tokens = job->prompt.data() + job->encoded,
            .rows = take,
            .image = chunk.image >= 0 ? &job->images[chunk.image] : nullptr});
      prefilling.push_back(job.get());
      prefill_rows.push_back(take);
      budget -= take;
    }
    std::vector<CanvasStep> batch;
    for (Job* job : stepping) {
      const DiffusionDefaults& d = job->options.diffusion;
      const ReadStepPlan plan =
          job->read ? PlanReadStep(job->steps, job->read->max_steps)
                    : ReadStepPlan{};
      // Each step draws its Gumbel noise from its own stream.
      batch.push_back(CanvasStep{
          .session = job->session,
          .tokens = job->current.data(),
          .self_condition = job->read ? plan.self_condition : job->steps > 0,
          .make_probabilities = !job->read || plan.make_probabilities,
          .has_pins = job->read && !job->read->pinned_positions.empty(),
          .temperature = StepTemperature(d, d.max_denoising_steps - job->steps),
          .seed = MixSeed(job->block_seed ^
                          (static_cast<std::uint64_t>(job->steps) << 40)),
          .suppressed = job->suppressed,
          .entropy = job->row_entropy.data(),
          .argmax = job->row_argmax.data(),
          .sample = job->row_sample.data(),
          .scores = job->read && plan.finish ? &job->scores : nullptr,
          .label_ids = job->read ? job->read->label_ids.data() : nullptr,
          .label_count =
              job->read ? static_cast<int>(job->read->label_ids.size()) : 0});
    }
    const auto start = Clock::now();
    if (!batch.empty() || !encodes.empty())
      model.Forward(batch, encodes);
    const double ms = Ms(start, Clock::now());
    for (std::size_t i = 0; i < prefilling.size(); ++i) {
      Job& job = *prefilling[i];
      job.encoded += prefill_rows[i];
      job.stats.prefill_ms += ms;
      if (job.encoded == job.prompt.size())
        StartGeneration(job);
    }
    for (Job* job : committing) {
      job->stats.commit_ms += ms;
      job->commit_pending = false;
      ++job->block;
      if (!StartBlock(*job))
        Finish(*job);
    }
    std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
    for (Job* job : stepping) {
      const DiffusionDefaults& d = job->options.diffusion;
      ++job->steps;
      if (job->read) {
        job->argmax_canvas = job->row_argmax;
        if (job->steps >= job->read->max_steps) {
          job->stats.denoise_ms += Ms(job->block_start, Clock::now());
          job->stats.denoising_steps = job->steps;
          job->stats.blocks = 1;
          job->stats.first_block_ms = Ms(job->start, Clock::now());
          job->read_result.stats = job->stats;
          job->read_result.rows.reserve(c.canvas);
          const int top_k = job->read->top_k;
          const int labels = static_cast<int>(job->read->label_ids.size());
          for (int row = 0; row < c.canvas; ++row) {
            ReadRow result;
            result.token = job->scores.argmax[row];
            result.logprob = job->scores.top_logprobs[row * 20];
            for (int rank = 0; rank < top_k; ++rank)
              result.top_logprobs.push_back(
                  {job->scores.top_ids[row * 20 + rank],
                   job->scores.top_logprobs[row * 20 + rank]});
            for (int label = 0; label < labels; ++label) {
              const int id = job->read->label_ids[label];
              if (std::none_of(result.top_logprobs.begin(),
                               result.top_logprobs.end(),
                               [id](const ReadCandidate& top) {
                                 return top.token == id;
                               }))
                result.label_logprobs.push_back(
                    {id, job->scores.label_logprobs[row * labels + label]});
            }
            job->read_result.rows.push_back(std::move(result));
          }
          Finish(*job);
          continue;
        }
        const auto accepted = AcceptRows(job->row_entropy, d.entropy_bound);
        std::vector<std::uint8_t> mask(c.canvas);
        for (int i = 0; i < c.canvas; ++i)
          mask[i] = accepted[i];
        AdvanceReadCanvas(job->current, job->read_seed, job->pins, mask,
                          job->row_sample, [&] { return uniform(job->rng); });
        continue;
      }
      const auto accepted = AcceptRows(job->row_entropy, d.entropy_bound);
      for (int i = 0; i < c.canvas; ++i)
        job->current[i] = accepted[i] ? job->row_sample[i] : uniform(job->rng);
      job->argmax_canvas = job->row_argmax;
      const bool done = BlockConverged(d, job->previous_argmax, job->row_argmax,
                                       job->row_entropy);
      job->previous_argmax = job->row_argmax;
      if (done || job->steps >= d.max_denoising_steps)
        FinishBlock(*job);
    }
  }

  /// The prompt is in the cache: set up the sampler and the first canvas.
  void StartGeneration(Job& job) {
    model.profiler().Report("prefill");
    const Config& c = config;
    job.rng.seed(MixSeed(job.options.seed));
    if (job.options.suppress_end)
      for (const int id : c.eos_ids)
        job.suppressed.ids[job.suppressed.count++] = id;
    job.max_blocks =
        std::max(1, (job.options.max_new_tokens + c.canvas - 1) / c.canvas);
    job.block = 0;
    if (StartBlock(job)) {
      job.phase = Job::Phase::kDenoise;
      if (job.read) {
        if (!job.read->seed_canvas.empty())
          job.current = job.read->seed_canvas;
        job.read_seed = job.current;
        job.pins.assign(c.canvas, 0);
        for (const int position : job.read->pinned_positions)
          job.pins[position] = 1;
        if (!job.read->pinned_positions.empty())
          HipCheck(
              hipMemcpyAsync(job.session->pinned.as<void>(), job.pins.data(),
                             c.canvas, hipMemcpyHostToDevice, model.stream()),
              "read pins upload");
        job.scores.top_ids.resize(std::size_t(c.canvas) * 20);
        job.scores.top_logprobs.resize(std::size_t(c.canvas) * 20);
        job.scores.label_logprobs.resize(std::size_t(c.canvas) *
                                         job.read->label_ids.size());
        job.scores.logsumexp.resize(c.canvas);
        job.scores.argmax.resize(c.canvas);
      }
    } else
      Finish(job);
  }

  /// Starts the job's next canvas; false when the context is full.
  bool StartBlock(Job& job) {
    const Config& c = config;
    if (job.session->cache.length() + c.canvas > options.max_context)
      return false;
    job.block_seed =
        MixSeed(job.options.seed + kBlockSeedStride * (job.block + 1));
    // The canvas starts as random tokens; rows not accepted at a step are
    // drawn again.
    std::uniform_int_distribution<std::int32_t> uniform(0, c.vocab - 1);
    job.current.resize(c.canvas);
    for (auto& token : job.current)
      token = uniform(job.rng);
    job.row_entropy.resize(c.canvas);
    job.row_argmax.resize(c.canvas);
    job.row_sample.resize(c.canvas);
    job.previous_argmax.clear();
    job.steps = 0;
    job.block_start = Clock::now();
    return true;
  }

  void FinishBlock(Job& job) {
    const Config& c = config;
    GenerationStats& stats = job.stats;
    stats.denoise_ms += Ms(job.block_start, Clock::now());
    stats.denoising_steps += job.steps;
    ++stats.blocks;
    const auto reply =
        TakeReply(job.argmax_canvas, c.eos_ids,
                  job.options.max_new_tokens - stats.generated_tokens);
    stats.stopped_by_eos = reply.end_of_sequence;
    stats.generated_tokens += reply.tokens;
    if (job.block == 0)
      stats.first_block_ms = Ms(job.start, Clock::now());
    {
      std::lock_guard<std::mutex> lock(job.mutex);
      job.blocks.emplace_back(job.argmax_canvas.begin(),
                              job.argmax_canvas.begin() + reply.tokens);
    }
    job.ready.notify_all();
    if (stats.stopped_by_eos ||
        stats.generated_tokens >= job.options.max_new_tokens ||
        job.block + 1 == job.max_blocks || job.cancel.load()) {
      Finish(job);
      return;
    }
    // Reached only when the whole canvas belongs to the reply: it is
    // encoded into the cache in the next pass, then the next canvas starts.
    job.commit_pending = true;
  }

  void Finish(Job& job) {
    if (job.phase == Job::Phase::kDone)
      return;
    job.phase = Job::Phase::kDone;
    if (job.session != nullptr) {
      job.session->busy = false;
      job.session->last_used = ++session_clock;
    }
    model.ReportExperts();
    model.profiler().Report("denoise and commit");
    {
      std::lock_guard<std::mutex> lock(job.mutex);
      job.finished = true;
    }
    job.ready.notify_all();
    // A freed session may admit a queued request.
    queue_ready.notify_all();
  }

  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           std::span<const ImageInput> images,
                           const GenerationOptions& request,
                           const BlockCallback& on_block) {
    const Config& c = config;
    if (prompt.empty())
      throw std::invalid_argument("empty prompt");
    if (request.suppress_end &&
        c.eos_ids.size() > kernels::SuppressedTokens::kMax)
      throw std::invalid_argument("too many end-of-sequence tokens");
    auto job = std::make_shared<Job>();
    job->prompt.assign(prompt.begin(), prompt.end());
    job->images.assign(images.begin(), images.end());
    job->image_identities.reserve(images.size());
    for (const auto& image : images) {
      if (!image.embedding)
        throw std::invalid_argument("image span has no encoder output");
      job->image_identities.push_back(image);
    }
    ValidateImagePrompt(prompt, options.prefill_chunk, options.max_context,
                        c.canvas, job->image_identities);
    job->options = request;
    job->options.cancelled = nullptr;
    job->stats.prompt_tokens = static_cast<int>(prompt.size());
    job->start = Clock::now();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (stopping)
        throw std::runtime_error("DiffusionGemma engine stopped");
      queue.push_back(job);
    }
    queue_ready.notify_all();
    bool consumer_stopped = false;
    try {
      for (;;) {
        std::deque<std::vector<std::int32_t>> ready;
        bool finished = false;
        {
          std::unique_lock<std::mutex> lock(job->mutex);
          job->ready.wait_for(lock, std::chrono::milliseconds(20), [&] {
            return !job->blocks.empty() || job->finished;
          });
          ready.swap(job->blocks);
          finished = job->finished;
        }
        if (request.cancelled && request.cancelled())
          job->cancel = true;
        for (const auto& block : ready) {
          if (consumer_stopped)
            break;
          if (on_block && !on_block(block)) {
            consumer_stopped = true;
            job->cancel = true;
          }
        }
        if (finished)
          break;
      }
    } catch (...) {
      CancelAndWait(job);
      throw;
    }
    if (job->error)
      std::rethrow_exception(job->error);
    GenerationStats stats = job->stats;
    stats.cancelled = stats.cancelled || consumer_stopped;
    return stats;
  }

  ReadResult Read(std::span<const std::int32_t> prompt,
                  std::span<const ImageInput> images,
                  const ReadOptions& request) {
    const Config& c = config;
    ValidateReadOptions(request, c.canvas, c.vocab);
    if (prompt.empty())
      throw std::invalid_argument("empty prompt");
    auto job = std::make_shared<Job>();
    job->prompt.assign(prompt.begin(), prompt.end());
    job->images.assign(images.begin(), images.end());
    job->image_identities.reserve(images.size());
    for (const auto& image : images) {
      if (!image.embedding)
        throw std::invalid_argument("image span has no encoder output");
      job->image_identities.push_back(image);
    }
    ValidateImagePrompt(prompt, options.prefill_chunk, options.max_context,
                        c.canvas, job->image_identities);
    job->options.diffusion = defaults;
    job->options.seed = request.seed;
    job->options.reuse_prompt = request.reuse_prompt;
    job->read = request;
    job->read->cancelled = nullptr;
    job->stats.prompt_tokens = static_cast<int>(prompt.size());
    job->start = Clock::now();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (stopping)
        throw std::runtime_error("DiffusionGemma engine stopped");
      queue.push_back(job);
    }
    queue_ready.notify_all();
    try {
      for (;;) {
        bool finished = false;
        {
          std::unique_lock<std::mutex> lock(job->mutex);
          job->ready.wait_for(lock, std::chrono::milliseconds(20),
                              [&] { return job->finished; });
          finished = job->finished;
        }
        if (request.cancelled && request.cancelled())
          job->cancel = true;
        if (finished)
          break;
      }
    } catch (...) {
      CancelAndWait(job);
      throw;
    }
    if (job->error)
      std::rethrow_exception(job->error);
    if (job->stats.cancelled)
      throw std::runtime_error("diffusion read cancelled");
    return std::move(job->read_result);
  }
};

Engine::Engine(const std::filesystem::path& root, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(root, options)) {}
Engine::~Engine() = default;

const Config& Engine::config() const {
  return impl_->config;
}
const Tokenizer& Engine::tokenizer() const {
  return impl_->model.tokenizer();
}
const DiffusionDefaults& Engine::diffusion_defaults() const {
  return impl_->defaults;
}
int Engine::max_context() const {
  return impl_->options.max_context;
}
int Engine::prefill_chunk() const {
  return impl_->options.prefill_chunk;
}
std::size_t Engine::weight_bytes() const {
  return impl_->model.weight_bytes();
}

GenerationStats Engine::Generate(std::span<const std::int32_t> prompt,
                                 const GenerationOptions& options,
                                 const BlockCallback& on_block) {
  return impl_->Generate(prompt, {}, options, on_block);
}

GenerationStats Engine::Generate(std::span<const std::int32_t> prompt,
                                 std::span<const ImageInput> images,
                                 const GenerationOptions& options,
                                 const BlockCallback& on_block) {
  return impl_->Generate(prompt, images, options, on_block);
}

ReadResult Engine::Read(std::span<const std::int32_t> prompt,
                        const ReadOptions& options) {
  return impl_->Read(prompt, {}, options);
}

ReadResult Engine::Read(std::span<const std::int32_t> prompt,
                        std::span<const ImageInput> images,
                        const ReadOptions& options) {
  return impl_->Read(prompt, images, options);
}

double Engine::Warmup() {
  auto& impl = *impl_;
  const auto start = Clock::now();
  const int filler = impl.model.tokenizer().TokenId("\xE2\x96\x81the");
  // A context below the prefill chunk (--context 4096 with the default 8192
  // chunk) must still load: the warm-up chunk fits the context.
  std::vector<std::int32_t> tokens(
      std::min(impl.options.prefill_chunk, impl.options.max_context),
      filler >= 0 ? filler : 1);
  tokens[0] = impl.config.bos_id;
  // An encoder chunk, then one short request per session: their two
  // denoising steps (with and without self-conditioning) and commit load
  // every kernel, at every batch size up to all sessions, before the first
  // request.
  Reset();
  (void)Encode(tokens);
  GenerationOptions options;
  options.diffusion = impl.defaults;
  options.diffusion.max_denoising_steps = 2;
  options.max_new_tokens = 2 * impl.config.canvas;
  std::vector<std::thread> requests;
  for (int i = 0; i < impl.options.sessions; ++i)
    requests.emplace_back([&impl, &tokens, &options, i] {
      // Distinct prompts, so that each takes its own session.
      std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 64);
      prompt[1] = impl.config.bos_id + 1 + i;
      (void)impl.Generate(prompt, {}, options, {});
    });
  for (auto& request : requests)
    request.join();
  Reset();
  return Ms(start, Clock::now());
}

void Engine::Reset() {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  for (auto& session : impl_->sessions)
    session->cache.Clear();
}

double Engine::Encode(std::span<const std::int32_t> tokens) {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  const double ms = impl_->model.Encode(*impl_->sessions[0], tokens);
  impl_->model.profiler().Report();
  return ms;
}

int Engine::cached_tokens() const {
  std::lock_guard<std::mutex> gpu(impl_->gpu_mutex);
  return impl_->sessions[0]->cache.length();
}

std::vector<float> Engine::DebugLogits(
    std::span<const std::int32_t> prompt,
    const std::vector<std::int32_t>& canvas,
    const std::vector<std::int32_t>& canvas2) {
  Reset();
  auto& impl = *impl_;
  std::lock_guard<std::mutex> gpu(impl.gpu_mutex);
  auto& session = *impl.sessions[0];
  const std::size_t rows = impl.config.canvas, vocab = impl.config.vocab;
  (void)impl.model.Encode(session, prompt);
  std::vector<float> out(2 * rows * vocab);
  std::vector<float> e(rows);
  std::vector<std::int32_t> a(rows), b(rows);
  const float cap = impl.config.final_softcap;
  for (int step = 0; step < 2; ++step) {
    const CanvasStep canvas_step{
        .session = &session,
        .tokens = (step == 0 ? canvas : canvas2).data(),
        .self_condition = step == 1,
        .temperature = 1.0F,
        .seed = static_cast<std::uint64_t>(step + 1),
        .suppressed = {},
        .entropy = e.data(),
        .argmax = a.data(),
        .sample = b.data()};
    impl.model.Forward(std::span<const CanvasStep>(&canvas_step, 1), {});
    float* dst = out.data() + step * rows * vocab;
    HipCheck(hipMemcpy(dst, impl.model.last_logits(), rows * vocab * 4,
                       hipMemcpyDeviceToHost),
             "debug readback");
    for (std::size_t i = 0; i < rows * vocab; ++i)
      dst[i] = cap * std::tanh(dst[i] / cap);
  }
  return out;
}

}  // namespace gufo::models::diffusion_gemma
