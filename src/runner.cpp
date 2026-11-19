// DiffusionGemma as a Gufo text runner.
//
// Gufo's runner contract is autoregressive: prefill, then decode steps. Here
// a decode step is a whole diffusion block. The runner declares
// multi_token_decode, and one DecodeStep denoises a canvas (up to the
// checkpoint's step limit), returns its reply tokens and leaves the canvas
// to be committed into the cache before the next block. DecodeBatch batches
// the denoising steps of concurrent sessions into shared forward passes.
#include "src/runner.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <deque>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/sha256.hpp>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/block.hpp"
#include "src/chat.hpp"
#include "src/continuation.hpp"
#include "src/denoise.hpp"
#include "src/model.hpp"
#include "src/tool_dialect.hpp"

namespace gufo::models::diffusion_gemma {
namespace {
using namespace gufo::ext;

constexpr const char* kModelId = "diffusiongemma-26b-a4b-it";
constexpr const char* kStateAbi = "diffusion-gemma-bf16-v1";

class State final : public TextRunnerState {
public:
  State(Session& session, cache::Adapter& adapter, cache::Slot& slot)
      : session(&session), adapter_(&adapter), slot_(&slot) {}

  Session* session;
  BlockGeneration generation;
  bool begun{false};
  /// Tokens of a block handed out one at a time through SelectNext.
  std::deque<std::int32_t> pending;
  bool pending_stop{false};
  CancellationCheck cancelled;

  void SetCancellationCheck(const CancellationCheck& check) override {
    cancelled = check;
  }
  /// A new prompt or a moved frontier ends the reply in progress.
  void ResetReply() noexcept {
    begun = false;
    pending.clear();
    pending_stop = false;
  }
  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    ResetReply();
  }

private:
  cache::Adapter* adapter_;
  cache::Slot* slot_;
};

State& Get(TextRunnerState& state) {
  return dynamic_cast<State&>(state);
}
const State& Get(const TextRunnerState& state) {
  return dynamic_cast<const State&>(state);
}

std::vector<std::int32_t> ToIds(std::span<const TextRunnerToken> tokens) {
  return {tokens.begin(), tokens.end()};
}

/// OpenAI-style chat request to the checkpoint's chat template (as Bubo's
/// HTTP backend converts it).
std::string RenderRequest(const ChatRequest& request) {
  std::vector<Message> messages;
  for (const auto& message : request.messages) {
    Message out;
    switch (message.role) {
      case ChatRole::kSystem:
        out.role = "system";
        break;
      case ChatRole::kDeveloper:
        out.role = "developer";
        break;
      case ChatRole::kAssistant:
        out.role = "assistant";
        break;
      case ChatRole::kTool:
        out.role = "tool";
        break;
      default:
        out.role = "user";
        break;
    }
    out.content = message.content;
    // Server-authored instructions that follow system or developer content.
    out.content += message.framing_suffix;
    out.tool_call_id = message.tool_call_id;
    out.name = message.name;
    for (const auto& call : message.tool_calls) {
      ToolCall converted;
      converted.id = call.id;
      converted.name = call.name;
      for (const auto& argument : call.arguments) {
        json::Value value(argument.value);
        if (!argument.is_string) {
          try {
            value = json::parse(argument.value);
          } catch (const std::exception&) {
          }
        }
        converted.arguments[argument.name] = std::move(value);
      }
      out.tool_calls.push_back(std::move(converted));
    }
    messages.push_back(std::move(out));
  }
  std::vector<json::Value> tools;
  if (request.tool_choice != ChatRequest::ToolChoice::kNone)
    for (const auto& tool : request.tools) {
      json::Value definition;
      try {
        definition = json::parse(tool.definition_json);
      } catch (const std::exception&) {
      }
      if (!definition.is_object() || definition.find("function") == nullptr) {
        json::Value function = json::Value::object();
        function["name"] = tool.name;
        function["description"] = tool.description;
        function["parameters"] = json::parse(tool.parameters_json);
        definition = json::Value::object();
        definition["type"] = "function";
        definition["function"] = std::move(function);
      }
      tools.push_back(std::move(definition));
    }
  return RenderChat(messages, tools, request.reasoning.enabled.value_or(false));
}

class Runner final : public HipTextModelRunner {
public:
  Runner(std::shared_ptr<Model> model, std::uint32_t context,
         std::string artifact_identity)
      : model_(std::move(model)), context_(context) {
    std::ostringstream identity;
    identity << "schema=gufo-text-continuation-v2\nmodel_kind=diffusion-gemma"
             << "\nartifact=" << artifact_identity << "\nstate_abi="
             << kStateAbi << "\ntokenizer=gemma4-tokenizer-json-v1"
             << "\nchat_template=diffusion-gemma-native-v1"
             << "\ncontext_tokens=" << context_
             << "\nsliding_keep=" << model_->config().sliding_window - 1
             << "\nadapters=none\n";
    const auto text = identity.str();
    component_identity_.assign(text.begin(), text.end());
  }

  // ------------------------------------------------------------ contract

  TextRunnerDescriptor Descriptor() const override {
    const auto width = static_cast<std::size_t>(model_->options().max_batch);
    return {
        .model_id = kModelId,
        .state_abi = kStateAbi,
        .max_context = context_,
        .capabilities = {.supports_vision = false,
                         .incremental_prefill = true,
                         // A block's reply tokens are never fed back one by
                         // one: the canvas is committed whole.
                         .final_token_advance_required = false,
                         .incremental_text_is_exact = false,
                         .multi_token_decode = true,
                         .batched_multi_token_decode = width > 1,
                         .batched_multi_token_decode_max_width = width,
                         .prefix_reuse = true,
                         .partial_prefill_retention = true},
        // The diffusion sampler ignores temperature, top-k/p and penalties.
        .sampling_defaults = std::nullopt,
        // A prefill pass is one prompt chunk, well under half a second, so
        // it runs unsplit beside decoders.
        .scheduling = TextSchedulingProfile{
            .pass_tokens =
                static_cast<std::size_t>(model_->options().prefill_chunk),
            .max_layer_groups = 1,
            .max_yield_decode_width = width}};
  }

  TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 0,
            .per_request_state_bytes = 0,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
            .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes()};
  }

  std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{TextExecutionPlanKind::kSerial, 1}};
    for (int width = 2; width <= model_->options().max_batch; ++width)
      plans.push_back(
          {TextExecutionPlanKind::kBatched, static_cast<std::size_t>(width)});
    return plans;
  }

  // The host binds a tool-call grammar to every request with tools, so the
  // runner offers a constraint vocabulary. The diffusion sampler cannot
  // follow a grammar token by token: calls are read back from the text by
  // the dialect, as in Bubo. Structured output, which would need the
  // grammar, is refused in PreparePrompt.
  std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    const auto& c = model_->config();
    return std::make_shared<const sampling::ConstraintVocabulary>(
        VocabularySize(), [this, &c](std::uint32_t id) {
          const bool stop = std::find(c.eos_ids.begin(), c.eos_ids.end(),
                                      static_cast<int>(id)) != c.eos_ids.end();
          return sampling::ConstraintVocabulary::Piece{Piece(id), stop};
        });
  }

  std::string ToolDialectId() const override {
    return std::string(kToolDialectId);
  }

  std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (request.response_format)
      throw std::invalid_argument("model does not support structured output");
    return TextModelRunner::PreparePrompt(request);
  }

  ReasoningTags ReasoningBudgetTags() const override {
    return {"<|channel>", "<channel|>", {"<channel|>"}};
  }
  ReasoningMarkup ReasoningDelimiters() const override {
    // The channel's name is not part of the reasoning.
    return {"<|channel>thought\n", "<channel|>"};
  }

  std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->config().vocab);
  }

  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    const auto ids = model_->tokenizer().Encode(text);
    return {ids.begin(), ids.end()};
  }

  std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderRequest(request);
  }

  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return Tokenize(RenderRequest(request));
  }

  ext::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    // The model may open its thinking channel itself; the host then returns
    // it as reasoning rather than content.
    return ext::InitialOutputState::kAuto;
  }

  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    const auto ids = ToIds(tokens);
    return model_->tokenizer().Decode(ids, /*skip_special=*/false);
  }

  // ------------------------------------------------------- continuation

  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    return hip::CreateHipComponentCacheResources(
        std::make_unique<ContinuationAdapter>(model_, context_,
                                              component_identity_),
        ledger,
        {[] { return HostSnapshotCeilingBytes(); }, [](std::string_view) {},
         [](std::string_view) {}});
  }

  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& continuation = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<State>(continuation.GetSession(slot), adapter,
                                   slot);
  }

  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t) const override {
    Get(state).ResetReply();
  }

  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return static_cast<std::size_t>(Get(state).session->cache.length());
  }

  // -------------------------------------------------------------- prefill

  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t max_input_tokens) const override {
    auto& s = Get(state);
    if (offset != static_cast<std::size_t>(s.session->cache.length()) ||
        offset >= prompt.size())
      throw std::logic_error("DiffusionGemma prefill frontier mismatch");
    const auto count = std::min<std::size_t>(
        {static_cast<std::size_t>(model_->options().prefill_chunk),
         max_input_tokens, prompt.size() - offset});
    if (count == 0)
      throw std::invalid_argument("DiffusionGemma prefill budget is zero");
    // A reply needs room for at least one token after the prompt.
    if (offset + count >= context_)
      throw std::length_error(
          "prompt of " + std::to_string(prompt.size()) +
          " tokens leaves no room for a reply in the context of " +
          std::to_string(context_) + " tokens");
    s.ResetReply();
    const auto ids = ToIds(prompt.subspan(offset, count));
    const EncodeWork work{.session = s.session,
                          .tokens = ids.data(),
                          .rows = static_cast<int>(count)};
    {
      const std::lock_guard lock(gpu_mutex_);
      model_->Forward({}, std::span<const EncodeWork>(&work, 1));
    }
    return {.consumed_tokens = count,
            .decode_ready = offset + count == prompt.size()};
  }

  // --------------------------------------------------------------- decode

  TextDecodeStep DecodeStep(TextRunnerState& state, std::size_t max_tokens,
                            sampling::SamplerState& sampler) const override {
    State* states[] = {&Get(state)};
    const std::size_t budgets[] = {max_tokens};
    sampling::SamplerState* samplers[] = {&sampler};
    return std::move(RunDecode(states, budgets, samplers).front());
  }

  std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    std::vector<TextDecodeStep> steps;
    steps.reserve(decodes.size());
    const auto width = static_cast<std::size_t>(model_->options().max_batch);
    for (std::size_t first = 0; first < decodes.size(); first += width) {
      const auto end = std::min(decodes.size(), first + width);
      std::vector<State*> states;
      std::vector<std::size_t> budgets;
      std::vector<sampling::SamplerState*> samplers;
      for (std::size_t i = first; i < end; ++i) {
        states.push_back(&Get(decodes[i].state.get()));
        budgets.push_back(decodes[i].max_tokens);
        samplers.push_back(&decodes[i].sampler.get());
      }
      try {
        for (auto& step : RunDecode(states, budgets, samplers))
          steps.push_back(std::move(step));
      } catch (...) {
        const auto failure = std::current_exception();
        for (std::size_t i = first; i < end; ++i) {
          TextDecodeStep failed;
          failed.failure = failure;
          steps.push_back(std::move(failed));
        }
      }
    }
    return steps;
  }

  // The host decodes through DecodeStep (multi_token_decode). SelectNext
  // and Advance serve callers that take one token at a time: a block is
  // generated whole and handed out token by token.
  TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = Get(state);
    if (s.pending.empty() && !s.pending_stop) {
      auto step = DecodeStep(state, static_cast<std::size_t>(INT_MAX),
                             sampler);
      for (const auto& selection : step.selections)
        s.pending.push_back(static_cast<std::int32_t>(selection.token));
      s.pending_stop = step.stop;
    }
    if (s.pending.empty())
      return {.stop = true};
    const auto token = static_cast<TextRunnerToken>(s.pending.front());
    s.pending.pop_front();
    return {.stop = false, .token = token, .piece = Piece(token)};
  }

  void Advance(TextRunnerState&, TextRunnerToken) const override {
    // SelectNext already advanced the state by the whole block.
  }

private:
  std::string Piece(TextRunnerToken token) const {
    const std::int32_t id = static_cast<std::int32_t>(token);
    return model_->tokenizer().Decode(std::span<const std::int32_t>(&id, 1),
                                      /*skip_special=*/false);
  }

  std::uint64_t ReplySeed(sampling::SamplerState& sampler) const {
    const auto seed = sampler.config().seed;
    if (seed >= 0)
      return static_cast<std::uint64_t>(seed);
    // Draw from the request's random stream and move it on.
    const auto drawn = MixSeed(sampler.rng_state());
    sampler.SetRngState(drawn);
    return drawn;
  }

  /// One block for each state, their denoising steps batched.
  std::vector<TextDecodeStep> RunDecode(
      std::span<State* const> states, std::span<const std::size_t> max_tokens,
      std::span<sampling::SamplerState* const> samplers) const {
    const auto& c = model_->config();
    const std::size_t n = states.size();
    std::vector<TextDecodeStep> steps(n);

    // States still handing out a block through SelectNext answer from it.
    std::vector<std::size_t> live;
    for (std::size_t i = 0; i < n; ++i) {
      auto& s = *states[i];
      if (!s.pending.empty() || s.pending_stop) {
        while (!s.pending.empty() && steps[i].selections.size() < max_tokens[i]) {
          const auto token = static_cast<TextRunnerToken>(s.pending.front());
          s.pending.pop_front();
          steps[i].selections.push_back(
              {.stop = false, .token = token, .piece = Piece(token)});
        }
        steps[i].stop = s.pending.empty() && s.pending_stop;
        continue;
      }
      if (!s.begun) {
        BeginGeneration(s.generation, c, model_->defaults(),
                        ReplySeed(*samplers[i]), !s.stop_at_eos());
        s.begun = true;
      }
      live.push_back(i);
    }
    if (live.empty())
      return steps;

    std::vector<BlockGeneration*> generations;
    std::vector<int> cached, budgets;
    for (const auto i : live) {
      generations.push_back(&states[i]->generation);
      const int length = states[i]->session->cache.length();
      cached.push_back(length);
      // The reply so far and this block stay within the context (the host
      // caps max_tokens likewise; this guards other callers).
      const auto& g = states[i]->generation;
      const auto pending = g.commit_pending ? c.canvas : 0;
      const auto room = static_cast<std::size_t>(std::max<long long>(
          0, static_cast<long long>(context_) - length - pending));
      budgets.push_back(static_cast<int>(std::min<std::size_t>(
          {max_tokens[i], room, static_cast<std::size_t>(INT_MAX)})));
    }
    const auto forward = [&](std::span<const BlockStep> block_steps,
                             std::span<const BlockCommit> commits) {
      std::vector<CanvasStep> canvases;
      canvases.reserve(block_steps.size());
      for (const auto& step : block_steps) {
        CanvasStep canvas{
            .session = states[live[step.generation]]->session,
            .tokens = step.tokens,
            .self_condition = step.self_condition,
            .make_probabilities = true,
            .has_pins = false,
            .temperature = step.temperature,
            .seed = step.seed,
            .entropy = step.entropy,
            .argmax = step.argmax,
            .sample = step.sample};
        for (int k = 0; k < step.suppressed_count; ++k)
          canvas.suppressed.ids[canvas.suppressed.count++] = step.suppressed[k];
        canvases.push_back(canvas);
      }
      std::vector<EncodeWork> encodes;
      for (const auto& commit : commits)
        encodes.push_back({.session = states[live[commit.generation]]->session,
                           .tokens = commit.tokens,
                           .rows = c.canvas});
      const std::lock_guard lock(gpu_mutex_);
      model_->Forward(canvases, encodes);
    };
    const auto cancelled = [&](std::size_t k) {
      const auto& check = states[live[k]]->cancelled;
      return check && check();
    };
    const auto outcomes = RunBlocks(c, static_cast<int>(context_), generations,
                                    cached, budgets, forward, cancelled);

    for (std::size_t k = 0; k < live.size(); ++k) {
      const auto i = live[k];
      const auto& out = outcomes[k];
      auto& step = steps[i];
      for (const auto id : out.tokens) {
        const auto token = static_cast<TextRunnerToken>(id);
        step.selections.push_back(
            {.stop = false, .token = token, .piece = Piece(token)});
      }
      step.stop = out.end_of_sequence || out.context_full || out.cancelled;
    }
    return steps;
  }

  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  cache::Identity component_identity_;
  mutable std::mutex gpu_mutex_;
};

/// Content identity of the checkpoint: config, generation config and every
/// safetensors header (tensor names, types, shapes and offsets).
std::string ArtifactIdentity(const std::filesystem::path& root) {
  crypto::Sha256Hasher hash;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(root))
    if (entry.is_regular_file()) {
      const auto name = entry.path().filename().string();
      if (name == "config.json" || name == "generation_config.json" ||
          name == "tokenizer.json" || name.ends_with(".safetensors"))
        files.push_back(entry.path());
    }
  std::sort(files.begin(), files.end());
  for (const auto& file : files) {
    const auto name = file.filename().string();
    hash.Update({reinterpret_cast<const std::uint8_t*>(name.data()),
                 name.size()});
    std::ifstream in(file, std::ios::binary);
    std::vector<char> bytes;
    if (name.ends_with(".safetensors")) {
      // The 8-byte header length and the JSON header.
      std::uint64_t header = 0;
      in.read(reinterpret_cast<char*>(&header), 8);
      if (!in || header > (std::uint64_t{64} << 20))
        throw std::runtime_error("invalid safetensors header in " +
                                 file.string());
      bytes.resize(8 + header);
      std::memcpy(bytes.data(), &header, 8);
      in.read(bytes.data() + 8, static_cast<std::streamsize>(header));
    } else {
      bytes.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
    }
    hash.Update({reinterpret_cast<const std::uint8_t*>(bytes.data()),
                 bytes.size()});
  }
  const auto digest = hash.Digest();
  static const char* hex = "0123456789abcdef";
  std::string out;
  for (const auto byte : digest) {
    out += hex[byte >> 4];
    out += hex[byte & 15];
  }
  return out;
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateDiffusionGemmaRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto fail = [&](std::string message)
      -> std::shared_ptr<ext::TextModelRunner> {
    if (error)
      *error = std::move(message);
    return nullptr;
  };
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled)
    return fail("DiffusionGemma does not support speculative decoding");
  if (!options.vision_model_path.empty())
    return fail("DiffusionGemma image input is not available yet (--mmproj)");
  try {
    const std::filesystem::path root(options.model_path);
    ModelOptions model_options;
    model_options.max_batch =
        static_cast<int>(std::max<std::size_t>(1, options.session_count));
    // The continuation cache restores prefixes itself.
    model_options.reuse_history = 0;
    model_options.stage_prompts = true;
    // A llama.cpp GGUF (what `gufo serve llm` opens) or a checkpoint
    // directory (the safetensors layout Bubo serves).
    std::error_code ec;
    const bool gguf = std::filesystem::is_regular_file(root, ec);
    std::shared_ptr<const core::GgufReader> reader = options.reader;
    if (gguf && !reader) {
      std::string open_error;
      auto opened = core::GgufReader::OpenFile(
          root, &open_error, {.allow_unsized_tensors = true});
      if (!opened)
        return fail(open_error);
      reader = std::move(opened);
    }
    const auto config = gguf ? LoadConfigGguf(*reader) : LoadConfig(root);
    std::uint32_t context = options.max_context;
    if (context == 0)
      context = 131072;
    context = std::min<std::uint32_t>(
        context, static_cast<std::uint32_t>(config.max_positions));
    model_options.max_context = static_cast<int>(context);
    // A durable (disk) cache needs the content identity of the files.
    // Hashing a GGUF reads all of it, so without a disk cache the loaded
    // model instance is the identity, as in the LFM3 extension.
    std::string identity;
    if (!gguf) {
      identity = "checkpoint-sha256:" + ArtifactIdentity(root);
    } else if (!options.disk_cache.directory.empty()) {
      identity = options.disk_cache.model_artifact_fingerprint.empty()
                     ? core::GgufIdentityHex(*reader)
                     : options.disk_cache.model_artifact_fingerprint;
      identity = std::string(core::kGgufIdentityScheme) + ':' + identity;
    }
    auto model = std::make_shared<Model>(root, model_options);
    context = static_cast<std::uint32_t>(model->options().max_context);
    if (identity.empty()) {
      std::ostringstream process;
      process << "process-model:" << model.get();
      identity = process.str();
    }
    return std::make_shared<Runner>(std::move(model), context,
                                    std::move(identity));
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

}  // namespace gufo::models::diffusion_gemma
