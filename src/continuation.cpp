#include "src/continuation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/snapshot_copy.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>


namespace gufo::models::diffusion_gemma {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

ContinuationLayout::ContinuationLayout(const Config& c, std::uint32_t context,
                                       int sliding_keep)
    : context_(context), sliding_keep_(sliding_keep) {
  if (!context || c.layers <= 0 ||
      static_cast<int>(c.is_global.size()) != c.layers || sliding_keep <= 0)
    throw std::invalid_argument("invalid DiffusionGemma continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, int layer, int head, bool value) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back(
        {{.id = {id},
          .layout_version = 1,
          .kind = row ? cache::ComponentKind::kAppendRows
                      : cache::ComponentKind::kPrivateState,
          .row_bytes = row,
          .rows_per_chunk = row ? cache::Rows{256} : 0,
          .state_bytes = state},
         part,
         layer,
         head,
         value});
  };
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes, 0, 0, false);
  for (int l = 0; l < c.layers; ++l) {
    const std::size_t row = std::size_t(c.head_dim(l)) * 2;  // BF16
    if (c.is_global[l]) {
      for (int h = 0; h < c.kv_heads(l); ++h) {
        add(ContinuationPart::kGlobalRows, row, 0, l, h, false);
        add(ContinuationPart::kGlobalRows, row, 0, l, h, true);
      }
    } else {
      add(ContinuationPart::kSlidingWindow, 0,
          2 * std::size_t(c.kv_heads(l)) * sliding_keep * row, l, 0, false);
    }
  }
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position > context_)
    throw std::invalid_argument("invalid DiffusionGemma continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_)
    positions.push_back({component.descriptor.id, position});
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.row_bytes * position;
  return bytes;
}

std::vector<cache::BackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // As LFM3: contiguous page groups retain one payload owner and charge all
  // slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

class ContinuationAdapter::State final : public cache::Slot,
                                         public SessionHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->hooks = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    Release();
    session->hooks = nullptr;
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed;
  }
  void RequireIdle() const {
    if (restoring || failed || pending)
      throw std::logic_error("DiffusionGemma slot is not ready for execution");
  }
  // A pass writes rows [first, end): an encoded chunk extends the frontier,
  // a canvas being denoised fills the rows just past it. Rows past the
  // context are scratch the cache never holds.
  void BeforeWrite(int first, int end) override {
    RequireIdle();
    const auto limit = static_cast<int>(adapter.context_);
    if (first >= limit)
      return;
    end = std::min(end, limit);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, first, end);
  }
  void GuardAll() {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
  }
  void Release() noexcept {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      layout_(model_->config(), context,
              CacheState(model_->config().sliding_window,
                         model_->options().reuse_history)
                  .sliding_keep()),
      identity_(std::move(identity)) {
  if (identity_.empty())
    throw std::invalid_argument("invalid DiffusionGemma adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(1);  // Component layout version: BF16 K/V rows per head.
  put(context);
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another DiffusionGemma adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  return std::make_unique<State>(*this, guard, model_->CreateSession());
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid DiffusionGemma slot");
  return layout_.Positions(
      static_cast<std::uint32_t>(state.session->cache.length()));
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid DiffusionGemma prefill plan");
  std::vector<Rows> result;
  const auto capacity = static_cast<Rows>(model_->options().prefill_chunk);
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid DiffusionGemma component ID");
  return layout_.Components()[id.value - 1];
}
std::byte* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  if (component.part != ContinuationPart::kGlobalRows)
    throw std::invalid_argument("component has no append rows");
  const auto& c = model_->config();
  auto& buffer = component.value ? state.session->v_cache[component.layer]
                                 : state.session->k_cache[component.layer];
  return buffer.as<std::byte>() +
         static_cast<std::size_t>(component.head) * model_->global_capacity() *
             c.head_dim(component.layer) * 2;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.RequireIdle();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete DiffusionGemma restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate DiffusionGemma frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered.front().valid_rows > context_)
    throw std::invalid_argument("DiffusionGemma frontier exceeds context");
  const auto target = static_cast<std::uint32_t>(ordered.front().valid_rows);
  if (ordered != layout_.Positions(target))
    throw std::invalid_argument("inconsistent DiffusionGemma component frontiers");
  // Reserve first; any refusal leaves execution valid. Guard all ranges while
  // the original state is still readable.
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

// Logical byte layout of one private component at a frontier. A load may
// arrive before the metadata, so regions follow from the frontier alone. Rings
// are copied in physical order, which keeps the absolute-position mapping.
ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component,
    std::uint32_t position) const {
  Region result;
  const auto add = [&](void* pointer, std::size_t offset, std::size_t bytes,
                       bool host = false) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  auto& session = *state.session;
  switch (component.part) {
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), 0, bytes, true);
      break;
    case ContinuationPart::kSlidingWindow: {
      // [kv heads][keep][head dim] keys, then values; the live rows of each
      // head are the newest min(position, keep) positions at slots 0...
      const auto& c = model_->config();
      const int l = component.layer;
      const std::size_t row = std::size_t(c.head_dim(l)) * 2;
      const std::size_t keep = layout_.sliding_keep();
      const std::size_t live =
          std::min<std::size_t>(position, keep) * row;
      const std::size_t head_stride =
          std::size_t(model_->sliding_capacity()) * row;
      for (int h = 0; h < c.kv_heads(l); ++h) {
        add(session.k_cache[l].as<std::byte>() + h * head_stride,
            h * keep * row, live);
        add(session.v_cache[l].as<std::byte>() + h * head_stride,
            bytes / 2 + h * keep * row, live);
      }
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect DiffusionGemma private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid DiffusionGemma frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid DiffusionGemma private piece");
  const auto position =
      static_cast<std::uint32_t>(state.session->cache.length());
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  auto region = [&] {
    if (component.part == ContinuationPart::kMetadata) {
      const Metadata metadata{.position = position};
      std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
      return Region{{{host_metadata.data(), 0, host_metadata.size(), true}}};
    }
    return PrivateRegion(state, component, position);
  }();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes: clearing
        // the whole recycled copy on the host dominates capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DiffusionGemma load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect DiffusionGemma private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DiffusionGemma load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid DiffusionGemma private piece");
    const auto region = PrivateRegion(
        state, component,
        static_cast<std::uint32_t>(state.expected.front().valid_rows));
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid DiffusionGemma slot");
  if (component.part != ContinuationPart::kGlobalRows)
    throw std::invalid_argument("component has no append rows");
  const auto positions = Positions(slot);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > positions[id.value - 1].valid_rows ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid DiffusionGemma row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : RowData(state, component) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (!bytes.empty())
          Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                        transfer.Native()));
      });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DiffusionGemma load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid DiffusionGemma row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : RowData(state, component) + first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (!bytes.empty())
            Check(hip::CopySnapshotToDevice(destination, bytes.data(),
                                            bytes.size(), transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("DiffusionGemma validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  if (metadata.version != 1 ||
      metadata.position != state.expected.front().valid_rows)
    return refuse();
  state.session->cache.RestoreLength(static_cast<int>(metadata.position));
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    state.Release();
    state.session->cache.Clear();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}
}  // namespace gufo::models::diffusion_gemma
