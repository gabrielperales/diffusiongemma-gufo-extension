#ifndef DIFFUSION_GEMMA_EXT_CONTINUATION_HPP_
#define DIFFUSION_GEMMA_EXT_CONTINUATION_HPP_

#include <cstdint>
#include <functional>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/transfer_pool.hpp>
#include <memory>
#include <span>
#include <vector>

#include "src/config.hpp"
#include "src/model.hpp"

namespace gufo::models::diffusion_gemma {

/// What one continuation component holds. IDs refer to this inventory, never
/// to an allocation or a serving slot.
enum class ContinuationPart {
  kMetadata,
  /// One KV head of a global layer: the key or value rows of every position,
  /// final once written and shared between checkpoints.
  kGlobalRows,
  /// A sliding layer's keys then values of the newest kept positions. The
  /// cache shifts them after every pass, so they are copied whole.
  kSlidingWindow,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  int layer{0};
  int head{0};
  bool value{false};
};

inline constexpr std::size_t kContinuationMetadataBytes = 16;

/// The component inventory of DiffusionGemma's state at a context size.
class ContinuationLayout {
public:
  ContinuationLayout(const Config& config, std::uint32_t context,
                     int sliding_keep);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t position) const;
  [[nodiscard]] int sliding_keep() const { return sliding_keep_; }
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  /// Row bytes of every append component at a frontier.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t position) const;
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  int sliding_keep_;
  std::vector<ContinuationComponent> components_;
};

/// The continuation cache's view of DiffusionGemma sessions. Each slot owns
/// one Model session; the runner drives it through Model::Forward. The
/// caller supplies the artifact identity; the adapter adds its layout and
/// context.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model> model, std::uint32_t context,
                      cache::Identity identity);
  ~ContinuationAdapter() override;

  [[nodiscard]] Session& GetSession(cache::Slot& slot);

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const override {
    return layout_.BackingPlan(ram_bytes);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard& guard) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot& slot, cache::Rows first,
      cache::Rows end) const override;
  void BeginRestore(cache::Slot& slot,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Region;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] Region PrivateRegion(State&, const ContinuationComponent&,
                                     std::uint32_t position) const;
  [[nodiscard]] std::byte* RowData(State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);

  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  ContinuationLayout layout_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_CONTINUATION_HPP_
