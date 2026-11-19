#ifndef DIFFUSION_GEMMA_EXT_IMAGE_CHAT_HPP_
#define DIFFUSION_GEMMA_EXT_IMAGE_CHAT_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/chat.hpp"
#include <gufo/ext/text.hpp>

namespace gufo::models::diffusion_gemma {

/// Where the Gemma 4 chat template places one image's soft tokens.
inline constexpr std::string_view kImagePlaceholder = "<|image|>";

struct ImageChatRendered {
  std::string text;
  std::vector<std::size_t> image_offsets;
};

/// Add image placeholders at the HTTP content-part offsets without changing
/// DiffusionGemma's chat framing, including tool responses and later turns.
ImageChatRendered RenderImageChat(
    std::span<const ext::ChatMessage> source,
    std::vector<Message> messages, const std::vector<json::Value>& tools,
    bool thinking);

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_IMAGE_CHAT_HPP_
