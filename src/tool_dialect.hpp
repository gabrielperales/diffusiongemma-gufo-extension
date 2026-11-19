#ifndef DIFFUSION_GEMMA_EXT_TOOL_DIALECT_HPP_
#define DIFFUSION_GEMMA_EXT_TOOL_DIALECT_HPP_

#include <cstddef>
#include <gufo/ext/text.hpp>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::diffusion_gemma {

/// The identity of DiffusionGemma's tool dialect in Gufo's registry.
inline constexpr std::string_view kToolDialectId = "diffusion-gemma-native";

/// Gemma 4's native call syntax, as the chat template renders calls:
///   <|tool_call>call:NAME{key:value,...}<tool_call|>
/// with strings between <|"|>, unquoted keys sorted by name and no
/// whitespace.
inline constexpr std::string_view kToolCallOpen = "<|tool_call>";
inline constexpr std::string_view kToolCallClose = "<tool_call|>";
inline constexpr std::string_view kToolNamePrefix = "call:";
inline constexpr std::string_view kStringQuote = "<|\"|>";

/// Reads the calls in `text`, which starts at a call marker: each complete
/// call, and with `partial` the one still being written (see
/// ext::ToolDialect). A malformed call is left out, so it stays text.
void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial);

/// Registers the dialect with Gufo; call during extension registration.
void RegisterToolDialect();

}  // namespace gufo::models::diffusion_gemma

#endif  // DIFFUSION_GEMMA_EXT_TOOL_DIALECT_HPP_
