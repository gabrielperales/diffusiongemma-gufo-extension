#include "src/tool_dialect.hpp"

#include <cctype>
#include <gufo/ext/json.hpp>
#include <optional>
#include <string>

#include "src/chat.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool IsNameByte(char c) {
  const auto u = static_cast<unsigned char>(c);
  return std::isalnum(u) || c == '_' || c == '-' || c == '.' || c == ':' ||
         c == '/' || c == '$' || u >= 0x80;
}

/// True when `rest` is a strict prefix of `literal`: the text ends where the
/// literal may still be arriving.
bool PrefixOf(std::string_view rest, std::string_view literal) {
  return rest.size() < literal.size() && literal.starts_with(rest);
}

/// How long the tail of `text` that may be the start of `literal` is.
std::size_t PendingLiteral(std::string_view text, std::string_view literal) {
  for (std::size_t n = std::min(text.size(), literal.size() - 1); n > 0; --n)
    if (literal.starts_with(text.substr(text.size() - n)))
      return n;
  return 0;
}

enum class Outcome { kComplete, kIncomplete, kMalformed };

/// Parses one call starting at `text[pos]` (at kToolCallOpen). On kComplete,
/// `*end` is the byte after its closer. On kIncomplete (text ended inside
/// it), `call` holds what is final so far.
Outcome ParseCall(std::string_view text, std::size_t pos, std::size_t* end,
                  ext::ScannedToolCall* call) {
  std::size_t p = pos + kToolCallOpen.size();
  const auto rest = [&] { return text.substr(p); };
  if (!rest().starts_with(kToolNamePrefix))
    return PrefixOf(rest(), kToolNamePrefix) ? Outcome::kIncomplete
                                             : Outcome::kMalformed;
  p += kToolNamePrefix.size();
  const std::size_t name_start = p;
  while (p < text.size() && IsNameByte(text[p]) && text[p] != '{')
    ++p;
  if (p == text.size())
    return Outcome::kIncomplete;
  if (text[p] != '{' || p == name_start)
    return Outcome::kMalformed;
  call->name = std::string(text.substr(name_start, p - name_start));
  ++p;  // '{'
  for (bool first = true;; first = false) {
    while (p < text.size() && IsSpace(text[p]))
      ++p;
    if (p == text.size())
      return Outcome::kIncomplete;
    if (text[p] == '}') {
      ++p;
      break;
    }
    if (!first) {
      if (text[p] != ',')
        return Outcome::kMalformed;
      ++p;
      while (p < text.size() && IsSpace(text[p]))
        ++p;
      if (p == text.size())
        return Outcome::kIncomplete;
    }
    // Key: unquoted up to ':', or between quotes.
    std::string key;
    if (rest().starts_with(kStringQuote)) {
      const auto close = text.find(kStringQuote, p + kStringQuote.size());
      if (close == std::string_view::npos)
        return Outcome::kIncomplete;
      key = std::string(text.substr(p + kStringQuote.size(),
                                    close - p - kStringQuote.size()));
      p = close + kStringQuote.size();
      if (p == text.size())
        return Outcome::kIncomplete;
      if (text[p] != ':')
        return Outcome::kMalformed;
    } else {
      const auto colon = text.find_first_of(":{}[],", p);
      if (colon == std::string_view::npos)
        return Outcome::kIncomplete;  // The key is still arriving.
      if (text[colon] != ':')
        return Outcome::kMalformed;
      key = std::string(text.substr(p, colon - p));
      while (!key.empty() && IsSpace(key.back()))
        key.pop_back();
      p = colon;
      if (key.empty())
        return Outcome::kMalformed;
    }
    ++p;  // ':'
    while (p < text.size() && IsSpace(text[p]))
      ++p;
    if (p == text.size() || PrefixOf(rest(), kStringQuote))
      return Outcome::kIncomplete;
    // A string value: raw text up to the closing quote. While it is still
    // open, report what has arrived, minus a tail that may be the quote.
    if (rest().starts_with(kStringQuote)) {
      const auto start = p + kStringQuote.size();
      const auto close = text.find(kStringQuote, start);
      if (close == std::string_view::npos) {
        auto value = text.substr(start);
        value.remove_suffix(PendingLiteral(value, kStringQuote));
        call->unfinished = ext::ScannedToolArgument{
            .name = key, .value = std::string(value), .is_string = true};
        return Outcome::kIncomplete;
      }
      call->arguments.push_back(
          {.name = key,
           .value = std::string(text.substr(start, close - start)),
           .is_string = true});
      p = close + kStringQuote.size();
      continue;
    }
    // Any other value is converted to JSON text once it is whole.
    std::size_t consumed = 0;
    const auto value = ParseNativeValue(rest(), &consumed);
    if (!value) {
      // An object or array may still be arriving; a bare literal is whole
      // only when its delimiter has arrived.
      return Outcome::kIncomplete;
    }
    if (value->is_string())
      call->arguments.push_back(
          {.name = key, .value = value->get_str(), .is_string = true});
    else
      call->arguments.push_back(
          {.name = key, .value = value->dump(), .is_string = false});
    p += consumed;
  }
  while (p < text.size() && IsSpace(text[p]))
    ++p;
  if (!rest().starts_with(kToolCallClose))
    return p == text.size() || PrefixOf(rest(), kToolCallClose)
               ? Outcome::kIncomplete
               : Outcome::kMalformed;
  *end = p + kToolCallClose.size();
  return Outcome::kComplete;
}

}  // namespace

void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> /*markers*/,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial) {
  std::size_t pos = text.find(kToolCallOpen);
  while (pos != std::string_view::npos) {
    ext::ScannedToolCall call;
    std::size_t end = 0;
    const auto outcome = ParseCall(text, pos, &end, &call);
    if (outcome == Outcome::kComplete) {
      call.valid = true;
      call.complete = true;
      calls->push_back(std::move(call));
      spans->emplace_back(pos, end);
      pos = text.find(kToolCallOpen, end);
      continue;
    }
    if (outcome == Outcome::kIncomplete) {
      // Only a call whose name has arrived can be streamed; the final parse
      // leaves an unterminated call as text.
      if (partial && !call.name.empty()) {
        call.valid = true;
        call.complete = false;
        calls->push_back(std::move(call));
      }
      return;
    }
    // Malformed: it stays text; look for a later call.
    pos = text.find(kToolCallOpen, pos + kToolCallOpen.size());
  }
}

void RegisterToolDialect() {
  sampling::JsonConstraint::ToolSyntax syntax;
  syntax.open = std::string(kToolCallOpen);
  syntax.name_prefix = std::string(kToolNamePrefix);
  syntax.name_suffix = "";
  syntax.close = std::string(kToolCallClose);
  syntax.name_delimiters = "{";
  syntax.values.string_open = std::string(kStringQuote);
  syntax.values.string_close = std::string(kStringQuote);
  syntax.values.bare_keys = true;
  syntax.values.whitespace = false;
  syntax.values.sorted_keys = true;
  syntax.reasoning_header = true;
  ext::RegisterToolDialect({
      .format = sampling::JsonConstraint::ToolFormat::kJson,
      .markers = {kToolCallOpen},
      .reasoning_markers = {"<|tool_call>call:"},
      .closers = {kToolCallClose},
      .scan_tool_calls = &ScanToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = std::move(syntax),
  });
}

}  // namespace gufo::models::diffusion_gemma
