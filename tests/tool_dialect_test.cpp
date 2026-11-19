// Gemma 4 native tool calls through the extension's ToolDialect scanner:
// value conversion, several calls, malformed calls, and partial (streamed)
// parses that only grow towards the final one.
#include <cstdio>
#include <cstdlib>
#include <gufo/ext/json.hpp>
#include <string>
#include <vector>

#include "src/tool_dialect.hpp"

namespace dg = gufo::models::diffusion_gemma;
using gufo::ext::ScannedToolCall;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

struct Scan {
  std::vector<ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
};

Scan Run(std::string_view text, bool partial) {
  Scan out;
  const std::string_view markers[] = {dg::kToolCallOpen};
  dg::ScanToolCalls(text, markers, &out.calls, &out.spans, partial);
  return out;
}

const gufo::ext::ScannedToolArgument* Arg(const ScannedToolCall& call,
                                          const std::string& name) {
  for (const auto& a : call.arguments)
    if (a.name == name)
      return &a;
  return nullptr;
}

}  // namespace

int main() {
  const std::string q = std::string(dg::kStringQuote);
  const std::string weather =
      "<|tool_call>call:get_weather{city:" + q + "Madrid, {España}" + q +
      ",days:3,metric:true,opts:{hours:[6,12],lang:" + q + "es" + q +
      "}}<tool_call|>";

  // One call: every value type.
  {
    const auto s = Run(weather, false);
    Check(s.calls.size() == 1 && s.spans.size() == 1, "one call");
    if (s.calls.size() == 1) {
      const auto& c = s.calls[0];
      Check(c.name == "get_weather" && c.valid && c.complete, "name, complete");
      const auto* city = Arg(c, "city");
      Check(city && city->is_string && city->value == "Madrid, {España}",
            "string with comma and braces");
      const auto* days = Arg(c, "days");
      Check(days && !days->is_string &&
                gufo::json::parse(days->value).as_double() == 3,
            "number as JSON");
      const auto* metric = Arg(c, "metric");
      Check(metric && !metric->is_string && metric->value == "true",
            "boolean as JSON");
      const auto* opts = Arg(c, "opts");
      bool nested = false;
      if (opts && !opts->is_string) {
        const auto v = gufo::json::parse(opts->value);
        nested = v.find("lang") && v.find("lang")->get_str() == "es" &&
                 v.find("hours") && v.find("hours")->size() == 2;
      }
      Check(nested, "nested object and array as JSON");
      Check(s.spans[0].first == 0 && s.spans[0].second == weather.size(),
            "span covers the call");
    }
  }

  // Two calls with text between and after; the text stays content.
  {
    const std::string second = "<|tool_call>call:noop{}<tool_call|>";
    const std::string text = weather + "\nthen " + second + " done";
    const auto s = Run(text, false);
    Check(s.calls.size() == 2 && s.calls[1].name == "noop" &&
              s.calls[1].arguments.empty(),
          "second call without arguments");
    Check(s.spans.size() == 2 && s.spans[1].first == weather.size() + 6 &&
              text.substr(s.spans[1].second) == " done",
          "spans leave the text between and after");
  }

  // A malformed call stays text; a later call is still read.
  {
    const std::string text =
        "<|tool_call>get_weather{city:x}<tool_call|> " + weather;
    const auto s = Run(text, false);
    Check(s.calls.size() == 1 && s.calls[0].name == "get_weather" &&
              s.spans.size() == 1 && s.spans[0].first > 0,
          "malformed call stays text");
  }

  // The final parse leaves an unterminated call as text.
  {
    const auto s = Run(weather.substr(0, weather.size() - 3), false);
    Check(s.calls.empty() && s.spans.empty(), "unterminated call is text");
  }

  // Partial parses of every prefix agree with the final call and grow.
  {
    const auto final_call = Run(weather, false).calls.at(0);
    std::size_t last_args = 0, last_unfinished = 0;
    bool grows = true, consistent = true;
    for (std::size_t n = 1; n <= weather.size(); ++n) {
      const auto s = Run(std::string_view(weather).substr(0, n), true);
      if (s.calls.empty())
        continue;
      const auto& c = s.calls[0];
      consistent &= c.name == final_call.name;
      for (const auto& a : c.arguments) {
        const auto* f = Arg(final_call, a.name);
        consistent &= f && f->value == a.value && f->is_string == a.is_string;
      }
      std::size_t unfinished = 0;
      if (c.unfinished) {
        const auto* f = Arg(final_call, c.unfinished->name);
        consistent &= f && f->is_string && f->value.starts_with(c.unfinished->value);
        unfinished = c.unfinished->value.size();
      }
      if (c.arguments.size() < last_args ||
          (c.arguments.size() == last_args && unfinished < last_unfinished &&
           c.unfinished))
        grows = false;
      if (c.arguments.size() > last_args)
        last_unfinished = 0;
      last_args = c.arguments.size();
      if (c.unfinished)
        last_unfinished = unfinished;
      if (n == weather.size())
        consistent &= c.complete;
    }
    Check(consistent, "partial parses agree with the final call");
    Check(grows, "partial parses only grow");
  }

  if (failures == 0)
    std::puts("diffusion_gemma tool dialect: ok");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
