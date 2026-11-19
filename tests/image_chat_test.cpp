// RenderImageChat on Gufo's public ChatMessage: placeholders land at the
// content-part offsets, in order, inside DiffusionGemma's chat framing.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/image_chat.hpp"

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

gufo::ext::ChatMessage::ImagePart Image(std::size_t offset) {
  gufo::ext::ChatMessage::ImagePart part;
  part.offset = offset;
  part.bytes = std::make_shared<const std::vector<std::uint8_t>>(
      std::vector<std::uint8_t>{0x89, 'P', 'N', 'G'});
  return part;
}

}  // namespace

int main() {
  gufo::ext::ChatMessage user;
  user.content = "Compare  please";
  user.images = {Image(0), Image(8)};  // before "Compare", before " please"

  dg::Message message;
  message.role = "user";
  message.content = user.content;

  const std::vector<gufo::ext::ChatMessage> source{user};
  const auto out = dg::RenderImageChat(source, {message}, {}, false);

  Check(out.image_offsets.size() == 2, "two image offsets");
  if (out.image_offsets.size() == 2) {
    Check(out.text.compare(out.image_offsets[0], dg::kImagePlaceholder.size(),
                           dg::kImagePlaceholder) == 0,
          "first placeholder at its offset");
    Check(out.text.compare(out.image_offsets[1], dg::kImagePlaceholder.size(),
                           dg::kImagePlaceholder) == 0,
          "second placeholder at its offset");
    Check(out.image_offsets[0] < out.image_offsets[1], "images keep order");
    const auto between = out.text.substr(
        out.image_offsets[0] + dg::kImagePlaceholder.size(),
        out.image_offsets[1] - out.image_offsets[0] -
            dg::kImagePlaceholder.size());
    Check(between == "Compare ", "text between images");
  }
  Check(out.text.find('\x1f') == std::string::npos, "no marker left");

  bool threw = false;
  try {
    (void)dg::RenderImageChat(source, {}, {}, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "mismatched message count throws");

  auto bad = user;
  bad.images = {Image(99)};
  const std::vector<gufo::ext::ChatMessage> bad_source{bad};
  threw = false;
  try {
    (void)dg::RenderImageChat(bad_source, {message}, {}, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "offset past content throws");

  if (failures == 0)
    std::puts("diffusion_gemma image_chat: ok");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
