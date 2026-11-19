# Changelog

Notable changes to this extension are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

To release, add a `## [X.Y.Z] - YYYY-MM-DD` section above the previous one,
set the same `version` in `extension.json` and merge to `main`. CI then tags
`vX.Y.Z` and publishes a GitHub release with that section as its notes.

## [Unreleased]

## [0.1.0] - 2026-11-19

First release. Experimental: tested on one machine (Radeon 8060S, gfx1151,
Windows 11) with Gufo v0.35.0 (extension API 0.3.20).

### Added

- DiffusionGemma 26B-A4B-it served through `gufo serve llm` as a Gufo
  `TextModelRunner`, ported from Bubo (HaloForge Labs): one decode step
  denoises and commits one 256-token canvas.
- Batched denoising across concurrent sessions (`DecodeBatch`) and Gufo's
  continuation cache, with append-only global KV rows and per-checkpoint
  sliding-window state.
- The block-diffusion sampler (entropy-bound acceptance, temperature schedule,
  convergence test), seeded from the request.
- Weight loading from a llama.cpp `diffusion-gemma` Q8_0 GGUF, and from NVIDIA's
  NVFP4 checkpoint packed into a GGUF by `tools/convert_nvfp4_gguf.py`.
- Sampler settings read from the GGUF (`diffusion.eb_*`).
- Gemma 4 tool calls, buffered and streamed, through the
  `diffusion-gemma-native` tool dialect.
- Gemma 4 thinking returned as `reasoning_content`.
- CPU tests for config, chat template, tokenizer, sampler decisions, cache
  bookkeeping, block control flow, GGUF metadata and the tool dialect; GPU
  tests for the HIP kernels and the NVFP4 GGUF.

### Known limits

- No vision (`--mmproj` is refused) and no structured output
  (`response_format` is refused).
- No `gufo prompt`, `gufo chat` or `gufo bench` drivers, and the runner does
  not yet declare its per-session memory to Gufo.

[Unreleased]: https://github.com/gabrielperales/diffusiongemma-gufo-extension/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gabrielperales/diffusiongemma-gufo-extension/releases/tag/v0.1.0
