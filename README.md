# diffusiongemma-gufo-extension

A [Gufo](https://github.com/gufo-org/gufo) extension that serves
DiffusionGemma 26B-A4B-it, Google's block-diffusion Gemma 4 model, on
Strix Halo (gfx1151). The model code is ported from
[Bubo](https://github.com/haloforge-labs/bubo), a Gufo fork that runs it
natively on Windows (see NOTICE).

**Status: phase 2 of 8 (host code).** The extension recognises a checkpoint
directory but does not run it yet.

| | |
| --- | --- |
| Gufo | v0.35.0 (extension API 0.3.21) |
| Platforms | Windows x86-64 (native, primary), Linux |
| Weights | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` safetensors + Gemma 4 vision `mmproj-*-BF16.gguf` |
| Maintainer | _to be named_ |

## How it maps onto Gufo

Gufo's text runner is autoregressive: prefill, then decode steps. Here a
decode step is a whole diffusion block. The runner declares
`multi_token_decode`, and one `DecodeStep` denoises a 256-token canvas (up
to 48 steps), commits it through the causal encoder and returns its tokens.
`DecodeBatch` batches the denoising steps of concurrent sessions. The prompt
goes through the causal encoder in `Prefill`. Global-layer KV rows are
append-only cache components; the sliding-window KV is private state.

Not available in v1: JSON-constrained output (`response_format`), Bubo's
`diffusion_*` read fields (they need Gufo HTTP support) and speculative
decoding.

## Build

```cmd
cd C:\src\gufo
tools\windows\vs-env.cmd cmake --preset windows-release -DGUFO_EXTENSION_DIRS=C:/src/diffusiongemma-gufo-extension
tools\windows\vs-env.cmd cmake --build --preset windows-release
```

CPU tests on Linux, from a Gufo checkout inside `nix develop`:

```sh
cmake --preset cpu-test -DGUFO_EXTENSION_DIRS=$HOME/src/diffusiongemma-gufo-extension
cmake --build --preset cpu-test --target diffusion_gemma_config_test
ctest --preset cpu-test -R '^diffusion_gemma\.'
```

## Roadmap

0. Gufo built from source; Bubo reference outputs (logits, seeded replies).
1. Scaffold, manifest, probe, config. **(done)**
2. Host code: tokenizer, chat framing, image placeholders, sampler decisions, cache bookkeeping. **(done)**
3. HIP kernels and NVFP4 weight loading.
4. `TextModelRunner`: prefill, block decode, batching, component cache.
5. Gemma 4 tool dialect (tolerant of a missing opening marker).
6. Vision through the Gemma 4 encoder and `--mmproj`.
7. Drivers, conformance, benchmarks against Bubo.
8. Release, registry review, llama-swap switch-over.
