# diffusiongemma-gufo-extension

A [Gufo](https://github.com/gufo-org/gufo) extension that serves
DiffusionGemma 26B-A4B-it, Google's block-diffusion Gemma 4 model, on
Strix Halo (gfx1151). The model code is ported from
[Bubo](https://github.com/haloforge-labs/bubo), a Gufo fork that runs it
natively on Windows (see NOTICE).

**Status: phase 4 of 8 (serving through `gufo serve llm`).** The extension recognises a checkpoint
directory but does not run it yet.

| | |
| --- | --- |
| Gufo | v0.35.0 (extension API 0.3.20) |
| Platforms | Windows x86-64 (native, primary), Linux |
| Weights | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` safetensors + Gemma 4 vision `mmproj-*-BF16.gguf` |
| Maintainer | _to be named_ |

## Weights

`gufo serve llm` opens text models as GGUF files. Two kinds work:

- **NVIDIA's NVFP4, repacked (recommended).** `tools/convert_nvfp4_gguf.py`
  writes the decoder tensors of `nvidia/diffusiongemma-26B-A4B-it-NVFP4`
  byte for byte into one GGUF (16.5 GiB), taking the metadata from a
  llama.cpp `diffusion-gemma` GGUF of the same model:

  ```sh
  python tools/convert_nvfp4_gguf.py \
      --checkpoint path/to/diffusiongemma-26B-A4B-it-NVFP4 \
      --metadata-from AlexAtomic/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf \
      --out diffusiongemma-26B-A4B-it-NVFP4.gguf --source-revision <commit>
  ```

  Packed E2M1 values and E4M3 block scales are GGUF I8 tensors; the
  provider accepts tensors Gufo's core cannot size. `nvfp4_gguf_test`
  compares every tensor with the checkpoint.
- **llama.cpp Q8_0** (`AlexAtomic/diffusiongemma-26B-A4B-it-GGUF`). It
  needs about 28 GB of GPU memory instead of 17, since the non-expert
  matrices are widened to BF16. Avoid `unsloth/diffusiongemma-26B-A4B-it-GGUF`:
  it was converted from an earlier release of the weights.

Measured on a Radeon 8060S with two sessions and a 16K context: the NVFP4
GGUF loads in 7.6 s and holds 29.5 GB of GPU memory; the Q8_0 takes 21 s
and 40.7 GB.

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
3. HIP kernels and NVFP4 weight loading. **(done)**
4. `TextModelRunner`: prefill, block decode, batching, component cache. **(serving; drivers pending)**
5. Gemma 4 tool dialect (tolerant of a missing opening marker).
6. Vision through the Gemma 4 encoder and `--mmproj`.
7. Drivers, conformance, benchmarks against Bubo.
8. Release, registry review, llama-swap switch-over.
