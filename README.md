# diffusiongemma-gufo-extension

A [Gufo](https://github.com/gufo-org/gufo) extension that serves Google's
**DiffusionGemma 26B-A4B-it**, a block-diffusion Gemma 4 model, on AMD Strix
Halo (gfx1151). It runs behind `gufo serve llm` with Gufo's OpenAI-compatible
API, continuation cache, request batching and tool calls.

The model code is ported from [Bubo](https://github.com/haloforge-labs/bubo),
a Gufo fork by HaloForge Labs that runs DiffusionGemma natively on Windows
(see [NOTICE](NOTICE)).

> **Status: experimental.** Tested on one machine (Ryzen AI Max+ 395, Radeon
> 8060S, Windows 11) with Gufo v0.35.0. Not yet reviewed for Gufo's extension
> registry. See [Quality](#quality) and [Known limits](#known-limits).

| | |
| --- | --- |
| Gufo | v0.35.0 (extension API 0.3.20), built from source |
| Platforms | Windows x86-64, gfx1151: tested. Linux: builds and passes CPU tests; not run on a GPU |
| Weights | NVIDIA's NVFP4 checkpoint packed into a GGUF (recommended), or a llama.cpp Q8_0 GGUF |
| Maintainer | [@gabrielperales](https://github.com/gabrielperales) |
| License | MIT (code). Weights: Apache 2.0 and the Gemma Terms of Use |

## How it works

Gufo's text runner is autoregressive: it prefills a prompt and then decodes
tokens. DiffusionGemma instead denoises a 256-token canvas over up to 48
steps, then commits it to its KV cache and starts the next one. The runner
maps one decode step to one canvas:

- **Prefill** runs the causal encoder over prompt chunks.
- **Decode** (`multi_token_decode`) denoises a canvas, returns its reply
  tokens and commits it before the next block. `DecodeBatch` batches the
  denoising steps of concurrent sessions into shared forward passes.
- **Continuation cache**: each global layer's KV heads are append-only rows
  shared between checkpoints; each sliding layer's window is private state.
- **Sampler**: the block-diffusion sampler (entropy-bound acceptance,
  temperature schedule, convergence test) uses the request's seed. Temperature,
  top-k/top-p and penalties do not apply.
- **Tool calls** use Gemma 4's native syntax
  (`<|tool_call>call:NAME{...}<tool_call|>`) through the
  `diffusion-gemma-native` tool dialect, including streamed calls.
- **Thinking**: Gemma 4's thinking channel is returned as reasoning
  (`reasoning_content`); enable it with `chat_template_kwargs.enable_thinking`.

## Build (Windows)

Follow Gufo's [BUILD-WINDOWS.md](https://github.com/gufo-org/gufo/blob/v0.35.0/BUILD-WINDOWS.md)
(Visual Studio Build Tools, TheRock 10.0.0, vcpkg, Rust 1.88 and the XRT SDK,
whose headers configure needs even without the NPU), then:

```cmd
cd C:\src\gufo
tools\windows\vs-env.cmd cmake --preset windows-release -DBUILD_TESTING=ON ^
    -DGUFO_EXTENSION_DIRS=C:/src/diffusiongemma-gufo-extension
tools\windows\vs-env.cmd cmake --build --preset windows-release --target gufo ^
    diffusion_gemma_run diffusion_gemma_kernels_hip_test diffusion_gemma_block_test
tools\windows\vs-env.cmd ctest --test-dir build\windows-release -R "^diffusion_gemma\."
```

Build only the targets you need: building `all` with `BUILD_TESTING=ON` also
builds Gufo's own extension fixture, which does not configure on Windows.
`gufo.exe` needs the DLLs and the `hipblaslt` and `rocblas` folders beside it
in the build directory; copy them together to run it elsewhere.

CPU tests on Linux, inside Gufo's `nix develop`:

```sh
cmake --preset cpu-test -DGUFO_EXTENSION_DIRS=$HOME/src/diffusiongemma-gufo-extension
cmake --build --preset cpu-test --target gufo diffusion_gemma_block_test
ctest --preset cpu-test -R '^diffusion_gemma\.'
```

## Weights

`gufo serve llm` opens text models as GGUF files.

### NVIDIA's NVFP4, packed into a GGUF (recommended)

`tools/convert_nvfp4_gguf.py` writes the decoder tensors of
[nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
byte for byte into one 16.5 GiB GGUF. Its metadata (architecture, tokenizer,
chat template, sampler settings) comes from a llama.cpp `diffusion-gemma`
GGUF of the same model; the token types come from the checkpoint's
`tokenizer.json`.

```sh
python tools/convert_nvfp4_gguf.py \
    --checkpoint path/to/diffusiongemma-26B-A4B-it-NVFP4 \
    --metadata-from path/to/AlexAtomic/diffusiongemma-26B-A4B-it-Q8_0.gguf \
    --out diffusiongemma-26B-A4B-it-NVFP4.gguf \
    --source-revision ec4ff3df205028f4e81c954c2227f9312b3ec2ea
```

It takes about two minutes and checks every tensor after writing; it also
writes per-tensor SHA-256 hashes beside the output. Packed E2M1 values and
E4M3 block scales are stored as GGUF `I8` tensors, which Gufo's core cannot
size, so the provider accepts unsized tensors. Only this extension reads this
layout. Vision tensors are left out (they belong in a separate mmproj file).

### llama.cpp Q8_0

[AlexAtomic/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/AlexAtomic/diffusiongemma-26B-A4B-it-GGUF)
`Q8_0` works directly. It needs about 28 GB of GPU memory instead of 17, since
the non-expert matrices are widened to BF16. Other GGUFs checked:

- `unsloth/diffusiongemma-26B-A4B-it-GGUF` was converted from an earlier
  release of the weights (`output_norm` differs from the final checkpoint);
  avoid it.
- `FreedomAISVR/DiffusionGemma-26B-A4B-it-NVFP4-GGUF` lacks the
  self-conditioning and encoder-scale tensors: it cannot run block diffusion.
- K-quants (Q4_K_M, Q5_K_M, Q6_K) are not supported: the GGUF path loads only
  Q8_0 and F32 tensors.

## Run

```sh
gufo serve -p 8090 llm -m diffusiongemma-26B-A4B-it-NVFP4.gguf \
    -c 131072 -j 2 --served-model-name diffusiongemma-26b
```

`-j` sets the sessions whose denoising steps are batched together. Requests
use the OpenAI chat API; pass `seed` for a reproducible reply, `tools` for
tool calls and `chat_template_kwargs: {"enable_thinking": true}` to think.

## Quality

Measured on a Radeon 8060S (Windows 11, Gufo v0.35.0) with the NVFP4 GGUF.

**Fidelity to Bubo**
- The NVFP4 GGUF holds the checkpoint's 35,191 decoder tensors byte for byte
  (`nvfp4_gguf_test`) and gives the same replies as the checkpoint directory.
- With the same seed, the runner's replies equal Bubo's engine's.

**Serving** (64K context, 4 sessions)
- Needle retrieval at 8K, 33K and 63K tokens: found each time. Prefill 4,264,
  2,491 and 1,864 tokens/s.
- Agent loop: 6 tool-calling turns, with the cache growing each turn.
- A checkpoint restored from RAM (284 MB) gives the same reply, character for
  character, as the live session it was captured from.
- 6 streams cancelled mid-block, 8 concurrent mixed requests (wall 7.9 s
  against 36 s serially), prompts over the context (HTTP 400) and a reply
  that fills the context exactly (`finish_reason: length`).
- Memory: 30.7 GB of GPU memory, steady; 2.7 to 5.5 GB of host RAM.
- Load: 7.6 s. A single request decodes 35 to 40 tokens/s for short replies;
  the denoiser takes about 130 to 150 ms per step.

**Tool calls**: 68 of 68 parsed across seeds, Spanish and English prompts,
two to three tools, with and without thinking, buffered and streamed.

**Tests**: config, chat template, tokenizer (against `tokenizer.json`),
sampler decisions, cache bookkeeping, block control flow, GGUF metadata,
tool dialect (including streamed partial calls) and HIP kernels.

## Known limits

- **No vision yet.** `--mmproj` is refused.
- **No structured output.** The diffusion sampler cannot follow a grammar
  token by token, so `response_format` is refused. For the same reason
  `tool_choice: required` and `strict` tools are best effort.
- **Replies depend on how the prompt was computed.** The prefill chunk size
  changes floating-point results (the MoE kernels group rows by pass size),
  so a seeded reply served from the cache may differ from an uncached one.
  Both are deterministic. This is inherited from Bubo's kernels.
- **Missing call opener.** Bubo saw the model sometimes write a call without
  `<|tool_call>`. Not seen in these tests; if it happens the call stays text,
  since Gufo's dialect API only scans from a call marker.
- **No `gufo prompt`, `gufo chat` or `gufo bench` drivers yet**, and the
  runner does not yet declare its per-session memory to Gufo.
- **Sliding-window state is copied whole** into each checkpoint (about 210 MB)
  instead of using Gufo's window-row components.
- Bubo's diffusion read fields (`diffusion_*`) are not available: they need
  HTTP support in Gufo.

## Roadmap

1. Scaffold, manifest, probe, config. **(done)**
2. Host code: tokenizer, chat template, sampler, cache bookkeeping. **(done)**
3. HIP kernels and weight loading. **(done)**
4. `TextModelRunner` and `gufo serve`. **(done; drivers and memory claim pending)**
5. Gemma 4 tool dialect. **(done)**
6. Vision through the Gemma 4 encoder and `--mmproj`.
7. Drivers, conformance suite, benchmarks.
8. Registry review and `gufo model install` entries.

## Credits

- Model: [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it);
  NVFP4 quantization by NVIDIA.
- DiffusionGemma kernels, engine, tokenizer and chat code: Bubo by HaloForge
  Labs (MIT).
- Continuation adapter: after Gufo's LFM3 extension (MIT).
- GGUF metadata: AlexAtomic's llama.cpp conversion.
