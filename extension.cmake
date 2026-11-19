# DiffusionGemma 26B-A4B block-diffusion text model, ported from Bubo
# (haloforge-labs/bubo, MIT). See README.md.

# Host code: configuration, probe, tokenizer, chat framing and the
# block-diffusion sampler decisions. Builds and tests without a GPU.
gufo_extension_target(
  NAME gufo_diffusion_gemma
  SOURCES
    src/config.cpp
    src/probe.cpp
    src/tokenizer.cpp
    src/chat.cpp
    src/image_chat.cpp
    src/denoise.cpp
    src/cache_state.cpp
    src/image_prompt.cpp
    src/read.cpp
    src/extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

if(TARGET gufo_diffusion_gemma)
  target_include_directories(gufo_diffusion_gemma PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_diffusion_gemma PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# GPU engine: kernels, weight loading (NVFP4, FP8, GGUF) and the
# block-diffusion engine. The kernels are written for explicit FP32
# arithmetic and BF16 boundaries, so contraction stays off. clang-cl ignores
# GCC-style options without the /clang: prefix that GUFO_CLANG_OPTION adds.
gufo_extension_target(
  NAME gufo_diffusion_gemma_hip
  SOURCES
    src/profiler.cpp
    src/upload.cpp
    src/weights.cpp
    src/weights_gguf.cpp
    src/weights_fp8.cpp
    src/weights_nvfp4.cpp
    src/model.cpp
    src/engine.cpp
  LINK_LIBRARIES
    gufo_diffusion_gemma
    gufo_ext_runtime
    hip::host
    Threads::Threads
)

if(TARGET gufo_diffusion_gemma_hip)
  gufo_extension_hip_sources(gufo_diffusion_gemma_hip
    SOURCES
      src/kernels.hip
      src/attention.hip
      src/moe.hip
      src/soft_embed.hip
      src/dense.hip
  )
  target_compile_options(gufo_diffusion_gemma_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
  target_include_directories(gufo_diffusion_gemma_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# The text provider (phase 4 adds the runner and drivers).
gufo_extension_target(
  NAME gufo_diffusion_gemma_text
  SOURCES
    src/provider.cpp
  LINK_LIBRARIES
    gufo_diffusion_gemma_hip
    gufo_diffusion_gemma
    gufo_ext_runtime
)

if(TARGET gufo_diffusion_gemma_text)
  target_include_directories(gufo_diffusion_gemma_text PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# gufo_ext_runtime only declares the public services; standalone test
# executables link their implementations.
set(_gufo_diffusion_gemma_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_diffusion_gemma_test_libs gufo_ext_hip)
endif()

foreach(_dg_test config chat tokenizer denoise cache_state image_chat)
  gufo_extension_target(
    NAME diffusion_gemma_${_dg_test}_test
    SOURCES
      tests/${_dg_test}_test.cpp
    LINK_LIBRARIES
      gufo_diffusion_gemma
      gufo_ext_runtime
      ${_gufo_diffusion_gemma_test_libs}
  )
  if(TARGET diffusion_gemma_${_dg_test}_test)
    add_test(NAME diffusion_gemma.${_dg_test}
      COMMAND diffusion_gemma_${_dg_test}_test)
    # The tokenizer test needs DIFFUSION_GEMMA_MODEL and skips (77) without.
    set_tests_properties(diffusion_gemma.${_dg_test} PROPERTIES
      LABELS "diffusion-gemma;cpu" SKIP_RETURN_CODE 77)
  endif()
endforeach()

# GPU tests and the standalone runner (reference comparisons with Bubo).
gufo_extension_target(
  NAME diffusion_gemma_kernels_hip_test
  SOURCES
    tests/kernels_hip_test.cpp
  LINK_LIBRARIES
    gufo_diffusion_gemma_hip
    gufo_diffusion_gemma
    gufo_ext_runtime
    ${_gufo_diffusion_gemma_test_libs}
)

if(TARGET diffusion_gemma_kernels_hip_test)
  add_test(NAME diffusion_gemma.kernels_hip
    COMMAND diffusion_gemma_kernels_hip_test)
  set_tests_properties(diffusion_gemma.kernels_hip PROPERTIES
    LABELS "diffusion-gemma;hip;gpu;gfx1151" SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME diffusion_gemma_run
  SOURCES
    tools/diffusion_gemma_run.cpp
  LINK_LIBRARIES
    gufo_diffusion_gemma_hip
    gufo_diffusion_gemma
    gufo_ext_runtime
    ${_gufo_diffusion_gemma_test_libs}
)

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/NOTICE"
  DESTINATION share/licenses/gufo/extensions/diffusion_gemma)
