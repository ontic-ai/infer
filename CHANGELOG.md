# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.3.2] - 2026-05-11

- Made the `turboquant` dependency opt-in behind a new `turboquant` feature so default `infer` builds no longer pull the conflicting `ort` graph.
- Kept TurboQuant KV configuration types available in the public API while failing fast with a clear message if TurboQuant is requested without enabling the feature.

## [0.3.1] - 2026-05-11

- Updated direct dependencies to the latest published releases for `tokio`, `uuid`, `llama-cpp-2`, `criterion`, and `shaderc`.

## [0.3.0] - 2026-05-11

- Added `InferenceParams::top_k`, `InferenceParams::repeat_penalty`, and `InferenceParams::stop_sequences` for explicit sampler and stop control.
- Wired llama generation through a real sampler chain so temperature, top-k, top-p, and repeat penalty now affect completions and streams, including the degraded KV-quant fallback path.
- Enforced stop-sequence truncation in both `complete()` and `stream()` and made streaming emit incrementally on background workers instead of buffering the full response first.
- Set llama context `n_batch` and `n_ubatch` from the active context window to avoid prompt-decode `ubatch` assertions on long prompts.
- Added algorithm-oriented KV quantization types: `KvQuantization`, `RotorQuantization`, `TurboQuantization`, and `TurboQuantStrategy`.
- Added initial TurboQuant integration through the upstream `turboquant` crate in the CPU/reference KV pipeline.
- Modularized `kv_quant` around generic compressed payloads so RotorQuant-style and TurboQuant algorithms share one pipeline surface.
- Added TurboQuant benchmark variants and README use-case guidance, including the current llama.cpp support boundary.

## [0.2.0] - 2026-05-06

- Added native KV cache quantization API: `KvCacheConfig`, `KvQuantization`.
- Implemented CPU reference kernels for Planar2/Planar3 (2D Givens) and Iso4/Iso3 (4D quaternion) with Lloyd-Max scalar quantization.
- Added pure Rust KV quantization pipeline with compression/decompression round-trip tests.
- Added Vulkan-gated shader assets and build-time GLSL -> SPIR-V compilation pipeline.
- Added `InferenceParams::kv_cache` (default `KvCacheConfig::none()`) for backward-compatible per-request configuration.
- Added `LlamaBackend::with_kv_cache(KvCacheConfig)` and per-request KV cache dtype mapping via llama.cpp context params.
- Added precomputed Lloyd-Max centroids for 2/3/4-bit scalar quantization.

## [0.1.2] - 2026-04-09

- Added Vulkan GPU backend support for Windows and Linux.
- Added `vulkan` feature and prioritized backend selection: `metal` on macOS, Vulkan on Windows/Linux, CUDA as legacy opt-in, and CPU fallback otherwise.
- Bumped `llama-cpp-2` dependency to `0.1.143`.
- Changed default feature set to empty so CPU/mock builds no longer require `llama`.
- Added `BUILDING.md` with Vulkan SDK setup and Ninja generator guidance.
- Updated README feature documentation and backend detection behavior.

## [0.1.0] - 2026-04-05

- Initial public release.
- Added `InferenceBackend` trait with `complete`, `stream`, `embed`, and `extract`.
- Added `MockBackend` for deterministic test behavior.
- Added `LlamaBackend` using `llama-cpp-2` for production inference.
- Added model discovery for Ollama-style GGUF directories.
- Organized backend code under `src/backend/`.

