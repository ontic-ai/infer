# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

- No current unreleased changes.

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

