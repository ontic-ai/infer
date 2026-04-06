# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

- No current unreleased changes.

## [0.1.0] - 2026-04-05

- Initial public release.
- Added `InferenceBackend` trait with `complete`, `stream`, `embed`, and `extract`.
- Added `MockBackend` for deterministic test behavior.
- Added `LlamaBackend` using `llama-cpp-2` for production inference.
- Added model discovery for Ollama-style GGUF directories.
- Organized backend code under `src/backend/`.

