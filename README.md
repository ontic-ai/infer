# infer

`infer` is a lightweight Rust crate for local LLM inference and model discovery.
It defines a single backend trait and supports multiple backends behind a
consistent API.

## What this crate provides

- A unified `InferenceBackend` trait for text completion, token streaming,
  embedding, and extraction.
- A clean backend module layout under `src/backend/`.
- A deterministic `MockBackend` for tests and local development.
- A production-grade `LlamaBackend` powered by `llama-cpp-2`.
- Model discovery for Ollama-style GGUF model directories.

## Quick start

Add this crate to your project:

```toml
infer = { git = "https://github.com/ontic-ai/infer", tag = "v0.1.0" }
```

Use the public API:

```rust
use infer::{auto_backend, BackendType, InferenceParams};

let model_path = std::path::PathBuf::from("/path/to/model.gguf");
let backend = auto_backend(&model_path, BackendType::auto_detect())?;
let params = InferenceParams {
    prompt: "Write a short Rust function that reverses a string.".to_string(),
    ..Default::default()
};
let response = backend.complete(&params)?;
println!("{}", response);
```

## Features

No features are enabled by default. CPU is the zero-config fallback (uses `MockBackend`
without the `llama` feature).

| Feature  | Backend                  | Requirement |
|----------|--------------------------|-------------|
| (none)   | CPU (MockBackend)        | None — compiles everywhere |
| `llama`  | CPU (real llama.cpp)     | C/C++ toolchain + clang (for bindgen) |
| `vulkan` | Vulkan GPU               | implies `llama`; Vulkan SDK + `VULKAN_SDK` env var on Windows |
| `cuda`   | CUDA GPU (legacy opt-in) | implies `llama`; CUDA toolkit installed |
| `metal`  | Metal GPU                | implies `llama`; macOS only |

See [BUILDING.md](BUILDING.md) for platform-specific build instructions.

## Structure

The crate exposes a single root module and keeps backend implementations
organized in `src/backend/`:

- `src/backend/mod.rs`
- `src/backend/llama.rs`
- `src/backend/mock.rs`

Other support modules include `discovery`, `manifest`, `chat_template`,
`registry`, and `error`.

## Testing

Run the unit tests with:

```sh
cargo test --no-default-features
```

To verify the default feature set:

```sh
cargo build
cargo clippy -- -D warnings
cargo fmt --check
```

## License

This project is licensed under the MIT license.
