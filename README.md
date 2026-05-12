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
infer = { git = "https://github.com/ontic-ai/infer", tag = "v0.3.2" }
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
| `turboquant` | CPU/reference KV TurboQuant | pulls the upstream `turboquant` + `ort` stack |
| `llama`  | CPU (real llama.cpp)     | C/C++ toolchain + clang (for bindgen) |
| `vulkan` | Vulkan GPU               | implies `llama`; Vulkan SDK + `VULKAN_SDK` env var on Windows |
| `cuda`   | CUDA GPU (legacy opt-in) | implies `llama`; CUDA toolkit installed |
| `metal`  | Metal GPU                | implies `llama`; macOS only |

See [BUILDING.md](BUILDING.md) for platform-specific build instructions.

## KV Cache Quantization

The crate exposes KV cache quantization controls through `KvCacheConfig` and
the algorithm-oriented `KvQuantization` API. Today that surface covers two
families:

- RotorQuant-style algorithms already implemented in this crate.
- TurboQuant algorithms wrapped from the upstream `turboquant` crate when the `turboquant` feature is enabled.

Current backend support boundary:

- RotorQuant-style variants can flow through the current llama.cpp native KV dtype mapping.
- TurboQuant variants are available in the CPU/reference pipeline and benchmarks, but the current llama.cpp integration does not yet expose the software-managed KV hooks needed to run them in production inference.

Enable TurboQuant explicitly when you need that path:

```toml
infer = { version = "0.3.2", features = ["turboquant"] }
```

Reference presets:

- `KvCacheConfig::none()`
- `KvCacheConfig::deferred_k()`
- `KvCacheConfig::symmetric_3bit()`
- `KvCacheConfig::turbo_balanced_4bit()`
- `KvCacheConfig::turbo_compact_3bit()`

### KV Quantization Use Cases

| Algorithm | Example config | Best use case | Quality focus | Runtime support today | Notes |
|-----------|----------------|---------------|---------------|-----------------------|-------|
| `None` | `KvCacheConfig::none()` | Maximum fidelity, enough VRAM | Exact FP16 K/V | All backends | Baseline for comparisons |
| Rotor `Planar2` | `k=KvQuantization::planar2()` | Conservative memory reduction with low risk | Reconstruction quality | Native llama dtype mapping | Good first step when VRAM pressure is mild |
| Rotor `Planar3` | `KvCacheConfig::deferred_k()` | Quantize keys first under moderate VRAM pressure | Balanced compression | Native llama dtype mapping | Current recommended default preset |
| Rotor `Iso4` | `k=KvQuantization::iso4()` | 4-bit RotorQuant with better quality than `Planar3` | Reconstruction quality | Native llama dtype mapping | Better quality-per-bit than 3-bit planar in current in-repo family |
| Rotor `Iso3` | `KvCacheConfig::symmetric_3bit()` | Aggressive memory savings for long contexts | Compression efficiency | Native llama dtype mapping | Strongest current in-repo compression preset |
| TurboQuant `Mse` | `v=KvQuantization::turbo_mse(4)` | Evaluate reconstruction-oriented TurboQuant behavior | MSE / value reconstruction | CPU/reference pipeline only | Aligns with upstream `turboquant::kv_cache::QuantStrategy::MSE` |
| TurboQuant `Prod` | `k=KvQuantization::turbo_prod(4)` | Evaluate attention-score-sensitive key compression | Inner product fidelity | CPU/reference pipeline only | Aligns with upstream `turboquant::kv_cache::QuantStrategy::Prod` |

API comparison against upstream `turboquant::kv_cache`:

- `infer` keeps quantization declarative and request-scoped through `InferenceParams::kv_cache` and backend defaults.
- Upstream `turboquant::kv_cache` owns the full software-managed cache lifecycle with append, stats, sliding-window eviction, and multi-head cache types.
- `infer` now wraps TurboQuant algorithms, but it still needs lower-level KV hooks in the llama backend to expose the same software-managed cache lifecycle in production inference.

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
