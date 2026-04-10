//! Universal inference engine for the Ontic ecosystem.
//!
//! # Overview
//!
//! The `infer` crate provides a single, unified [`InferenceBackend`] trait
//! that every inference backend implements. The two built-in implementations
//! are:
//!
//! - **[`LlamaBackend`]** — production backend built on llama-cpp-2 for
//!   running GGUF models locally (enabled by the default `llama` feature).
//! - **[`MockBackend`]** — deterministic test backend with configurable
//!   canned responses and call counters.
//!
//! # Quick start
//!
//! ```no_run
//! use infer::{auto_backend, BackendType};
//!
//! let path = std::path::PathBuf::from("/models/llama-3.gguf");
//! let backend = auto_backend(&path, BackendType::auto_detect()).expect("backend");
//!
//! let params = infer::InferenceParams {
//!     prompt: "The capital of France is".to_string(),
//!     ..Default::default()
//! };
//! let response = backend.complete(&params).expect("inference");
//! println!("{response}");
//! ```
//!
//! # Feature flags
//!
//! | Feature  | Default | Description |
//! |----------|---------|-------------|
//! | `llama`  |         | Enable llama-cpp-2 backend (GGUF inference) |
//! | `vulkan` |         | Enable Vulkan GPU offloading — Windows/Linux, cross-vendor (implies `llama`) |
//! | `cuda`   |         | Enable NVIDIA CUDA GPU offloading — opt-in legacy (implies `llama`) |
//! | `metal`  |         | Enable Apple Metal GPU offloading — macOS only (implies `llama`) |
//! | (none)   | ✓       | CPU fallback — no features required; uses `MockBackend` without `llama` |

// Module declarations
pub mod backend;
pub mod chat_template;
pub mod discovery;
pub mod error;
pub mod manifest;
pub mod registry;

// ---------------------------------------------------------------------------
// Re-exports: backend types
// ---------------------------------------------------------------------------

pub use backend::{BackendError, BackendType, ExtractionResult, InferenceBackend, InferenceParams};

// ---------------------------------------------------------------------------
// Re-exports: chat template
// ---------------------------------------------------------------------------

pub use chat_template::ChatTemplate;

// ---------------------------------------------------------------------------
// Re-exports: model discovery
// ---------------------------------------------------------------------------

pub use discovery::{auto_backend, discover_models, ollama_models_dir};

// ---------------------------------------------------------------------------
// Re-exports: error
// ---------------------------------------------------------------------------

pub use error::InferError;

// ---------------------------------------------------------------------------
// Re-exports: mock backend
// ---------------------------------------------------------------------------

pub use backend::mock::{MockBackend, MockConfig};

// ---------------------------------------------------------------------------
// Re-exports: model registry
// ---------------------------------------------------------------------------

pub use registry::{ModelInfo, ModelRegistry, Quantization};

// ---------------------------------------------------------------------------
// Re-exports: llama backend (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "llama")]
pub use backend::llama::LlamaBackend;
#[cfg(feature = "llama")]
pub use backend::llama::suppress_logs as suppress_llama_logs;
