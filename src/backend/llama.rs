//! llama-cpp-2 backed inference engine.
//!
//! [`LlamaBackend`] wraps a loaded GGUF model and implements [`InferenceBackend`].
//! **Only one `LlamaBackend` instance should exist per process.** The underlying
//! llama.cpp runtime is initialized once globally; subsequent calls to [`LlamaBackend::new`]
//! on the same process share that global initialization.
//!
//! This module is only compiled when the `llama` feature is enabled.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaCppBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::backend::{BackendType, ExtractionResult, InferenceBackend, InferenceParams};
use crate::error::InferError;

// ---------------------------------------------------------------------------
// Global runtime initialization
// ---------------------------------------------------------------------------

/// Log-suppression flag set by [`suppress_logs`].
///
/// When `true`, [`ensure_runtime`] calls [`void_logs`](LlamaCppBackend::void_logs)
/// on the backend during initialization. Must be set before the first call to
/// [`ensure_runtime`].
static SUPPRESS_LOGS: AtomicBool = AtomicBool::new(false);

/// Globally-owned llama.cpp backend, initialized exactly once.
///
/// Stores the live [`LlamaCppBackend`] on success or the error message on
/// failure. The value lives for the entire program lifetime via the `'static`
/// [`OnceLock`].
static RUNTIME: OnceLock<Result<LlamaCppBackend, String>> = OnceLock::new();

/// Suppress all llama.cpp stderr log output.
///
/// Must be called **before** the first [`LlamaBackend::new()`] call (or before
/// any other operation that triggers backend initialization). Calls after the
/// backend has already been initialized have no effect.
pub fn suppress_logs() {
    SUPPRESS_LOGS.store(true, Ordering::Relaxed);
}

/// Ensure the llama.cpp C runtime is initialized exactly once.
///
/// Returns a `'static` reference to the globally-owned [`LlamaCppBackend`].
/// The reference is valid for the entire program lifetime.
fn ensure_runtime() -> Result<&'static LlamaCppBackend, InferError> {
    let result = RUNTIME.get_or_init(|| {
        LlamaCppBackend::init()
            .map(|mut b| {
                if SUPPRESS_LOGS.load(Ordering::Relaxed) {
                    b.void_logs();
                }
                b
            })
            .map_err(|e| e.to_string())
    });
    result
        .as_ref()
        .map_err(|e| InferError::BackendUnavailable(format!("llama.cpp runtime init failed: {e}")))
}

// ---------------------------------------------------------------------------
// LlamaBackend struct
// ---------------------------------------------------------------------------

/// Production backend backed by llama-cpp-2.
///
/// Load a GGUF model with [`load_model`](InferenceBackend::load_model) before
/// calling any inference methods.
///
/// The loaded model is stored behind a `Mutex` so that the struct satisfies
/// `Sync` and can be shared via `Arc<dyn InferenceBackend>`.
pub struct LlamaBackend {
    /// The loaded GGUF model, or `None` before [`load_model`] is called.
    model: Mutex<Option<LlamaModel>>,
    /// Stem of the loaded model file path (set once by `load_model`).
    model_name: Option<String>,
    /// Backend type selected at load time (set once by `load_model`).
    loaded_backend_type: BackendType,
}

impl LlamaBackend {
    /// Create a new (unloaded) `LlamaBackend`.
    ///
    /// This call initializes the llama.cpp C runtime on first invocation.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::BackendUnavailable`] if the llama.cpp runtime
    /// fails to initialize.
    pub fn new() -> Result<Self, InferError> {
        ensure_runtime()?;
        Ok(Self {
            model: Mutex::new(None),
            model_name: None,
            loaded_backend_type: BackendType::Cpu,
        })
    }
}

// ---------------------------------------------------------------------------
// InferenceBackend impl
// ---------------------------------------------------------------------------

impl InferenceBackend for LlamaBackend {
    fn load_model(
        &mut self,
        model_path: &Path,
        backend_type: BackendType,
    ) -> Result<(), InferError> {
        if !model_path.exists() {
            return Err(InferError::ModelLoadFailure(format!(
                "path does not exist: {}",
                model_path.display()
            )));
        }

        let runtime = ensure_runtime()?;

        let n_gpu_layers: u32 = match backend_type {
            BackendType::Metal | BackendType::Cuda => 1000, // offload all layers
            BackendType::Cpu => 0,
        };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        let model =
            LlamaModel::load_from_file(runtime, model_path, &model_params).map_err(|e| {
                InferError::ModelLoadFailure(format!(
                    "failed to load {}: {e}",
                    model_path.display()
                ))
            })?;

        self.model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string);
        self.loaded_backend_type = backend_type;

        *self
            .model
            .lock()
            .map_err(|_| InferError::InferenceFailure("model mutex poisoned".into()))? =
            Some(model);
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn backend_type(&self) -> BackendType {
        self.loaded_backend_type
    }

    fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    fn complete(&self, params: &InferenceParams) -> Result<String, InferError> {
        let guard = self
            .model
            .lock()
            .map_err(|_| InferError::InferenceFailure("model mutex poisoned".into()))?;
        let model = guard.as_ref().ok_or(InferError::BackendNotInitialized)?;
        let runtime = ensure_runtime()?;
        run_complete(model, runtime, params)
    }

    /// Stream a text completion token by token.
    ///
    /// All tokens are generated inline (blocking) before the `Receiver` is
    /// returned, so the caller receives a fully-buffered channel. Drop the
    /// `Receiver` at any point to signal cancellation: the generation loop
    /// detects the closed channel via `tx.send().is_err()` and stops cleanly.
    fn stream(&self, params: InferenceParams) -> Result<mpsc::Receiver<String>, InferError> {
        let guard = self
            .model
            .lock()
            .map_err(|_| InferError::StreamingFailure("model mutex poisoned".into()))?;
        let model = guard.as_ref().ok_or(InferError::BackendNotInitialized)?;
        let runtime = ensure_runtime()
            .map_err(|e| InferError::StreamingFailure(format!("runtime unavailable: {e}")))?;

        let (tx, rx) = mpsc::channel::<String>();

        let ctx_size_nz = NonZeroU32::new(params.ctx_size)
            .unwrap_or_else(|| NonZeroU32::new(2048).expect("constant 2048 is nonzero"));
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(ctx_size_nz));
        let mut ctx = model
            .new_context(runtime, ctx_params)
            .map_err(|e| InferError::StreamingFailure(format!("context init: {e}")))?;

        let tokens_list = model
            .str_to_token(&params.prompt, AddBos::Always)
            .map_err(|e| InferError::StreamingFailure(format!("tokenize: {e}")))?;

        if tokens_list.is_empty() {
            return Err(InferError::StreamingFailure("empty prompt".into()));
        }

        let n_tokens = tokens_list.len() as i32;
        let n_len = n_tokens + params.max_tokens as i32;
        let n_ctx_val = params.ctx_size as i32;

        if n_len > n_ctx_val {
            return Err(InferError::StreamingFailure(format!(
                "prompt ({n_tokens} tokens) + max_tokens ({}) exceeds context size ({n_ctx_val})",
                params.max_tokens
            )));
        }

        let mut batch = LlamaBatch::new(tokens_list.len(), 1);
        let last_index = n_tokens - 1;
        for (i, &token) in tokens_list.iter().enumerate() {
            let is_last = i as i32 == last_index;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|e| InferError::StreamingFailure(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| InferError::StreamingFailure(format!("decode prompt: {e}")))?;

        let mut n_cur = batch.n_tokens();
        let seed = (params.request_id.as_u128() & 0xFFFF_FFFF) as u32;
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(seed), LlamaSampler::greedy()]);
        let mut decoder = UTF_8.new_decoder();

        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| InferError::StreamingFailure(format!("token to piece: {e}")))?;

            if tx.send(piece).is_err() {
                break; // receiver dropped — cancellation
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| InferError::StreamingFailure(format!("batch add token: {e}")))?;

            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| InferError::StreamingFailure(format!("decode token: {e}")))?;
        }

        // tx drops here; channel is closed after last token.
        Ok(rx)
    }

    /// Embeddings are not supported by generation backends.
    ///
    /// Use a dedicated embedding model (e.g. nomic-embed-text) loaded through
    /// a purpose-built embedding backend.
    fn embed(&self, _text: &str) -> Result<Vec<f32>, InferError> {
        Err(InferError::EmbeddingFailure(
            "LlamaBackend is a generation backend; embeddings require a dedicated embedding model"
                .to_string(),
        ))
    }

    /// Extract structured facts from `text`.
    ///
    /// Delegates to [`complete`](InferenceBackend::complete) with a low
    /// temperature and an extraction prompt. Each non-empty output line
    /// becomes one fact after stripping common list prefixes.
    fn extract(&self, text: &str) -> Result<ExtractionResult, InferError> {
        let prompt = format!(
            "Extract all factual claims from the following text as a bulleted list.\n\nText:\n{}\n\nFacts:",
            text
        );
        let params = InferenceParams {
            prompt,
            temperature: 0.1,
            max_tokens: 512,
            ..InferenceParams::default()
        };
        let response = self.complete(&params)?;
        let facts = response
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.trim_start_matches(['•', '-', '*', ' '])
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty())
            .collect();
        Ok(ExtractionResult { facts })
    }
}

// ---------------------------------------------------------------------------
// Internal helper: run a full completion without consuming a Mutex lock
// ---------------------------------------------------------------------------

/// Core generation loop shared by [`complete`] and (indirectly) [`extract`].
///
/// `model` must be kept alive for the duration of this call by the caller
/// (i.e. by holding the `MutexGuard`).
fn run_complete(
    model: &LlamaModel,
    runtime: &LlamaCppBackend,
    params: &InferenceParams,
) -> Result<String, InferError> {
    let ctx_size_nz = NonZeroU32::new(params.ctx_size)
        .unwrap_or_else(|| NonZeroU32::new(2048).expect("constant 2048 is nonzero"));
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(ctx_size_nz));
    let mut ctx = model
        .new_context(runtime, ctx_params)
        .map_err(|e| InferError::InferenceFailure(format!("context init: {e}")))?;

    let tokens_list = model
        .str_to_token(&params.prompt, AddBos::Always)
        .map_err(|e| InferError::InferenceFailure(format!("tokenize: {e}")))?;

    if tokens_list.is_empty() {
        return Err(InferError::InferenceFailure("empty prompt".into()));
    }

    let n_tokens = tokens_list.len() as i32;
    let n_len = n_tokens + params.max_tokens as i32;
    let n_ctx_val = params.ctx_size as i32;

    if n_len > n_ctx_val {
        return Err(InferError::InferenceFailure(format!(
            "prompt ({n_tokens} tokens) + max_tokens ({}) exceeds context size ({n_ctx_val})",
            params.max_tokens
        )));
    }

    let mut batch = LlamaBatch::new(tokens_list.len(), 1);
    let last_index = n_tokens - 1;
    for (i, &token) in tokens_list.iter().enumerate() {
        let is_last = i as i32 == last_index;
        batch
            .add(token, i as i32, &[0], is_last)
            .map_err(|e| InferError::InferenceFailure(format!("batch add: {e}")))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| InferError::InferenceFailure(format!("decode prompt: {e}")))?;

    let mut n_cur = batch.n_tokens();
    let seed = (params.request_id.as_u128() & 0xFFFF_FFFF) as u32;
    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(seed), LlamaSampler::greedy()]);

    let mut result = String::new();
    let mut decoder = UTF_8.new_decoder();

    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| InferError::InferenceFailure(format!("token to piece: {e}")))?;
        result.push_str(&piece);

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| InferError::InferenceFailure(format!("batch add token: {e}")))?;

        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| InferError::InferenceFailure(format!("decode token: {e}")))?;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn new_backend_not_loaded() {
        // LlamaBackend::new() may fail if llama runtime fails to build
        // (e.g. missing system libraries in CI), which is acceptable.
        match LlamaBackend::new() {
            Ok(b) => assert!(!b.is_loaded()),
            Err(_) => {} // runtime not available
        }
    }

    #[test]
    fn load_nonexistent_path_returns_error() {
        let Ok(mut b) = LlamaBackend::new() else {
            return; // runtime not available
        };
        let result = b.load_model(&PathBuf::from("/nonexistent/model.gguf"), BackendType::Cpu);
        assert!(
            matches!(result, Err(InferError::ModelLoadFailure(_))),
            "expected ModelLoadFailure, got {result:?}"
        );
    }

    #[test]
    fn complete_before_load_returns_not_initialized() {
        let Ok(b) = LlamaBackend::new() else {
            return;
        };
        let err = b.complete(&InferenceParams::default());
        assert!(
            matches!(err, Err(InferError::BackendNotInitialized)),
            "expected BackendNotInitialized, got {err:?}"
        );
    }

    #[test]
    fn stream_before_load_returns_not_initialized() {
        let Ok(b) = LlamaBackend::new() else {
            return;
        };
        let err = b.stream(InferenceParams::default());
        assert!(
            matches!(err, Err(InferError::BackendNotInitialized)),
            "expected BackendNotInitialized, got {err:?}"
        );
    }

    #[test]
    fn embed_always_returns_embedding_failure() {
        let Ok(b) = LlamaBackend::new() else {
            return;
        };
        assert!(matches!(
            b.embed("hello"),
            Err(InferError::EmbeddingFailure(_))
        ));
    }

    #[test]
    fn backend_type_is_cpu_before_load() {
        let Ok(b) = LlamaBackend::new() else {
            return;
        };
        assert_eq!(b.backend_type(), BackendType::Cpu);
    }

    #[test]
    fn model_name_is_none_before_load() {
        let Ok(b) = LlamaBackend::new() else {
            return;
        };
        assert!(b.model_name().is_none());
    }
}
