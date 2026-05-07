//! Dedicated embedding backend for GGUF embedding models.
//!
//! [`LlamaEmbedBackend`] loads a GGUF model with embeddings enabled (mean
//! pooling) and implements [`InferenceBackend::embed`].  It reuses the
//! process-global llama.cpp C runtime that is owned by [`super::llama`].
//!
//! This module is only compiled when the `llama` feature is enabled.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, mpsc};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::backend::{BackendType, ExtractionResult, InferenceBackend, InferenceParams};
use crate::error::InferError;

// ---------------------------------------------------------------------------
// LlamaEmbedBackend
// ---------------------------------------------------------------------------

/// Embedding-only backend using a GGUF model loaded with mean-pooling.
///
/// Call [`LlamaEmbedBackend::load`] to construct an initialized instance.
/// Only [`InferenceBackend::embed`] is meaningful; all other inference methods
/// return [`InferError::InferenceFailure`].
pub struct LlamaEmbedBackend {
    model: Arc<LlamaModel>,
    model_name: Option<String>,
    backend_type: BackendType,
}

impl LlamaEmbedBackend {
    /// Load a GGUF embedding model from `model_path`.
    ///
    /// Requires the llama.cpp runtime to already be initialized (call
    /// [`crate::LlamaBackend::new()`] first, or call this before the
    /// generation backend — either order works as long as one of them
    /// initializes the runtime).
    ///
    /// # Errors
    ///
    /// Returns [`InferError::BackendUnavailable`] if the runtime is not
    /// available, or [`InferError::ModelLoadFailure`] if the model file
    /// cannot be loaded.
    pub fn load(model_path: &Path, backend_type: BackendType) -> Result<Self, InferError> {
        let runtime = super::llama::LlamaBackend::global_runtime()?;

        let n_gpu_layers: u32 = match backend_type {
            BackendType::Cpu => 0,
            _ => 1000,
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model =
            LlamaModel::load_from_file(runtime, model_path, &model_params).map_err(|e| {
                InferError::ModelLoadFailure(format!(
                    "embed: failed to load {}: {e}",
                    model_path.display()
                ))
            })?;

        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string);

        Ok(Self {
            model: Arc::new(model),
            model_name,
            backend_type,
        })
    }

    fn embed_sync(model: &LlamaModel, text: &str) -> Result<Vec<f32>, InferError> {
        let runtime = super::llama::LlamaBackend::global_runtime()?;

        let n_ctx = NonZeroU32::new(512).expect("constant 512 is nonzero");
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_n_ctx(Some(n_ctx));

        let mut ctx = model
            .new_context(runtime, ctx_params)
            .map_err(|e| InferError::EmbeddingFailure(format!("embed: context init: {e}")))?;

        let tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| InferError::EmbeddingFailure(format!("embed: tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(InferError::EmbeddingFailure(
                "embed: empty token list after tokenization".to_string(),
            ));
        }

        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], true)
                .map_err(|e| InferError::EmbeddingFailure(format!("embed: batch add: {e}")))?;
        }

        ctx.encode(&mut batch)
            .map_err(|e| InferError::EmbeddingFailure(format!("embed: encode: {e}")))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| InferError::EmbeddingFailure(format!("embed: extract: {e}")))?;

        Ok(embedding.to_vec())
    }
}

impl InferenceBackend for LlamaEmbedBackend {
    fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    fn load_model(&mut self, _path: &Path, _backend_type: BackendType) -> Result<(), InferError> {
        Err(InferError::InferenceFailure(
            "LlamaEmbedBackend: use LlamaEmbedBackend::load() to construct, not load_model()"
                .to_string(),
        ))
    }

    fn complete(&self, _params: &InferenceParams) -> Result<String, InferError> {
        Err(InferError::InferenceFailure(
            "LlamaEmbedBackend is an embedding-only backend".to_string(),
        ))
    }

    fn stream(&self, _params: InferenceParams) -> Result<mpsc::Receiver<String>, InferError> {
        Err(InferError::StreamingFailure(
            "LlamaEmbedBackend is an embedding-only backend".to_string(),
        ))
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, InferError> {
        Self::embed_sync(&self.model, text)
    }

    fn extract(&self, _text: &str) -> Result<ExtractionResult, InferError> {
        Err(InferError::InferenceFailure(
            "LlamaEmbedBackend is an embedding-only backend".to_string(),
        ))
    }
}
