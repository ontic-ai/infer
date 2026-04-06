//! Public error type for the `infer` crate.
//!
//! [`InferError`] is the single error type surfaced to all callers.
//! Internal [`crate::backend::BackendError`] values convert automatically
//! via the `From<BackendError>` implementation defined here.

use crate::backend::BackendError;

/// The single public error type for the `infer` crate.
///
/// All public functions return `Result<_, InferError>`. Internal
/// [`BackendError`] values are automatically converted via `From`.
#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// Ollama is not installed or its models directory is missing.
    #[error("ollama-not-installed: {0}")]
    OllamaNotInstalled(String),

    /// Ollama is installed but no models have been pulled.
    #[error("no-models-found: Ollama is installed but no models have been pulled")]
    NoModelsFound,

    /// A manifest file could not be parsed.
    #[error("manifest-parse-failure: {0}")]
    ManifestParseFailure(String),

    /// A model file could not be loaded.
    #[error("model-load-failure: {0}")]
    ModelLoadFailure(String),

    /// Text generation (complete) failed.
    #[error("inference-failure: {0}")]
    InferenceFailure(String),

    /// Embedding generation failed.
    #[error("embedding-failure: {0}")]
    EmbeddingFailure(String),

    /// Structured extraction failed.
    #[error("extraction-failure: {0}")]
    ExtractionFailure(String),

    /// A backend method was called before a model was loaded.
    #[error("backend-not-initialized: load a model before calling inference methods")]
    BackendNotInitialized,

    /// The requested backend is not available (e.g. feature not compiled in).
    #[error("backend-unavailable: {0}")]
    BackendUnavailable(String),

    /// A streaming operation failed.
    #[error("streaming-failure: {0}")]
    StreamingFailure(String),

    /// An I/O error occurred during discovery or manifest parsing.
    #[error("io-error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<BackendError> for InferError {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::ModelLoadFailed(msg) => InferError::ModelLoadFailure(msg),
            BackendError::InferenceFailed(msg) => InferError::InferenceFailure(msg),
            BackendError::EmbeddingFailed(msg) => InferError::EmbeddingFailure(msg),
            BackendError::ExtractionFailed(msg) => InferError::ExtractionFailure(msg),
            BackendError::NotInitialized => InferError::BackendNotInitialized,
            BackendError::StreamingFailed(msg) => InferError::StreamingFailure(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_ollama_not_installed_contains_variant_name() {
        let e = InferError::OllamaNotInstalled("dir missing".to_string());
        assert!(e.to_string().contains("ollama-not-installed"));
    }

    #[test]
    fn display_no_models_found_contains_variant_name() {
        assert!(
            InferError::NoModelsFound
                .to_string()
                .contains("no-models-found")
        );
    }

    #[test]
    fn display_manifest_parse_failure_contains_variant_name() {
        let e = InferError::ManifestParseFailure("bad json".to_string());
        assert!(e.to_string().contains("manifest-parse-failure"));
    }

    #[test]
    fn display_model_load_failure_contains_variant_name() {
        let e = InferError::ModelLoadFailure("missing file".to_string());
        assert!(e.to_string().contains("model-load-failure"));
    }

    #[test]
    fn display_inference_failure_contains_variant_name() {
        let e = InferError::InferenceFailure("oom".to_string());
        assert!(e.to_string().contains("inference-failure"));
    }

    #[test]
    fn display_backend_not_initialized_contains_variant_name() {
        assert!(
            InferError::BackendNotInitialized
                .to_string()
                .contains("backend-not-initialized")
        );
    }

    #[test]
    fn display_streaming_failure_contains_variant_name() {
        let e = InferError::StreamingFailure("channel closed".to_string());
        assert!(e.to_string().contains("streaming-failure"));
    }

    #[test]
    fn backend_error_not_initialized_converts() {
        let e: InferError = BackendError::NotInitialized.into();
        assert!(matches!(e, InferError::BackendNotInitialized));
    }

    #[test]
    fn backend_error_model_load_failed_converts() {
        let e: InferError = BackendError::ModelLoadFailed("x".to_string()).into();
        assert!(matches!(e, InferError::ModelLoadFailure(_)));
    }

    #[test]
    fn backend_error_streaming_failed_converts() {
        let e: InferError = BackendError::StreamingFailed("x".to_string()).into();
        assert!(matches!(e, InferError::StreamingFailure(_)));
    }
}
