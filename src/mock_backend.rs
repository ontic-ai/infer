//! Deterministic mock backend for testing.
//!
//! [`MockBackend`] implements [`InferenceBackend`] with configurable canned
//! responses. The path passed to `load_model` is accepted but not used unless
//! `fail_load` is set. All outputs are deterministic for a given
//! [`MockConfig`], making this suitable for unit and integration tests.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use crate::backend::{BackendType, ExtractionResult, InferenceBackend, InferenceParams};
use crate::error::InferError;

/// Configuration for a [`MockBackend`] instance.
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// Response text returned by [`MockBackend::complete`].
    pub infer_response: String,
    /// Embedding vector returned by [`MockBackend::embed`].
    pub embed_vector: Vec<f32>,
    /// Facts returned by [`MockBackend::extract`].
    pub extract_facts: Vec<String>,
    /// When `true`, [`MockBackend::load_model`] returns an error.
    pub fail_load: bool,
    /// When `true`, all inference/embed/extract/stream calls return errors.
    pub fail_inference: bool,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            infer_response: "Mock inference response".to_string(),
            embed_vector: vec![0.1; 384],
            extract_facts: vec!["fact1".to_string(), "fact2".to_string()],
            fail_load: false,
            fail_inference: false,
        }
    }
}

/// Deterministic mock backend for unit and integration testing.
///
/// Call counters track how many times each inference method was invoked.
pub struct MockBackend {
    config: MockConfig,
    loaded: bool,
    loaded_backend_type: BackendType,
    model_name_loaded: Option<String>,
    /// Number of times [`complete`](MockBackend::complete) has been called.
    pub infer_call_count: AtomicUsize,
    /// Number of times [`embed`](MockBackend::embed) has been called.
    pub embed_call_count: AtomicUsize,
    /// Number of times [`extract`](MockBackend::extract) has been called.
    pub extract_call_count: AtomicUsize,
}

impl MockBackend {
    /// Create a mock backend with the default configuration.
    pub fn new() -> Self {
        Self::with_config(MockConfig::default())
    }

    /// Create a mock backend with a custom configuration.
    pub fn with_config(config: MockConfig) -> Self {
        Self {
            config,
            loaded: false,
            loaded_backend_type: BackendType::Cpu,
            model_name_loaded: None,
            infer_call_count: AtomicUsize::new(0),
            embed_call_count: AtomicUsize::new(0),
            extract_call_count: AtomicUsize::new(0),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn load_model(
        &mut self,
        model_path: &Path,
        backend_type: BackendType,
    ) -> Result<(), InferError> {
        if self.config.fail_load {
            return Err(InferError::ModelLoadFailure(
                "mock load failure".to_string(),
            ));
        }
        self.loaded = true;
        self.loaded_backend_type = backend_type;
        self.model_name_loaded = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string);
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn backend_type(&self) -> BackendType {
        self.loaded_backend_type
    }

    fn model_name(&self) -> Option<&str> {
        self.model_name_loaded.as_deref()
    }

    fn complete(&self, _params: &InferenceParams) -> Result<String, InferError> {
        if !self.loaded {
            return Err(InferError::BackendNotInitialized);
        }
        if self.config.fail_inference {
            return Err(InferError::InferenceFailure(
                "mock inference failure".to_string(),
            ));
        }
        self.infer_call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.config.infer_response.clone())
    }

    /// Stream tokens by splitting the fixed response into words.
    ///
    /// Each word (except the first) is prefixed with a space so that
    /// `tokens.collect::<String>()` reproduces the original response exactly.
    /// The output is fully deterministic: same configuration → same token sequence.
    fn stream(&self, _params: InferenceParams) -> Result<mpsc::Receiver<String>, InferError> {
        if !self.loaded {
            return Err(InferError::BackendNotInitialized);
        }
        if self.config.fail_inference {
            return Err(InferError::StreamingFailure(
                "mock streaming failure".to_string(),
            ));
        }

        let response = self.config.infer_response.clone();
        let (tx, rx) = mpsc::channel::<String>();

        for (i, word) in response.split_whitespace().enumerate() {
            let token = if i == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };
            if tx.send(token).is_err() {
                break; // receiver dropped — clean cancellation
            }
        }
        // tx dropped here; channel closes after the last token.
        Ok(rx)
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, InferError> {
        if !self.loaded {
            return Err(InferError::BackendNotInitialized);
        }
        if self.config.fail_inference {
            return Err(InferError::EmbeddingFailure(
                "mock embedding failure".to_string(),
            ));
        }
        self.embed_call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.config.embed_vector.clone())
    }

    fn extract(&self, _text: &str) -> Result<ExtractionResult, InferError> {
        if !self.loaded {
            return Err(InferError::BackendNotInitialized);
        }
        if self.config.fail_inference {
            return Err(InferError::ExtractionFailure(
                "mock extraction failure".to_string(),
            ));
        }
        self.extract_call_count.fetch_add(1, Ordering::Relaxed);
        Ok(ExtractionResult {
            facts: self.config.extract_facts.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn loaded() -> MockBackend {
        let mut m = MockBackend::new();
        m.load_model(&PathBuf::from("/test/model.gguf"), BackendType::Cpu)
            .expect("load");
        m
    }

    #[test]
    fn new_backend_not_loaded() {
        assert!(!MockBackend::new().is_loaded());
    }

    #[test]
    fn load_marks_as_loaded() {
        assert!(loaded().is_loaded());
    }

    #[test]
    fn complete_returns_expected_response() {
        let params = InferenceParams {
            prompt: "hello".to_string(),
            ..InferenceParams::default()
        };
        assert_eq!(
            loaded().complete(&params).expect("complete"),
            "Mock inference response"
        );
    }

    #[test]
    fn embed_returns_correct_dimension() {
        let v = loaded().embed("text").expect("embed");
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn extract_returns_expected_facts() {
        let r = loaded().extract("text").expect("extract");
        assert_eq!(r.facts, vec!["fact1", "fact2"]);
    }

    #[test]
    fn stream_tokens_concatenate_to_expected_response() {
        let params = InferenceParams {
            prompt: "hello".to_string(),
            ..InferenceParams::default()
        };
        let rx = loaded().stream(params).expect("stream");
        let collected: String = rx.iter().collect();
        assert_eq!(collected, "Mock inference response");
    }

    #[test]
    fn stream_yields_multiple_tokens() {
        let rx = loaded().stream(InferenceParams::default()).expect("stream");
        let tokens: Vec<String> = rx.iter().collect();
        // "Mock inference response" splits into 3 words
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn methods_fail_before_load() {
        let m = MockBackend::new();
        let p = InferenceParams::default();
        assert!(matches!(
            m.complete(&p),
            Err(InferError::BackendNotInitialized)
        ));
        assert!(matches!(
            m.embed("t"),
            Err(InferError::BackendNotInitialized)
        ));
        assert!(matches!(
            m.extract("t"),
            Err(InferError::BackendNotInitialized)
        ));
        assert!(matches!(
            m.stream(InferenceParams::default()),
            Err(InferError::BackendNotInitialized)
        ));
    }

    #[test]
    fn fail_inference_returns_errors() {
        let mut m = MockBackend::with_config(MockConfig {
            fail_inference: true,
            ..MockConfig::default()
        });
        m.load_model(&PathBuf::from("/test/m.gguf"), BackendType::Cpu)
            .expect("load");
        assert!(m.complete(&InferenceParams::default()).is_err());
        assert!(m.embed("t").is_err());
        assert!(m.extract("t").is_err());
    }

    #[test]
    fn fail_load_returns_error() {
        let mut m = MockBackend::with_config(MockConfig {
            fail_load: true,
            ..MockConfig::default()
        });
        assert!(
            m.load_model(&PathBuf::from("/test/m.gguf"), BackendType::Cpu)
                .is_err()
        );
        assert!(!m.is_loaded());
    }

    #[test]
    fn model_name_extracted_from_path_stem() {
        assert_eq!(loaded().model_name(), Some("model"));
    }

    #[test]
    fn backend_type_reflects_loaded_type() {
        let mut m = MockBackend::new();
        m.load_model(&PathBuf::from("/test/m.gguf"), BackendType::Metal)
            .expect("load");
        assert_eq!(m.backend_type(), BackendType::Metal);
    }
}
