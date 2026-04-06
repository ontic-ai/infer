//! Model registry: [`ModelInfo`], [`Quantization`], and [`ModelRegistry`].
//!
//! These types are fully owned by this crate with no dependency on any bus,
//! runtime, or framework type.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Metadata for a single discovered GGUF model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Human-readable model name, e.g. `"llama2"` or `"gemma:latest"`.
    pub name: String,
    /// Absolute path to the GGUF blob file on disk.
    pub path: PathBuf,
    /// Size of the GGUF file in bytes.
    pub size_bytes: u64,
    /// Quantization level parsed from the model name.
    pub quantization: Quantization,
}

/// Quantization level of a GGUF model.
///
/// Parsed heuristically from the model name. Falls back to
/// [`Quantization::Unknown`] when no recognised pattern is found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Quantization {
    /// 4-bit quantization, variant 0.
    Q4_0,
    /// 4-bit quantization, variant 1.
    Q4_1,
    /// 5-bit quantization, variant 0.
    Q5_0,
    /// 5-bit quantization, variant 1.
    Q5_1,
    /// 8-bit quantization, variant 0.
    Q8_0,
    /// 16-bit floating point.
    F16,
    /// 32-bit floating point (full precision).
    F32,
    /// Unrecognised quantization string preserved verbatim.
    Unknown(String),
}

/// Registry of discovered GGUF models.
///
/// Holds all models found during discovery. The default model is the one with
/// the largest file size unless overridden via
/// [`set_preferred_model`](ModelRegistry::set_preferred_model).
#[derive(Debug)]
pub struct ModelRegistry {
    models: Vec<ModelInfo>,
    default_model: Option<String>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            default_model: None,
        }
    }

    /// Build a registry from a list of models, selecting the largest as the default.
    pub fn from_models(models: Vec<ModelInfo>) -> Self {
        let default_model = models
            .iter()
            .max_by_key(|m| m.size_bytes)
            .map(|m| m.name.clone());
        Self {
            models,
            default_model,
        }
    }

    /// Add a model to the registry.
    ///
    /// The default model selection is **not** updated automatically.
    pub fn add_model(&mut self, model: ModelInfo) {
        self.models.push(model);
    }

    /// Return the name of the default model, or `None` if the registry is empty.
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// Return `true` if the registry contains no models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Return the number of models in the registry.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Return a slice of all models in the registry.
    pub fn models(&self) -> &[ModelInfo] {
        &self.models
    }

    /// Find a model by name (case-insensitive).
    pub fn find_by_name(&self, name: &str) -> Option<&ModelInfo> {
        self.models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// Override the default model with the given preferred name.
    ///
    /// If `preferred` is not found in the registry, the current default is kept.
    pub fn set_preferred_model(&mut self, preferred: &str) {
        if let Some(name) = self
            .models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(preferred))
            .map(|m| m.name.clone())
        {
            self.default_model = Some(name);
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(name: &str, size: u64) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            path: PathBuf::from(format!("/models/{}.gguf", name)),
            size_bytes: size,
            quantization: Quantization::Q4_0,
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let r = ModelRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.model_count(), 0);
        assert_eq!(r.default_model(), None);
    }

    #[test]
    fn from_models_selects_largest_as_default() {
        let r = ModelRegistry::from_models(vec![
            make("small", 1_000_000_000),
            make("large", 5_000_000_000),
            make("medium", 3_000_000_000),
        ]);
        assert_eq!(r.default_model(), Some("large"));
        assert_eq!(r.model_count(), 3);
    }

    #[test]
    fn single_model_selected_as_default() {
        let r = ModelRegistry::from_models(vec![make("only", 2_000_000_000)]);
        assert_eq!(r.default_model(), Some("only"));
    }

    #[test]
    fn empty_models_no_default() {
        let r = ModelRegistry::from_models(vec![]);
        assert_eq!(r.default_model(), None);
    }

    #[test]
    fn add_model_increments_count() {
        let mut r = ModelRegistry::new();
        r.add_model(make("m1", 1_000_000_000));
        assert_eq!(r.model_count(), 1);
        // add_model does not update default
        assert_eq!(r.default_model(), None);
    }

    #[test]
    fn find_by_name_case_insensitive() {
        let r = ModelRegistry::from_models(vec![make("LlamaTwo", 1_000_000_000)]);
        assert!(r.find_by_name("llamatwo").is_some());
        assert!(r.find_by_name("LLAMATWO").is_some());
        assert!(r.find_by_name("other").is_none());
    }

    #[test]
    fn set_preferred_model_overrides_default() {
        let mut r = ModelRegistry::from_models(vec![
            make("big", 5_000_000_000),
            make("small", 1_000_000_000),
        ]);
        assert_eq!(r.default_model(), Some("big"));
        r.set_preferred_model("small");
        assert_eq!(r.default_model(), Some("small"));
    }

    #[test]
    fn set_preferred_model_unknown_name_keeps_default() {
        let mut r = ModelRegistry::from_models(vec![make("big", 5_000_000_000)]);
        r.set_preferred_model("ghost");
        assert_eq!(r.default_model(), Some("big"));
    }

    #[test]
    fn quantization_roundtrips_json() {
        let q = Quantization::Q4_0;
        let json = serde_json::to_string(&q).expect("serialize");
        let back: Quantization = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(q, back);
    }
}
