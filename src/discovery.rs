//! Model discovery orchestration.
//!
//! Scans the Ollama model store and builds a [`ModelRegistry`]. Also exposes
//! [`auto_backend`], a convenience constructor that loads a GGUF model into
//! the best available backend without requiring fine-grained control.

use std::path::{Path, PathBuf};

use crate::backend::{BackendType, InferenceBackend};
use crate::error::InferError;
use crate::manifest;
use crate::registry::ModelRegistry;

/// Return the platform-appropriate Ollama models directory.
///
/// - **macOS / Linux** — `$HOME/.ollama/models`
/// - **Windows** — `%USERPROFILE%\.ollama\models`
///
/// # Errors
///
/// Returns [`InferError::OllamaNotInstalled`] when the required environment
/// variable (`HOME` or `USERPROFILE`) is not set.
pub fn ollama_models_dir() -> Result<PathBuf, InferError> {
    #[cfg(target_os = "windows")]
    {
        let root = std::env::var("USERPROFILE").map_err(|_| {
            InferError::OllamaNotInstalled(
                "USERPROFILE environment variable is not set".to_string(),
            )
        })?;
        Ok(PathBuf::from(root).join(".ollama").join("models"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        let home = std::env::var("HOME").map_err(|_| {
            InferError::OllamaNotInstalled("HOME environment variable is not set".to_string())
        })?;
        Ok(PathBuf::from(home).join(".ollama").join("models"))
    }
}

/// Discover all available GGUF models in the given Ollama models directory.
///
/// Parses Ollama manifests under `models_dir` and returns a populated
/// [`ModelRegistry`] with the largest model selected as the default.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|--------------|
/// | `models_dir` does not exist | [`InferError::OllamaNotInstalled`] |
/// | No models have been pulled | [`InferError::NoModelsFound`] |
/// | Manifest JSON is malformed | [`InferError::ManifestParseFailure`] |
pub fn discover_models(models_dir: &Path) -> Result<ModelRegistry, InferError> {
    if !models_dir.exists() {
        return Err(InferError::OllamaNotInstalled(format!(
            "Ollama models directory not found at {}",
            models_dir.display()
        )));
    }

    let models = manifest::parse_ollama_manifests(models_dir)?;

    if models.is_empty() {
        return Err(InferError::NoModelsFound);
    }

    Ok(ModelRegistry::from_models(models))
}

/// Construct and immediately load the best available backend for `model_path`.
///
/// When the `llama` feature is enabled, constructs a [`crate::LlamaBackend`].
/// Without default features, falls back to [`crate::MockBackend`] — useful for
/// environments that cannot pull in the C++ toolchain.
///
/// # Errors
///
/// Returns [`InferError::ModelLoadFailure`] if the model file cannot be loaded.
pub fn auto_backend(
    model_path: &Path,
    backend_type: BackendType,
) -> Result<Box<dyn InferenceBackend + Send + Sync>, InferError> {
    #[cfg(feature = "llama")]
    {
        let mut backend = crate::backend::llama::LlamaBackend::new()?;
        backend.load_model(model_path, backend_type)?;
        Ok(Box::new(backend))
    }

    #[cfg(not(feature = "llama"))]
    {
        let mut backend = crate::backend::mock::MockBackend::new();
        backend.load_model(model_path, backend_type)?;
        Ok(Box::new(backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_ollama_structure(base: &Path) {
        let lib = base
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library");
        let model_dir = lib.join("test-model");
        fs::create_dir_all(&model_dir).expect("manifest dir");
        let manifest = r#"{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:testdigest","size":0}]}"#;
        fs::write(model_dir.join("latest"), manifest).expect("write manifest");
        let blobs = base.join("blobs");
        fs::create_dir_all(&blobs).expect("blobs dir");
        fs::write(blobs.join("sha256-testdigest"), vec![0u8; 1024]).expect("write blob");
    }

    #[test]
    fn returns_ollama_not_installed_for_missing_dir() {
        let tmp = tempdir().expect("tempdir");
        let result = discover_models(&tmp.path().join("nonexistent"));
        assert!(matches!(result, Err(InferError::OllamaNotInstalled(_))));
    }

    #[test]
    fn returns_no_models_found_for_empty_manifests_dir() {
        let tmp = tempdir().expect("tempdir");
        let lib = tmp
            .path()
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library");
        fs::create_dir_all(&lib).expect("create dir");
        assert!(matches!(
            discover_models(tmp.path()),
            Err(InferError::NoModelsFound)
        ));
    }

    #[test]
    fn discovers_models_and_returns_populated_registry() {
        let tmp = tempdir().expect("tempdir");
        make_ollama_structure(tmp.path());
        let r = discover_models(tmp.path()).expect("discover");
        assert!(!r.is_empty());
        assert_eq!(r.model_count(), 1);
        assert_eq!(r.default_model(), Some("test-model"));
    }
}
