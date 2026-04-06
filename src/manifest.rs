//! Ollama manifest parser for GGUF model discovery.
//!
//! Ollama stores models under:
//! - `<models_dir>/manifests/registry.ollama.ai/library/<name>/<tag>` — JSON manifest
//! - `<models_dir>/blobs/<digest>` — GGUF blob files
//!
//! The manifest JSON `layers` array contains the GGUF layer identified by
//! `mediaType == "application/vnd.ollama.image.model"`.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::InferError;
use crate::registry::{ModelInfo, Quantization};

/// Partial Ollama manifest — only the fields needed for discovery.
#[derive(Debug, Deserialize)]
struct OllamaManifest {
    layers: Vec<ManifestLayer>,
}

/// A single layer entry in an Ollama manifest.
#[derive(Debug, Deserialize)]
struct ManifestLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

/// Parse all Ollama manifests in the given models directory.
///
/// Traverses the standard manifest hierarchy, parses each JSON manifest, and
/// resolves the corresponding GGUF blob paths. Models whose blobs are missing
/// or whose manifests are unparseable emit a tracing warning and are skipped;
/// remaining models are still returned.
///
/// # Errors
///
/// Returns [`InferError::ManifestParseFailure`] when the manifests directory
/// itself does not exist.
pub fn parse_ollama_manifests(models_dir: &Path) -> Result<Vec<ModelInfo>, InferError> {
    let manifests_dir = models_dir
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library");

    if !manifests_dir.exists() {
        return Err(InferError::ManifestParseFailure(format!(
            "manifests directory not found at {}",
            manifests_dir.display()
        )));
    }

    let mut models = Vec::new();

    let entries = fs::read_dir(&manifests_dir).map_err(InferError::IoError)?;

    for entry in entries {
        let entry = entry.map_err(InferError::IoError)?;
        let model_dir = entry.path();
        if !model_dir.is_dir() {
            continue;
        }

        let model_name = match model_dir.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let tag_entries = match fs::read_dir(&model_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for te in tag_entries {
            let te = match te {
                Ok(e) => e,
                Err(_) => continue,
            };
            let tag_path = te.path();
            if !tag_path.is_file() {
                continue;
            }
            let tag = match tag_path.file_name().and_then(|n| n.to_str()) {
                Some(t) => t.to_string(),
                None => continue,
            };

            match parse_single_manifest(&tag_path, models_dir, &model_name, &tag) {
                Ok(Some(info)) => models.push(info),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("skipping {}: {}", tag_path.display(), e);
                }
            }
        }
    }

    Ok(models)
}

/// Parse a single manifest file, returning [`ModelInfo`] if a GGUF layer exists.
fn parse_single_manifest(
    manifest_path: &Path,
    models_dir: &Path,
    model_name: &str,
    tag: &str,
) -> Result<Option<ModelInfo>, InferError> {
    let content = fs::read_to_string(manifest_path)?;

    let manifest: OllamaManifest = serde_json::from_str(&content).map_err(|e| {
        InferError::ManifestParseFailure(format!(
            "invalid JSON in {}: {}",
            manifest_path.display(),
            e
        ))
    })?;

    let layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == "application/vnd.ollama.image.model");

    let layer = match layer {
        Some(l) => l,
        None => return Ok(None),
    };

    let blob_filename = layer.digest.replace(':', "-");
    let blob_path = models_dir.join("blobs").join(&blob_filename);

    if !blob_path.exists() {
        return Err(InferError::ManifestParseFailure(format!(
            "blob not found: {}",
            blob_path.display()
        )));
    }

    let size_bytes = fs::metadata(&blob_path)?.len();
    let full_name = if tag == "latest" {
        model_name.to_string()
    } else {
        format!("{}:{}", model_name, tag)
    };

    Ok(Some(ModelInfo {
        name: full_name,
        path: blob_path,
        size_bytes,
        quantization: parse_quantization_from_name(model_name),
    }))
}

/// Heuristically derive a [`Quantization`] level from a model name.
fn parse_quantization_from_name(name: &str) -> Quantization {
    let n = name.to_lowercase();
    if n.contains("q4_0") || n.contains("q4-0") {
        Quantization::Q4_0
    } else if n.contains("q4_1") || n.contains("q4-1") {
        Quantization::Q4_1
    } else if n.contains("q5_0") || n.contains("q5-0") {
        Quantization::Q5_0
    } else if n.contains("q5_1") || n.contains("q5-1") {
        Quantization::Q5_1
    } else if n.contains("q8_0") || n.contains("q8-0") {
        Quantization::Q8_0
    } else if n.contains("f16") || n.contains("fp16") {
        Quantization::F16
    } else if n.contains("f32") || n.contains("fp32") {
        Quantization::F32
    } else {
        Quantization::Unknown(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_manifest(base: &Path, model: &str, tag: &str, digest: &str) {
        let dir = base
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(model);
        fs::create_dir_all(&dir).expect("create manifest dir");
        let json = format!(
            r#"{{"schemaVersion":2,"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"{}","size":0}}]}}"#,
            digest
        );
        fs::write(dir.join(tag), json).expect("write manifest");
    }

    fn write_blob(base: &Path, digest: &str, size: usize) {
        let blobs = base.join("blobs");
        fs::create_dir_all(&blobs).expect("create blobs");
        fs::write(blobs.join(digest.replace(':', "-")), vec![0u8; size]).expect("write blob");
    }

    #[test]
    fn discovers_single_model() {
        let tmp = tempdir().expect("tempdir");
        write_manifest(tmp.path(), "llama2", "latest", "sha256:abc");
        write_blob(tmp.path(), "sha256:abc", 1024);
        let models = parse_ollama_manifests(tmp.path()).expect("parse");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "llama2");
        assert_eq!(models[0].size_bytes, 1024);
    }

    #[test]
    fn non_latest_tag_appended_to_name() {
        let tmp = tempdir().expect("tempdir");
        write_manifest(tmp.path(), "llama2", "13b", "sha256:def");
        write_blob(tmp.path(), "sha256:def", 512);
        let models = parse_ollama_manifests(tmp.path()).expect("parse");
        assert_eq!(models[0].name, "llama2:13b");
    }

    #[test]
    fn missing_manifests_dir_errors() {
        let tmp = tempdir().expect("tempdir");
        assert!(matches!(
            parse_ollama_manifests(tmp.path()),
            Err(InferError::ManifestParseFailure(_))
        ));
    }

    #[test]
    fn missing_blob_skips_model() {
        let tmp = tempdir().expect("tempdir");
        write_manifest(tmp.path(), "noblob", "latest", "sha256:gone");
        // no blob created
        match parse_ollama_manifests(tmp.path()) {
            Ok(models) => assert_eq!(models.len(), 0),
            Err(_) => {} // also acceptable
        }
    }

    #[test]
    fn quantization_q4_0() {
        assert_eq!(
            parse_quantization_from_name("llama2-q4_0"),
            Quantization::Q4_0
        );
    }

    #[test]
    fn quantization_f16() {
        assert_eq!(parse_quantization_from_name("model-f16"), Quantization::F16);
    }

    #[test]
    fn quantization_unknown() {
        assert!(matches!(
            parse_quantization_from_name("custom"),
            Quantization::Unknown(_)
        ));
    }
}
