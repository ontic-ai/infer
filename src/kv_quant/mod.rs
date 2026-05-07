//! KV cache quantization variants and configuration.

pub mod pipeline;
pub mod rotation;
#[cfg(feature = "vulkan")]
pub mod vulkan;

/// KV cache quantization variant.
///
/// Applied independently to K and V caches via [`KvCacheConfig`].
/// `None` is the default (FP16 K and V, no compression).
///
/// Quality/VRAM guidance is benchmark-driven (see `README.md`):
/// - `Planar2`: near-zero perplexity impact, moderate compression
/// - `Planar3`: higher compression, around 6.3% perplexity increase vs FP16 in reference runs
/// - `Iso4`: similar compression to `Planar3` with better quality
/// - `Iso3`: best quality-per-bit; strongest compression with around 4.2% perplexity increase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum KvQuantization {
    #[default]
    None,
    /// 2D Givens rotation + 2-bit Lloyd-Max scalar quantization.
    /// Approx 5.1x compression with near-zero quality impact.
    Planar2,
    /// 2D Givens rotation + 3-bit Lloyd-Max scalar quantization.
    /// Approx 10.3x compression with modest quality impact.
    Planar3,
    /// 4D quaternion isoclinic rotation + 4-bit Lloyd-Max.
    /// Approx 10.3x compression with better quality than Planar3.
    Iso4,
    /// 4D quaternion isoclinic rotation + 3-bit Lloyd-Max.
    /// Approx 10.3x compression and the best quality-per-bit in this set.
    Iso3,
}

/// KV cache quantization configuration.
///
/// `k` and `v` are configured independently.
///
/// Recommended presets:
/// - [`KvCacheConfig::deferred_k`]: quantize K only, typically best quality/VRAM tradeoff.
/// - [`KvCacheConfig::symmetric_3bit`]: quantize both K and V for maximum compression.
///
/// Practical recommendations by memory budget:
/// - Comfortable VRAM: use [`KvCacheConfig::none`]
/// - Moderate VRAM pressure: use [`KvCacheConfig::deferred_k`]
/// - Severe VRAM pressure / long contexts: use [`KvCacheConfig::symmetric_3bit`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct KvCacheConfig {
    pub k: KvQuantization,
    pub v: KvQuantization,
}

impl KvCacheConfig {
    /// No compression. FP16 K and V.
    pub fn none() -> Self {
        Self::default()
    }

    /// Deferred K quantization. Recommended default for VRAM-constrained hardware.
    pub fn deferred_k() -> Self {
        Self {
            k: KvQuantization::Planar3,
            v: KvQuantization::None,
        }
    }

    /// Symmetric 3-bit configuration for maximum KV compression.
    pub fn symmetric_3bit() -> Self {
        Self {
            k: KvQuantization::Iso3,
            v: KvQuantization::Iso3,
        }
    }

    /// Returns true if any quantization is enabled.
    pub fn is_enabled(&self) -> bool {
        self.k != KvQuantization::None || self.v != KvQuantization::None
    }
}
