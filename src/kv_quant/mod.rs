//! KV cache quantization variants and configuration.

pub mod pipeline;
pub mod rotation;
pub mod turbo;
#[cfg(feature = "vulkan")]
pub mod vulkan;

/// Default seed used by TurboQuant-backed configurations.
///
/// TurboQuant itself requires the `turboquant` feature at runtime.
pub const DEFAULT_TURBO_SEED: u64 = 42;

/// KV cache quantization algorithm.
///
/// Applied independently to K and V caches via [`KvCacheConfig`]. `None` keeps
/// the channel uncompressed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum KvQuantization {
    #[default]
    None,
    /// Existing in-repo RotorQuant-style Planar/Iso family.
    Rotor(RotorQuantization),
    /// TurboQuant family backed by the upstream `turboquant` crate.
    ///
    /// Requires the `turboquant` feature when used by the CPU/reference pipeline.
    Turbo(TurboQuantization),
}

impl KvQuantization {
    pub const fn planar2() -> Self {
        Self::Rotor(RotorQuantization::Planar2)
    }

    pub const fn planar3() -> Self {
        Self::Rotor(RotorQuantization::Planar3)
    }

    pub const fn iso4() -> Self {
        Self::Rotor(RotorQuantization::Iso4)
    }

    pub const fn iso3() -> Self {
        Self::Rotor(RotorQuantization::Iso3)
    }

    pub const fn turbo_mse(bits: u8) -> Self {
        Self::Turbo(TurboQuantization::mse(bits))
    }

    pub const fn turbo_prod(bits: u8) -> Self {
        Self::Turbo(TurboQuantization::prod(bits))
    }

    pub const fn bit_width(self) -> Option<u8> {
        match self {
            Self::None => None,
            Self::Rotor(rotor) => Some(rotor.bit_width()),
            Self::Turbo(turbo) => Some(turbo.bits),
        }
    }

    pub const fn is_native_llama_supported(self) -> bool {
        matches!(self, Self::None | Self::Rotor(_))
    }
}

/// Existing in-repo RotorQuant-style algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RotorQuantization {
    /// 2D Givens rotation + 2-bit Lloyd-Max scalar quantization.
    Planar2,
    /// 2D Givens rotation + 3-bit Lloyd-Max scalar quantization.
    Planar3,
    /// 4D quaternion isoclinic rotation + 4-bit Lloyd-Max scalar quantization.
    Iso4,
    /// 4D quaternion isoclinic rotation + 3-bit Lloyd-Max scalar quantization.
    Iso3,
}

impl RotorQuantization {
    pub const fn bit_width(self) -> u8 {
        match self {
            Self::Planar2 => 2,
            Self::Planar3 | Self::Iso3 => 3,
            Self::Iso4 => 4,
        }
    }
}

/// TurboQuant configuration for one KV channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TurboQuantization {
    pub bits: u8,
    pub strategy: TurboQuantStrategy,
    pub seed: u64,
}

impl TurboQuantization {
    pub const fn mse(bits: u8) -> Self {
        Self {
            bits,
            strategy: TurboQuantStrategy::Mse,
            seed: DEFAULT_TURBO_SEED,
        }
    }

    pub const fn prod(bits: u8) -> Self {
        Self {
            bits,
            strategy: TurboQuantStrategy::Prod,
            seed: DEFAULT_TURBO_SEED,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// TurboQuant operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TurboQuantStrategy {
    /// Reconstruction-optimized TurboQuant.
    Mse,
    /// Attention-score-optimized TurboQuant.
    Prod,
}

/// KV cache quantization configuration.
///
/// `k` and `v` are configured independently.
///
/// Recommended presets:
/// - [`KvCacheConfig::deferred_k`]: quantize K only, typically best quality/VRAM tradeoff.
/// - [`KvCacheConfig::symmetric_3bit`]: quantize both K and V for maximum compression.
/// - [`KvCacheConfig::turbo_balanced_4bit`]: TurboQuant `Prod` keys + `Mse` values.
/// - [`KvCacheConfig::turbo_compact_3bit`]: lower-bit TurboQuant preset for experiments.
///
/// Practical recommendations by memory budget:
/// - Comfortable VRAM: use [`KvCacheConfig::none`]
/// - Moderate VRAM pressure: use [`KvCacheConfig::deferred_k`]
/// - Severe VRAM pressure / long contexts: use [`KvCacheConfig::symmetric_3bit`]
///
/// TurboQuant presets are available through the CPU/reference pipeline today
/// when the `turboquant` feature is enabled.
/// The current llama.cpp integration only supports the RotorQuant-style family
/// through native KV dtype mapping.
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
            k: KvQuantization::planar3(),
            v: KvQuantization::None,
        }
    }

    /// Symmetric 3-bit configuration for maximum KV compression.
    pub fn symmetric_3bit() -> Self {
        Self {
            k: KvQuantization::iso3(),
            v: KvQuantization::iso3(),
        }
    }

    /// TurboQuant 4-bit preset tuned for attention-sensitive workloads.
    pub fn turbo_balanced_4bit() -> Self {
        Self {
            k: KvQuantization::turbo_prod(4),
            v: KvQuantization::turbo_mse(4),
        }
    }

    /// TurboQuant 3-bit preset for aggressive software-managed compression experiments.
    pub fn turbo_compact_3bit() -> Self {
        Self {
            k: KvQuantization::turbo_prod(3),
            v: KvQuantization::turbo_mse(3),
        }
    }

    /// Returns true if any quantization is enabled.
    pub fn is_enabled(&self) -> bool {
        self.k != KvQuantization::None || self.v != KvQuantization::None
    }
}
