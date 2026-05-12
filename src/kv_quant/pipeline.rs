use crate::kv_quant::rotation::{
    GivensParams, QuaternionParams, apply_iso_forward, apply_iso_inverse, apply_planar_forward,
    apply_planar_inverse, centroids, dequantize_scalar, quantize_scalar,
};
use crate::kv_quant::turbo::{TurboCompressed, TurboEngine};
use crate::kv_quant::{KvCacheConfig, KvQuantization, RotorQuantization, TurboQuantization};

#[derive(Debug, Clone)]
enum CompressedStorage {
    Raw(Vec<u8>),
    Rotor(Vec<u8>),
    Turbo(TurboCompressed),
    Zero,
}

/// Compressed KV vector.
#[derive(Debug, Clone)]
pub struct CompressedKv {
    pub quantization: KvQuantization,
    pub norm: f32,
    pub bit_width: u8,
    storage: CompressedStorage,
}

impl Default for CompressedKv {
    fn default() -> Self {
        Self {
            quantization: KvQuantization::None,
            norm: 1.0,
            bit_width: 0,
            storage: CompressedStorage::Raw(Vec::new()),
        }
    }
}

impl CompressedKv {
    pub fn encoded_len(&self) -> usize {
        match &self.storage {
            CompressedStorage::Raw(bytes) => bytes.len(),
            CompressedStorage::Rotor(indices) => indices.len(),
            CompressedStorage::Turbo(payload) => payload.encoded_len(),
            CompressedStorage::Zero => 0,
        }
    }
}

/// CPU reference quantizer used for KV cache compression/decompression.
#[derive(Debug)]
pub struct KvQuantizer {
    config: KvCacheConfig,
    dim: usize,
    planar_params: Vec<GivensParams>,
    iso_params: Vec<QuaternionParams>,
    turbo_k: Option<TurboEngine>,
    turbo_v: Option<TurboEngine>,
    #[cfg(feature = "vulkan")]
    vulkan_executor: crate::kv_quant::vulkan::VulkanExecutor,
}

impl KvQuantizer {
    /// Initialize with deterministic pseudo-random rotation parameters.
    pub fn new(config: KvCacheConfig, dim: usize) -> Self {
        let planar_params = (0..(dim / 2))
            .map(|i| {
                let theta = (i as f32 * 0.173_205_08).sin();
                GivensParams {
                    cos_theta: theta.cos(),
                    sin_theta: theta.sin(),
                }
            })
            .collect();

        let iso_params = (0..(dim / 4))
            .map(|i| {
                let t = (i + 1) as f32;
                QuaternionParams::new(1.0, (0.37 * t).sin(), (0.53 * t).cos(), (0.71 * t).sin())
            })
            .collect();

        Self {
            turbo_k: build_turbo_engine(config.k, dim),
            turbo_v: build_turbo_engine(config.v, dim),
            config,
            dim,
            planar_params,
            iso_params,
            #[cfg(feature = "vulkan")]
            vulkan_executor: crate::kv_quant::vulkan::VulkanExecutor::new(dim),
        }
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    pub fn set_config(&mut self, config: KvCacheConfig) {
        self.turbo_k = build_turbo_engine(config.k, self.dim);
        self.turbo_v = build_turbo_engine(config.v, self.dim);
        self.config = config;
    }

    pub fn compress_k(&self, v: &[f32]) -> CompressedKv {
        self.compress(v, self.config.k)
    }

    pub fn compress_v(&self, v: &[f32]) -> CompressedKv {
        self.compress(v, self.config.v)
    }

    pub fn decompress(&self, compressed: &CompressedKv) -> Vec<f32> {
        let mut out = vec![0.0; self.dim];
        self.decompress_into(compressed, &mut out);
        out
    }

    #[doc(hidden)]
    pub fn compress_k_into(&self, v: &[f32], scratch: &mut [f32], out: &mut CompressedKv) {
        self.compress_into(v, self.config.k, scratch, out);
    }

    #[doc(hidden)]
    pub fn compress_v_into(&self, v: &[f32], scratch: &mut [f32], out: &mut CompressedKv) {
        self.compress_into(v, self.config.v, scratch, out);
    }

    #[doc(hidden)]
    pub fn decompress_into(&self, compressed: &CompressedKv, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);

        match (compressed.quantization, &compressed.storage) {
            (_, CompressedStorage::Zero) => {
                out.fill(0.0);
            }
            (KvQuantization::None, CompressedStorage::Raw(bytes)) => {
                assert_eq!(bytes.len(), self.dim * 4);
                for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            (KvQuantization::Rotor(rotor), CompressedStorage::Rotor(indices)) => {
                self.decompress_rotor(indices, rotor, compressed, out);
            }
            (KvQuantization::Turbo(turbo), CompressedStorage::Turbo(payload)) => {
                let owned_engine;
                let engine = match self.turbo_engine_for(KvQuantization::Turbo(turbo)) {
                    Some(engine) => engine,
                    None => {
                        owned_engine = TurboEngine::new(turbo, self.dim);
                        &owned_engine
                    }
                };

                let normalized = engine.decompress_normalized(payload);
                assert_eq!(normalized.len(), self.dim);
                for (slot, value) in out.iter_mut().zip(normalized) {
                    *slot = value * compressed.norm;
                }
            }
            _ => panic!("compressed payload does not match the stored quantization"),
        }
    }

    fn compress_into(
        &self,
        v: &[f32],
        quant: KvQuantization,
        scratch: &mut [f32],
        out: &mut CompressedKv,
    ) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(scratch.len(), self.dim);

        match quant {
            KvQuantization::None => self.compress_raw_into(v, out),
            KvQuantization::Rotor(rotor) => self.compress_rotor_into(v, rotor, out, scratch),
            KvQuantization::Turbo(_) => {
                *out = self.compress(v, quant);
            }
        }
    }

    fn compress(&self, v: &[f32], quant: KvQuantization) -> CompressedKv {
        assert_eq!(v.len(), self.dim);

        match quant {
            KvQuantization::None => self.compress_raw(v),
            KvQuantization::Rotor(rotor) => self.compress_rotor(v, rotor),
            KvQuantization::Turbo(turbo) => self.compress_turbo(v, turbo),
        }
    }

    fn compress_raw(&self, v: &[f32]) -> CompressedKv {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &value in v {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        CompressedKv {
            quantization: KvQuantization::None,
            norm: 1.0,
            bit_width: 0,
            storage: CompressedStorage::Raw(bytes),
        }
    }

    fn compress_raw_into(&self, v: &[f32], out: &mut CompressedKv) {
        let mut bytes = match std::mem::replace(&mut out.storage, CompressedStorage::Zero) {
            CompressedStorage::Raw(bytes) => bytes,
            _ => Vec::new(),
        };
        bytes.resize(v.len() * 4, 0);
        for (chunk, &value) in bytes.chunks_exact_mut(4).zip(v.iter()) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        out.quantization = KvQuantization::None;
        out.norm = 1.0;
        out.bit_width = 0;
        out.storage = CompressedStorage::Raw(bytes);
    }

    fn compress_rotor(&self, v: &[f32], rotor: RotorQuantization) -> CompressedKv {
        let norm = vector_norm(v);
        if norm == 0.0 {
            return CompressedKv {
                quantization: KvQuantization::Rotor(rotor),
                norm,
                bit_width: rotor.bit_width(),
                storage: CompressedStorage::Zero,
            };
        }

        let mut rotated = v.iter().map(|value| value / norm).collect::<Vec<_>>();
        self.apply_rotor_forward(&mut rotated, rotor);

        let bit_width = rotor.bit_width();
        let centroids = centroids_for_bits(bit_width);
        let coord_scale = (self.dim as f32).sqrt();
        let indices = rotated
            .iter()
            .map(|&value| quantize_scalar(value * coord_scale, centroids))
            .collect();

        CompressedKv {
            quantization: KvQuantization::Rotor(rotor),
            norm,
            bit_width,
            storage: CompressedStorage::Rotor(indices),
        }
    }

    fn compress_rotor_into(
        &self,
        v: &[f32],
        rotor: RotorQuantization,
        out: &mut CompressedKv,
        scratch: &mut [f32],
    ) {
        let norm = vector_norm(v);
        if norm == 0.0 {
            out.quantization = KvQuantization::Rotor(rotor);
            out.norm = 0.0;
            out.bit_width = rotor.bit_width();
            out.storage = CompressedStorage::Zero;
            return;
        }

        for (slot, &value) in scratch.iter_mut().zip(v.iter()) {
            *slot = value / norm;
        }
        self.apply_rotor_forward(scratch, rotor);

        let mut indices = match std::mem::replace(&mut out.storage, CompressedStorage::Zero) {
            CompressedStorage::Rotor(indices) => indices,
            _ => Vec::new(),
        };
        indices.resize(self.dim, 0);

        let centroids = centroids_for_bits(rotor.bit_width());
        let coord_scale = (self.dim as f32).sqrt();
        for (slot, &value) in indices.iter_mut().zip(scratch.iter()) {
            *slot = quantize_scalar(value * coord_scale, centroids);
        }

        out.quantization = KvQuantization::Rotor(rotor);
        out.norm = norm;
        out.bit_width = rotor.bit_width();
        out.storage = CompressedStorage::Rotor(indices);
    }

    fn compress_turbo(&self, v: &[f32], turbo: TurboQuantization) -> CompressedKv {
        let norm = vector_norm(v);
        if norm == 0.0 {
            return CompressedKv {
                quantization: KvQuantization::Turbo(turbo),
                norm,
                bit_width: turbo.bits,
                storage: CompressedStorage::Zero,
            };
        }

        let normalized = v.iter().map(|value| value / norm).collect::<Vec<_>>();
        let owned_engine;
        let engine = match self.turbo_engine_for(KvQuantization::Turbo(turbo)) {
            Some(engine) => engine,
            None => {
                owned_engine = TurboEngine::new(turbo, self.dim);
                &owned_engine
            }
        };

        CompressedKv {
            quantization: KvQuantization::Turbo(turbo),
            norm,
            bit_width: turbo.bits,
            storage: CompressedStorage::Turbo(engine.compress_normalized(&normalized)),
        }
    }

    fn decompress_rotor(
        &self,
        indices: &[u8],
        rotor: RotorQuantization,
        compressed: &CompressedKv,
        out: &mut [f32],
    ) {
        assert_eq!(indices.len(), self.dim);

        let centroids = centroids_for_bits(compressed.bit_width);
        let coord_scale = (self.dim as f32).sqrt();
        for (slot, &index) in out.iter_mut().zip(indices.iter()) {
            *slot = dequantize_scalar(index, centroids) / coord_scale;
        }

        match rotor {
            RotorQuantization::Planar2 | RotorQuantization::Planar3 => {
                apply_planar_inverse(out, &self.planar_params)
            }
            RotorQuantization::Iso4 | RotorQuantization::Iso3 => {
                apply_iso_inverse(out, &self.iso_params)
            }
        }

        if compressed.norm > 0.0 {
            for value in out.iter_mut() {
                *value *= compressed.norm;
            }
        }
    }

    fn apply_rotor_forward(&self, values: &mut [f32], rotor: RotorQuantization) {
        match rotor {
            RotorQuantization::Planar2 | RotorQuantization::Planar3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor
                        .apply_planar(values, &self.planar_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_planar_forward(values, &self.planar_params)
                }
            }
            RotorQuantization::Iso4 | RotorQuantization::Iso3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor.apply_iso(values, &self.iso_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_iso_forward(values, &self.iso_params)
                }
            }
        }
    }

    fn turbo_engine_for(&self, quant: KvQuantization) -> Option<&TurboEngine> {
        if self.config.k == quant {
            self.turbo_k.as_ref()
        } else if self.config.v == quant {
            self.turbo_v.as_ref()
        } else {
            None
        }
    }
}

fn build_turbo_engine(quant: KvQuantization, dim: usize) -> Option<TurboEngine> {
    match quant {
        KvQuantization::Turbo(config) => Some(TurboEngine::new(config, dim)),
        _ => None,
    }
}

fn vector_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn centroids_for_bits(bits: u8) -> &'static [f32] {
    match bits {
        2 => &centroids::BITS_2,
        3 => &centroids::BITS_3,
        4 => &centroids::BITS_4,
        _ => panic!("unsupported bit-width: {bits}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na * nb)
        }
    }

    fn sample_unit_vec(dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32 + 1.0) * 0.123_45).sin() + ((i as f32 + 7.0) * 0.077).cos();
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn planar2_roundtrip_cosine_above_threshold() {
        let dim = 64;
        let mut v = vec![0.0f32; dim];
        for (i, x) in v.iter_mut().enumerate() {
            *x = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::planar2(),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[test]
    fn iso4_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::iso4(),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.98);
    }

    #[test]
    fn planar3_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::planar3(),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[test]
    fn iso3_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::iso3(),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[cfg(not(feature = "turboquant"))]
    #[test]
    #[should_panic(expected = "requires the `turboquant` feature")]
    fn turbo_quantization_requires_feature() {
        let _ = KvQuantizer::new(KvCacheConfig::turbo_balanced_4bit(), 128);
    }

    #[cfg(feature = "turboquant")]
    #[test]
    fn turbo_mse_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::turbo_mse(4),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.90);
    }

    #[cfg(feature = "turboquant")]
    #[test]
    fn turbo_prod_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::turbo_prod(4),
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert!(cosine_similarity(&v, &d) > 0.90);
    }

    #[test]
    fn none_roundtrip_is_exact() {
        let dim = 16;
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 3.0).collect();
        let q = KvQuantizer::new(KvCacheConfig::none(), dim);
        let c = q.compress_k(&v);
        let d = q.decompress(&c);
        assert_eq!(v, d);
    }

    #[test]
    fn scratch_paths_match_allocating_paths() {
        let dim = 128;
        let v = sample_unit_vec(dim);

        let quantizations = {
            #[cfg(feature = "turboquant")]
            {
                let mut quantizations = vec![
                    KvQuantization::None,
                    KvQuantization::planar2(),
                    KvQuantization::planar3(),
                    KvQuantization::iso4(),
                    KvQuantization::iso3(),
                ];
                quantizations.push(KvQuantization::turbo_mse(4));
                quantizations.push(KvQuantization::turbo_prod(4));
                quantizations
            }

            #[cfg(not(feature = "turboquant"))]
            {
                vec![
                    KvQuantization::None,
                    KvQuantization::planar2(),
                    KvQuantization::planar3(),
                    KvQuantization::iso4(),
                    KvQuantization::iso3(),
                ]
            }
        };

        for quant in quantizations {
            let q = KvQuantizer::new(KvCacheConfig { k: quant, v: quant }, dim);

            let expected_compressed = q.compress_k(&v);
            let mut scratch = vec![0.0; dim];
            let mut actual_compressed = CompressedKv::default();
            q.compress_k_into(&v, &mut scratch, &mut actual_compressed);

            assert_eq!(
                expected_compressed.quantization,
                actual_compressed.quantization
            );
            assert!((expected_compressed.norm - actual_compressed.norm).abs() < 1e-6);
            assert_eq!(expected_compressed.bit_width, actual_compressed.bit_width);
            assert_eq!(
                expected_compressed.encoded_len(),
                actual_compressed.encoded_len()
            );

            let expected_decompressed = q.decompress(&expected_compressed);
            let mut actual_decompressed = vec![0.0; dim];
            q.decompress_into(&actual_compressed, &mut actual_decompressed);

            for (expected, actual) in expected_decompressed.iter().zip(actual_decompressed.iter()) {
                assert!((expected - actual).abs() < 1e-6);
            }
        }
    }
}
