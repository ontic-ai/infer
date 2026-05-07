use crate::kv_quant::rotation::{
    GivensParams, QuaternionParams, apply_iso_forward, apply_iso_inverse, apply_planar_forward,
    apply_planar_inverse, centroids, dequantize_scalar, quantize_scalar,
};
use crate::kv_quant::{KvCacheConfig, KvQuantization};

/// Compressed KV vector.
#[derive(Debug, Clone, Default)]
pub struct CompressedKv {
    pub norm: f32,
    pub indices: Vec<u8>,
    pub bits: u8,
}

/// CPU reference quantizer used for KV cache compression/decompression.
#[derive(Debug, Clone)]
pub struct KvQuantizer {
    config: KvCacheConfig,
    dim: usize,
    planar_params: Vec<GivensParams>,
    iso_params: Vec<QuaternionParams>,
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
        self.config = config;
    }

    pub fn compress_k(&self, v: &[f32]) -> CompressedKv {
        self.compress(v, self.config.k)
    }

    pub fn compress_v(&self, v: &[f32]) -> CompressedKv {
        self.compress(v, self.config.v)
    }

    pub fn decompress(&self, compressed: &CompressedKv, quant: KvQuantization) -> Vec<f32> {
        if quant == KvQuantization::None {
            assert_eq!(compressed.indices.len(), self.dim * 4);
            let mut out = Vec::with_capacity(self.dim);
            for chunk in compressed.indices.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return out;
        }

        assert_eq!(compressed.indices.len(), self.dim);

        let centroids = centroids_for_bits(compressed.bits);
        let coord_scale = (self.dim as f32).sqrt();
        let mut out: Vec<f32> = compressed
            .indices
            .iter()
            .map(|&idx| dequantize_scalar(idx, centroids) / coord_scale)
            .collect();

        match quant {
            KvQuantization::Planar2 | KvQuantization::Planar3 => {
                apply_planar_inverse(&mut out, &self.planar_params)
            }
            KvQuantization::Iso4 | KvQuantization::Iso3 => {
                apply_iso_inverse(&mut out, &self.iso_params)
            }
            KvQuantization::None => unreachable!(),
        }

        if compressed.norm > 0.0 {
            for x in &mut out {
                *x *= compressed.norm;
            }
        }

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
    pub fn decompress_into(
        &self,
        compressed: &CompressedKv,
        quant: KvQuantization,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), self.dim);

        if quant == KvQuantization::None {
            assert_eq!(compressed.indices.len(), self.dim * 4);
            for (slot, chunk) in out.iter_mut().zip(compressed.indices.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            return;
        }

        assert_eq!(compressed.indices.len(), self.dim);

        let centroids = centroids_for_bits(compressed.bits);
        let coord_scale = (self.dim as f32).sqrt();
        for (slot, &idx) in out.iter_mut().zip(compressed.indices.iter()) {
            *slot = dequantize_scalar(idx, centroids) / coord_scale;
        }

        match quant {
            KvQuantization::Planar2 | KvQuantization::Planar3 => {
                apply_planar_inverse(out, &self.planar_params)
            }
            KvQuantization::Iso4 | KvQuantization::Iso3 => apply_iso_inverse(out, &self.iso_params),
            KvQuantization::None => unreachable!(),
        }

        if compressed.norm > 0.0 {
            for x in out.iter_mut() {
                *x *= compressed.norm;
            }
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

        if quant == KvQuantization::None {
            out.norm = 1.0;
            out.bits = 0;
            out.indices.resize(v.len() * 4, 0);
            for (chunk, &x) in out.indices.chunks_exact_mut(4).zip(v.iter()) {
                chunk.copy_from_slice(&x.to_le_bytes());
            }
            return;
        }

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (slot, &x) in scratch.iter_mut().zip(v.iter()) {
                *slot = x / norm;
            }
        } else {
            scratch.fill(0.0);
        }

        match quant {
            KvQuantization::Planar2 | KvQuantization::Planar3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor
                        .apply_planar(scratch, &self.planar_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_planar_forward(scratch, &self.planar_params)
                }
            }
            KvQuantization::Iso4 | KvQuantization::Iso3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor.apply_iso(scratch, &self.iso_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_iso_forward(scratch, &self.iso_params)
                }
            }
            KvQuantization::None => unreachable!(),
        }

        let bits = bits_for_quant(quant);
        let centroids = centroids_for_bits(bits);
        let coord_scale = (self.dim as f32).sqrt();
        out.norm = norm;
        out.bits = bits;
        out.indices.resize(self.dim, 0);
        for (slot, &x) in out.indices.iter_mut().zip(scratch.iter()) {
            *slot = quantize_scalar(x * coord_scale, centroids);
        }
    }

    fn compress(&self, v: &[f32], quant: KvQuantization) -> CompressedKv {
        assert_eq!(v.len(), self.dim);

        if quant == KvQuantization::None {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            return CompressedKv {
                norm: 1.0,
                indices: bytes,
                bits: 0,
            };
        }

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut rotated: Vec<f32> = if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            vec![0.0; self.dim]
        };

        match quant {
            KvQuantization::Planar2 | KvQuantization::Planar3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor
                        .apply_planar(&mut rotated, &self.planar_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_planar_forward(&mut rotated, &self.planar_params)
                }
            }
            KvQuantization::Iso4 | KvQuantization::Iso3 => {
                #[cfg(feature = "vulkan")]
                {
                    self.vulkan_executor
                        .apply_iso(&mut rotated, &self.iso_params);
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    apply_iso_forward(&mut rotated, &self.iso_params)
                }
            }
            KvQuantization::None => unreachable!(),
        }

        let bits = bits_for_quant(quant);
        let centroids = centroids_for_bits(bits);
        let coord_scale = (self.dim as f32).sqrt();
        let indices = rotated
            .iter()
            .map(|&x| quantize_scalar(x * coord_scale, centroids))
            .collect();

        CompressedKv {
            norm,
            indices,
            bits,
        }
    }
}

fn bits_for_quant(quant: KvQuantization) -> u8 {
    match quant {
        KvQuantization::Planar2 => 2,
        KvQuantization::Planar3 | KvQuantization::Iso3 => 3,
        KvQuantization::Iso4 => 4,
        KvQuantization::None => 0,
    }
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
                k: KvQuantization::Planar2,
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c, KvQuantization::Planar2);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[test]
    fn iso4_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::Iso4,
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c, KvQuantization::Iso4);
        assert!(cosine_similarity(&v, &d) > 0.98);
    }

    #[test]
    fn planar3_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::Planar3,
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c, KvQuantization::Planar3);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[test]
    fn iso3_roundtrip_cosine_above_threshold() {
        let dim = 128;
        let v = sample_unit_vec(dim);
        let q = KvQuantizer::new(
            KvCacheConfig {
                k: KvQuantization::Iso3,
                v: KvQuantization::None,
            },
            dim,
        );
        let c = q.compress_k(&v);
        let d = q.decompress(&c, KvQuantization::Iso3);
        assert!(cosine_similarity(&v, &d) > 0.97);
    }

    #[test]
    fn none_roundtrip_is_exact() {
        let dim = 16;
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 3.0).collect();
        let q = KvQuantizer::new(KvCacheConfig::none(), dim);
        let c = q.compress_k(&v);
        let d = q.decompress(&c, KvQuantization::None);
        assert_eq!(v, d);
    }

    #[test]
    fn scratch_paths_match_allocating_paths() {
        let dim = 128;
        let v = sample_unit_vec(dim);

        for quant in [
            KvQuantization::None,
            KvQuantization::Planar2,
            KvQuantization::Planar3,
            KvQuantization::Iso4,
            KvQuantization::Iso3,
        ] {
            let q = KvQuantizer::new(KvCacheConfig { k: quant, v: quant }, dim);

            let expected_compressed = q.compress_k(&v);
            let mut scratch = vec![0.0; dim];
            let mut actual_compressed = CompressedKv::default();
            q.compress_k_into(&v, &mut scratch, &mut actual_compressed);

            assert!((expected_compressed.norm - actual_compressed.norm).abs() < 1e-6);
            assert_eq!(expected_compressed.bits, actual_compressed.bits);
            assert_eq!(expected_compressed.indices, actual_compressed.indices);

            let expected_decompressed = q.decompress(&expected_compressed, quant);
            let mut actual_decompressed = vec![0.0; dim];
            q.decompress_into(&actual_compressed, quant, &mut actual_decompressed);

            for (expected, actual) in expected_decompressed.iter().zip(actual_decompressed.iter()) {
                assert!((expected - actual).abs() < 1e-6);
            }
        }
    }
}
