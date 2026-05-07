//! Pure Rust rotation and scalar quantization reference kernels.

/// Learned Givens rotation parameters for one 2D group.
#[derive(Debug, Clone, Copy)]
pub struct GivensParams {
    pub cos_theta: f32,
    pub sin_theta: f32,
}

/// Apply forward Givens rotation to a vector in-place.
/// `params` must have length d/2 for a d-dimensional vector.
pub fn apply_planar_forward(v: &mut [f32], params: &[GivensParams]) {
    debug_assert_eq!(v.len() / 2, params.len());
    for (chunk, p) in v.chunks_exact_mut(2).zip(params.iter()) {
        let x = chunk[0];
        let y = chunk[1];
        chunk[0] = x * p.cos_theta + y * p.sin_theta;
        chunk[1] = -x * p.sin_theta + y * p.cos_theta;
    }
}

/// Apply inverse Givens rotation (transpose = inverse for orthogonal matrices).
pub fn apply_planar_inverse(v: &mut [f32], params: &[GivensParams]) {
    debug_assert_eq!(v.len() / 2, params.len());
    for (chunk, p) in v.chunks_exact_mut(2).zip(params.iter()) {
        let x = chunk[0];
        let y = chunk[1];
        chunk[0] = x * p.cos_theta - y * p.sin_theta;
        chunk[1] = x * p.sin_theta + y * p.cos_theta;
    }
}

/// Unit quaternion rotation parameters for one 4D group.
#[derive(Debug, Clone, Copy)]
pub struct QuaternionParams {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl QuaternionParams {
    /// Construct and normalize. Panics if the input vector is zero.
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        assert!(norm > 0.0, "quaternion parameters cannot be a zero vector");
        Self {
            w: w / norm,
            x: x / norm,
            y: y / norm,
            z: z / norm,
        }
    }

    /// Conjugate (= inverse for unit quaternion).
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Apply forward left-isoclinic quaternion rotation in-place.
/// `params` must have length d/4 for a d-dimensional vector.
pub fn apply_iso_forward(v: &mut [f32], params: &[QuaternionParams]) {
    debug_assert_eq!(v.len() / 4, params.len());
    for (chunk, q) in v.chunks_exact_mut(4).zip(params.iter()) {
        let (a, b, c, d) = (chunk[0], chunk[1], chunk[2], chunk[3]);
        chunk[0] = q.w * a - q.x * b - q.y * c - q.z * d;
        chunk[1] = q.x * a + q.w * b - q.z * c + q.y * d;
        chunk[2] = q.y * a + q.z * b + q.w * c - q.x * d;
        chunk[3] = q.z * a - q.y * b + q.x * c + q.w * d;
    }
}

/// Apply inverse (conjugate) rotation.
pub fn apply_iso_inverse(v: &mut [f32], params: &[QuaternionParams]) {
    let inv_params: Vec<QuaternionParams> = params.iter().map(|q| q.conjugate()).collect();
    apply_iso_forward(v, &inv_params);
}

/// Pre-computed Lloyd-Max centroids for near-Gaussian input.
/// Centroids are symmetric around zero.
pub mod centroids {
    /// 2-bit: 4 centroids
    pub const BITS_2: [f32; 4] = [-1.510, -0.453, 0.453, 1.510];
    /// 3-bit: 8 centroids
    pub const BITS_3: [f32; 8] = [-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152];
    /// 4-bit: 16 centroids
    pub const BITS_4: [f32; 16] = [
        -2.733, -2.069, -1.618, -1.224, -0.874, -0.556, -0.255, 0.0, 0.255, 0.556, 0.874, 1.224,
        1.618, 2.069, 2.733, 3.395,
    ];
}

/// Quantize a single scalar to the nearest Lloyd-Max centroid.
/// Returns the centroid index.
#[inline]
pub fn quantize_scalar(value: f32, centroids: &[f32]) -> u8 {
    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;
    for (i, &c) in centroids.iter().enumerate() {
        let dist = (value - c).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx as u8
}

/// Dequantize by centroid index.
#[inline]
pub fn dequantize_scalar(index: u8, centroids: &[f32]) -> f32 {
    centroids[index as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn planar_forward_inverse_roundtrip() {
        let mut v = vec![0.1f32; 128];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 * 0.017).sin();
        }
        let original = v.clone();
        let params: Vec<GivensParams> = (0..64)
            .map(|i| {
                let theta = i as f32 * 0.03125;
                GivensParams {
                    cos_theta: theta.cos(),
                    sin_theta: theta.sin(),
                }
            })
            .collect();

        apply_planar_forward(&mut v, &params);
        apply_planar_inverse(&mut v, &params);

        assert!(max_abs_diff(&v, &original) < 1e-6);
    }

    #[test]
    fn iso_forward_inverse_roundtrip() {
        let mut v = vec![0.2f32; 128];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 * 0.013).cos();
        }
        let original = v.clone();
        let params: Vec<QuaternionParams> = (0..32)
            .map(|i| {
                let t = (i + 1) as f32;
                QuaternionParams::new(1.0, 0.1 * t, 0.2 * t, 0.3 * t)
            })
            .collect();

        apply_iso_forward(&mut v, &params);
        apply_iso_inverse(&mut v, &params);

        assert!(max_abs_diff(&v, &original) < 1e-6);
    }

    #[test]
    fn quantize_scalar_returns_valid_index() {
        let sets: [&[f32]; 3] = [&centroids::BITS_2, &centroids::BITS_3, &centroids::BITS_4];
        for centroids in sets {
            for v in -40..=40 {
                let idx = quantize_scalar(v as f32 * 0.1, centroids);
                assert!((idx as usize) < centroids.len());
            }
        }
    }
}
