use crate::kv_quant::rotation::{
    GivensParams, QuaternionParams, apply_iso_forward, apply_planar_forward,
};

/// Vulkan-backed rotation executor.
///
/// This wrapper compiles and embeds SPIR-V shaders at build time. The current
/// implementation keeps a CPU reference path for correctness and portability.
#[derive(Debug, Clone)]
pub struct VulkanExecutor {
    pub dim: usize,
    #[allow(dead_code)]
    planar_spv: &'static [u8],
    #[allow(dead_code)]
    iso_spv: &'static [u8],
}

impl VulkanExecutor {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            planar_spv: include_bytes!(concat!(env!("OUT_DIR"), "/kvq_planar_rotate.spv")),
            iso_spv: include_bytes!(concat!(env!("OUT_DIR"), "/kvq_iso_rotate.spv")),
        }
    }

    /// Rotate one vector using the Planar kernel contract.
    pub fn apply_planar(&self, v: &mut [f32], params: &[GivensParams]) {
        let _ = self.planar_spv;
        apply_planar_forward(v, params);
    }

    /// Rotate one vector using the Iso kernel contract.
    pub fn apply_iso(&self, v: &mut [f32], params: &[QuaternionParams]) {
        let _ = self.iso_spv;
        apply_iso_forward(v, params);
    }
}
