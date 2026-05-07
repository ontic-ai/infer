use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/kv_quant/vulkan/planar_rotate.glsl");
    println!("cargo:rerun-if-changed=src/kv_quant/vulkan/iso_rotate.glsl");

    if env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    compile_shader(
        "src/kv_quant/vulkan/planar_rotate.glsl",
        &out_dir.join("kvq_planar_rotate.spv"),
        "planar_rotate",
    );
    compile_shader(
        "src/kv_quant/vulkan/iso_rotate.glsl",
        &out_dir.join("kvq_iso_rotate.spv"),
        "iso_rotate",
    );
}

fn compile_shader(input_path: &str, output_path: &Path, name: &str) {
    let source = fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("failed to read shader {input_path}: {e}"));

    let compiler = shaderc::Compiler::new().expect("failed to create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("failed to create shader options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let artifact = compiler
        .compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            input_path,
            name,
            Some(&options),
        )
        .unwrap_or_else(|e| panic!("shader compile failure for {input_path}: {e}"));

    fs::write(output_path, artifact.as_binary_u8())
        .unwrap_or_else(|e| panic!("failed writing SPIR-V {}: {e}", output_path.display()));
}
