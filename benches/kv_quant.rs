//! Benchmarks for the kv_quant CPU pipeline.
//!
//! Run with:
//!   cargo bench --bench kv_quant
//!
//! Nine benchmark groups are registered:
//!   1. kv_round_trip  — compress_k then decompress, all variants × dims
//!   2. kv_compress    — compress_k only, all variants × dims
//!   3. kv_decompress  — decompress only (pre-compressed), all variants × dims
//!   4. rotation       — individual kernel calls, all kernels × dims
//!   5. scalar         — quantize_scalar / dequantize_scalar, all bit-widths
//!   6. kv_buffered_round_trip  — reusable-buffer round-trip batch benches
//!   7. kv_buffered_compress    — reusable-buffer compress batch benches
//!   8. kv_buffered_decompress  — reusable-buffer decompress batch benches
//!   9. kv_batch                — public-API batch throughput bench

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use infer::kv_quant::pipeline::{CompressedKv, KvQuantizer};
use infer::kv_quant::rotation::{
    GivensParams, QuaternionParams, apply_iso_forward, apply_iso_inverse, apply_planar_forward,
    apply_planar_inverse, centroids, dequantize_scalar, quantize_scalar,
};
use infer::kv_quant::{KvCacheConfig, KvQuantization};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dimensions to benchmark.
const DIMS: &[usize] = &[128, 256, 512];

/// Number of vectors in the batch throughput group.
const BATCH_SIZE: usize = 32;

/// Number of vectors in reusable-buffer groups.
const BUFFERED_BATCH_SIZE: usize = 64;

/// Number of vectors per rotation kernel sample to keep timing above the noise floor.
const ROTATION_BATCH_SIZE: usize = 128;

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .confidence_level(0.99)
        .noise_threshold(0.02)
}

/// Deterministic unit vector — same values every run, not random so that
/// the benchmark harness does not influence the result through cache effects.
fn make_unit_vec(dim: usize) -> Vec<f32> {
    make_unit_vec_seeded(dim, 0)
}

fn make_unit_vec_seeded(dim: usize, seed: usize) -> Vec<f32> {
    let phase = seed as f32 * 0.031_25;
    let mut v: Vec<f32> = (0..dim)
        .map(|i| {
            ((i as f32 + 1.0 + phase) * 0.123_45).sin()
                + ((i as f32 + 7.0 + phase * 3.0) * 0.077).cos()
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn make_unit_batch(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|seed| make_unit_vec_seeded(dim, seed + 1))
        .collect()
}

/// Rotation params using the same formula as `KvQuantizer::new`.
fn make_planar_params(dim: usize) -> Vec<GivensParams> {
    (0..(dim / 2))
        .map(|i| {
            let theta = (i as f32 * 0.173_205_08).sin();
            GivensParams {
                cos_theta: theta.cos(),
                sin_theta: theta.sin(),
            }
        })
        .collect()
}

fn make_iso_params(dim: usize) -> Vec<QuaternionParams> {
    (0..(dim / 4))
        .map(|i| {
            let t = (i + 1) as f32;
            QuaternionParams::new(1.0, (0.37 * t).sin(), (0.53 * t).cos(), (0.71 * t).sin())
        })
        .collect()
}

/// All four quantization variants used in benchmarks (excludes `None`).
const VARIANTS: &[(KvQuantization, &str)] = &[
    (KvQuantization::planar2(), "Planar2"),
    (KvQuantization::planar3(), "Planar3"),
    (KvQuantization::iso4(), "Iso4"),
    (KvQuantization::iso3(), "Iso3"),
    #[cfg(feature = "turboquant")]
    (KvQuantization::turbo_mse(4), "TurboMSE4"),
    #[cfg(feature = "turboquant")]
    (KvQuantization::turbo_prod(4), "TurboProd4"),
];

fn make_quantizer(quant: KvQuantization, dim: usize) -> KvQuantizer {
    KvQuantizer::new(
        KvCacheConfig {
            k: quant,
            v: KvQuantization::None,
        },
        dim,
    )
}

/// Pre-compress a vector so decompress-only benchmarks don't time compression.
fn pre_compress(quant: KvQuantization, dim: usize) -> CompressedKv {
    let q = make_quantizer(quant, dim);
    q.compress_k(&make_unit_vec(dim))
}

// ---------------------------------------------------------------------------
// Group 1: round-trip (compress_k → decompress)
// ---------------------------------------------------------------------------

fn bench_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_round_trip");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let v = make_unit_vec(dim);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| {
                    let compressed = q.compress_k(black_box(&v));
                    black_box(q.decompress(&compressed))
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: compress-only
// ---------------------------------------------------------------------------

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_compress");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let v = make_unit_vec(dim);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| black_box(q.compress_k(black_box(&v))));
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: decompress-only
// ---------------------------------------------------------------------------

fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_decompress");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let compressed = pre_compress(quant, dim);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| black_box(q.decompress(black_box(&compressed))));
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: batch throughput (BATCH_SIZE vectors through compress_k)
// ---------------------------------------------------------------------------

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_batch");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let vecs: Vec<Vec<f32>> = (0..BATCH_SIZE).map(|_| make_unit_vec(dim)).collect();
            let id = format!("{label}/{BATCH_SIZE}x{dim}");
            group.throughput(Throughput::Elements((BATCH_SIZE * dim) as u64));
            group.bench_function(&id, |b| {
                b.iter(|| {
                    for v in &vecs {
                        black_box(q.compress_k(black_box(v)));
                    }
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Buffered groups: allocator-free algorithm timing
// ---------------------------------------------------------------------------

fn bench_buffered_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_buffered_round_trip");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let inputs = make_unit_batch(dim, BUFFERED_BATCH_SIZE);
            let mut scratch = vec![vec![0.0; dim]; BUFFERED_BATCH_SIZE];
            let mut compressed = vec![CompressedKv::default(); BUFFERED_BATCH_SIZE];
            let mut outputs = vec![vec![0.0; dim]; BUFFERED_BATCH_SIZE];

            group.throughput(Throughput::Elements((BUFFERED_BATCH_SIZE * dim) as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| {
                    for index in 0..inputs.len() {
                        q.compress_k_into(
                            black_box(&inputs[index]),
                            black_box(&mut scratch[index]),
                            &mut compressed[index],
                        );
                        q.decompress_into(&compressed[index], &mut outputs[index]);
                    }
                    black_box(&outputs);
                });
            });
        }
    }
    group.finish();
}

fn bench_buffered_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_buffered_compress");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let inputs = make_unit_batch(dim, BUFFERED_BATCH_SIZE);
            let mut scratch = vec![vec![0.0; dim]; BUFFERED_BATCH_SIZE];
            let mut compressed = vec![CompressedKv::default(); BUFFERED_BATCH_SIZE];

            group.throughput(Throughput::Elements((BUFFERED_BATCH_SIZE * dim) as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| {
                    for index in 0..inputs.len() {
                        q.compress_k_into(
                            black_box(&inputs[index]),
                            black_box(&mut scratch[index]),
                            &mut compressed[index],
                        );
                    }
                    black_box(&compressed);
                });
            });
        }
    }
    group.finish();
}

fn bench_buffered_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_buffered_decompress");
    for &(quant, label) in VARIANTS {
        for &dim in DIMS {
            let q = make_quantizer(quant, dim);
            let inputs = make_unit_batch(dim, BUFFERED_BATCH_SIZE);
            let mut scratch = vec![vec![0.0; dim]; BUFFERED_BATCH_SIZE];
            let mut compressed = vec![CompressedKv::default(); BUFFERED_BATCH_SIZE];
            let mut outputs = vec![vec![0.0; dim]; BUFFERED_BATCH_SIZE];

            for index in 0..inputs.len() {
                q.compress_k_into(&inputs[index], &mut scratch[index], &mut compressed[index]);
            }

            group.throughput(Throughput::Elements((BUFFERED_BATCH_SIZE * dim) as u64));
            group.bench_with_input(BenchmarkId::new(label, dim), &dim, |b, _| {
                b.iter(|| {
                    for index in 0..compressed.len() {
                        q.decompress_into(
                            black_box(&compressed[index]),
                            black_box(&mut outputs[index]),
                        );
                    }
                    black_box(&outputs);
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: rotation kernels in isolation
// ---------------------------------------------------------------------------

fn bench_rotation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation");
    for &dim in DIMS {
        let planar = make_planar_params(dim);
        let iso = make_iso_params(dim);
        let inputs = make_unit_batch(dim, ROTATION_BATCH_SIZE);
        let elements = (ROTATION_BATCH_SIZE * dim) as u64;

        group.throughput(Throughput::Elements(elements));

        group.bench_with_input(
            BenchmarkId::new(format!("planar_forward/{ROTATION_BATCH_SIZE}x"), dim),
            &dim,
            |b, _| {
                b.iter_batched_ref(
                    || inputs.clone(),
                    |batch| {
                        for v in batch.iter_mut() {
                            apply_planar_forward(black_box(v), black_box(&planar));
                        }
                        black_box(batch);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("planar_inverse/{ROTATION_BATCH_SIZE}x"), dim),
            &dim,
            |b, _| {
                b.iter_batched_ref(
                    || inputs.clone(),
                    |batch| {
                        for v in batch.iter_mut() {
                            apply_planar_inverse(black_box(v), black_box(&planar));
                        }
                        black_box(batch);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("iso_forward/{ROTATION_BATCH_SIZE}x"), dim),
            &dim,
            |b, _| {
                b.iter_batched_ref(
                    || inputs.clone(),
                    |batch| {
                        for v in batch.iter_mut() {
                            apply_iso_forward(black_box(v), black_box(&iso));
                        }
                        black_box(batch);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("iso_inverse/{ROTATION_BATCH_SIZE}x"), dim),
            &dim,
            |b, _| {
                b.iter_batched_ref(
                    || inputs.clone(),
                    |batch| {
                        for v in batch.iter_mut() {
                            apply_iso_inverse(black_box(v), black_box(&iso));
                        }
                        black_box(batch);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 6: scalar quantize / dequantize throughput
// ---------------------------------------------------------------------------

fn bench_scalar(c: &mut Criterion) {
    const N: usize = 1024;
    let mut group = c.benchmark_group("scalar");

    let centroid_cases: &[(&[f32], &str)] = &[
        (&centroids::BITS_2, "2bit"),
        (&centroids::BITS_3, "3bit"),
        (&centroids::BITS_4, "4bit"),
    ];

    let values: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.007).sin() * 2.0).collect();

    for &(table, label) in centroid_cases {
        let ids: Vec<u8> = values.iter().map(|&x| quantize_scalar(x, table)).collect();

        group.throughput(Throughput::Elements(N as u64));
        group.bench_function(format!("quantize/{label}"), |b| {
            b.iter(|| {
                for &x in &values {
                    black_box(quantize_scalar(black_box(x), black_box(table)));
                }
            });
        });

        group.bench_function(format!("dequantize/{label}"), |b| {
            b.iter(|| {
                for &idx in &ids {
                    black_box(dequantize_scalar(black_box(idx), black_box(table)));
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        bench_round_trip,
        bench_compress,
        bench_decompress,
        bench_batch,
        bench_buffered_round_trip,
        bench_buffered_compress,
        bench_buffered_decompress,
        bench_rotation,
        bench_scalar,
}
criterion_main!(benches);
