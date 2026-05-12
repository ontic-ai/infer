#[cfg(feature = "turboquant")]
use turboquant::turboquant_mse::{QuantizedVector, TurboQuantMSE};
#[cfg(feature = "turboquant")]
use turboquant::turboquant_prod::{ProdQuantized, TurboQuantProd};

#[cfg(feature = "turboquant")]
use crate::kv_quant::TurboQuantStrategy;
use crate::kv_quant::TurboQuantization;

/// TurboQuant-compressed payload for one normalized vector.
#[derive(Debug, Clone)]
pub enum TurboCompressed {
    #[cfg(feature = "turboquant")]
    Mse(QuantizedVector),
    #[cfg(feature = "turboquant")]
    Prod(ProdQuantized),
    #[cfg(not(feature = "turboquant"))]
    Unsupported,
}

impl TurboCompressed {
    pub fn encoded_len(&self) -> usize {
        match self {
            #[cfg(feature = "turboquant")]
            Self::Mse(vector) => vector.bytes().ceil() as usize,
            #[cfg(feature = "turboquant")]
            Self::Prod(vector) => vector.bytes().ceil() as usize,
            #[cfg(not(feature = "turboquant"))]
            Self::Unsupported => unreachable_turboquant(),
        }
    }
}

/// Cached TurboQuant engine for one `(dim, bits, strategy, seed)` configuration.
#[derive(Debug)]
pub enum TurboEngine {
    #[cfg(feature = "turboquant")]
    Mse(TurboQuantMSE),
    #[cfg(feature = "turboquant")]
    Prod(TurboQuantProd),
    #[cfg(not(feature = "turboquant"))]
    Unsupported,
}

impl TurboEngine {
    pub fn new(config: TurboQuantization, dim: usize) -> Self {
        #[cfg(feature = "turboquant")]
        {
            match config.strategy {
                TurboQuantStrategy::Mse => Self::Mse(
                    TurboQuantMSE::new(dim, config.bits, config.seed)
                        .expect("TurboQuant MSE configuration should be valid"),
                ),
                TurboQuantStrategy::Prod => Self::Prod(
                    TurboQuantProd::new(dim, config.bits, config.seed)
                        .expect("TurboQuant Prod configuration should be valid"),
                ),
            }
        }

        #[cfg(not(feature = "turboquant"))]
        {
            unsupported_turboquant(config, dim)
        }
    }

    pub fn compress_normalized(&self, normalized: &[f32]) -> TurboCompressed {
        #[cfg(feature = "turboquant")]
        {
            let input: Vec<f64> = normalized.iter().map(|&value| value as f64).collect();
            match self {
                Self::Mse(quantizer) => TurboCompressed::Mse(
                    quantizer
                        .quantize(&input)
                        .expect("normalized vectors should quantize with TurboQuant MSE"),
                ),
                Self::Prod(quantizer) => TurboCompressed::Prod(
                    quantizer
                        .quantize(&input)
                        .expect("normalized vectors should quantize with TurboQuant Prod"),
                ),
            }
        }

        #[cfg(not(feature = "turboquant"))]
        {
            let _ = (self, normalized);
            unreachable_turboquant()
        }
    }

    pub fn decompress_normalized(&self, compressed: &TurboCompressed) -> Vec<f32> {
        #[cfg(feature = "turboquant")]
        {
            let output = match (self, compressed) {
                (Self::Mse(quantizer), TurboCompressed::Mse(vector)) => quantizer
                    .dequantize(vector)
                    .expect("TurboQuant MSE payload should dequantize"),
                (Self::Prod(quantizer), TurboCompressed::Prod(vector)) => quantizer
                    .dequantize(vector)
                    .expect("TurboQuant Prod payload should dequantize"),
                _ => panic!("TurboQuant payload does not match the configured engine"),
            };

            output.into_iter().map(|value| value as f32).collect()
        }

        #[cfg(not(feature = "turboquant"))]
        {
            let _ = (self, compressed);
            unreachable_turboquant()
        }
    }
}

#[cfg(not(feature = "turboquant"))]
#[cold]
#[track_caller]
fn unsupported_turboquant(config: TurboQuantization, dim: usize) -> ! {
    panic!(
        "TurboQuant {:?} for dim {} requires the `turboquant` feature; enable it to pull the upstream turboquant/ort stack",
        config, dim
    )
}

#[cfg(not(feature = "turboquant"))]
#[cold]
#[track_caller]
fn unreachable_turboquant() -> ! {
    panic!("TurboQuant payloads are unavailable without the `turboquant` feature")
}
