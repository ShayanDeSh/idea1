//! Candle backend for idea1.
//!
//! Implements [`idea1_core::BackendOps`] and related traits without exposing
//! Candle types to `models-llama` / generate-loop code.

mod device;
mod modules;
mod tensor;
mod weights;

pub use device::{candle_device, candle_dtype};
pub use modules::{CandleEmbedding, CandleKvCache, CandleLinear, CandleRmsNorm};
pub use tensor::CandleTensor;
pub use weights::CandleWeights;

use candle_core::Device;
use idea1_core::{BackendOps, DeviceKind, ElementDType, EngineResult};

/// Candle implementation of [`BackendOps`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CandleBackend;

impl BackendOps for CandleBackend {
    type Device = Device;
    type Tensor = CandleTensor;
    type Linear = CandleLinear;
    type Embedding = CandleEmbedding;
    type RmsNorm = CandleRmsNorm;
    type KvCache = CandleKvCache;

    fn new_device(kind: DeviceKind) -> EngineResult<Self::Device> {
        candle_device(kind)
    }

    fn dtype_for(kind: DeviceKind) -> ElementDType {
        idea1_core::default_dtype(kind)
    }
}

#[cfg(test)]
mod ops_tests;
