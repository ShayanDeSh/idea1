use candle_nn::{Embedding, Linear, Module, RmsNorm};
use idea1_core::{
    EmbeddingModule, EngineError, EngineResult, KvCache, LinearModule, RmsNormModule, TensorOps,
};

use crate::tensor::CandleTensor;
use crate::CandleBackend;

fn map_err(err: candle_core::Error) -> EngineError {
    EngineError::backend(err.to_string())
}

/// Candle linear layer: `y = x @ W.T` (no bias; Llama projections are bias-free).
#[derive(Debug, Clone)]
pub struct CandleLinear {
    inner: Linear,
    weight: CandleTensor,
}

impl LinearModule<CandleBackend> for CandleLinear {
    fn from_weight(weight: CandleTensor) -> EngineResult<Self> {
        let inner = Linear::new(weight.inner().clone(), None);
        Ok(Self { inner, weight })
    }

    fn forward(&self, x: &CandleTensor) -> EngineResult<CandleTensor> {
        self.inner
            .forward(x.inner())
            .map(CandleTensor::from_candle)
            .map_err(map_err)
    }

    fn weight(&self) -> &CandleTensor {
        &self.weight
    }
}

/// Token embedding table.
#[derive(Debug, Clone)]
pub struct CandleEmbedding {
    inner: Embedding,
    weight: CandleTensor,
}

impl EmbeddingModule<CandleBackend> for CandleEmbedding {
    fn from_weight(weight: CandleTensor) -> EngineResult<Self> {
        let dims = weight.shape();
        if dims.len() != 2 {
            return Err(EngineError::backend(format!(
                "embedding weight must be rank-2, got {:?}",
                dims
            )));
        }
        let hidden_size = dims[1];
        let inner = Embedding::new(weight.inner().clone(), hidden_size);
        Ok(Self { inner, weight })
    }

    fn forward(&self, token_ids: &CandleTensor) -> EngineResult<CandleTensor> {
        self.inner
            .forward(token_ids.inner())
            .map(CandleTensor::from_candle)
            .map_err(map_err)
    }

    fn weight(&self) -> &CandleTensor {
        &self.weight
    }
}

/// RMSNorm module.
#[derive(Debug, Clone)]
pub struct CandleRmsNorm {
    inner: RmsNorm,
}

impl RmsNormModule<CandleBackend> for CandleRmsNorm {
    fn from_weight(weight: CandleTensor, eps: f64) -> EngineResult<Self> {
        Ok(Self {
            inner: RmsNorm::new(weight.into_inner(), eps),
        })
    }

    fn forward(&self, x: &CandleTensor) -> EngineResult<CandleTensor> {
        self.inner
            .forward(x.inner())
            .map(CandleTensor::from_candle)
            .map_err(map_err)
    }
}

/// Stub KV cache; append/read arrives in Phase 4.
#[derive(Debug, Clone, Default)]
pub struct CandleKvCache {
    len: usize,
}

impl CandleKvCache {
    pub fn new() -> Self {
        Self { len: 0 }
    }
}

impl KvCache<CandleBackend> for CandleKvCache {
    fn reset(&mut self) {
        self.len = 0;
    }

    fn current_len(&self) -> usize {
        self.len
    }
}
