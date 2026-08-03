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

/// Growing K/V cache: concatenates along the sequence axis on each append.
///
/// Tensors are `[batch, n_kv_heads, seq, head_dim]` after the first write.
#[derive(Debug, Clone, Default)]
pub struct CandleKvCache {
    k: Option<CandleTensor>,
    v: Option<CandleTensor>,
    len: usize,
}

impl CandleKvCache {
    fn validate_append(k: &CandleTensor, v: &CandleTensor) -> EngineResult<usize> {
        let ks = k.shape();
        let vs = v.shape();
        if ks.len() != 4 || vs.len() != 4 {
            return Err(EngineError::backend(format!(
                "KV append expects rank-4 tensors, got k={ks:?} v={vs:?}"
            )));
        }
        if ks[0] != vs[0] || ks[1] != vs[1] || ks[2] != vs[2] || ks[3] != vs[3] {
            return Err(EngineError::backend(format!(
                "KV append shape mismatch: k={ks:?} v={vs:?}"
            )));
        }
        if ks[2] == 0 {
            return Err(EngineError::backend("KV append seq_new must be > 0"));
        }
        Ok(ks[2])
    }

    fn validate_against_cached(
        cached: &CandleTensor,
        new: &CandleTensor,
        name: &str,
    ) -> EngineResult<()> {
        let c = cached.shape();
        let n = new.shape();
        if c[0] != n[0] || c[1] != n[1] || c[3] != n[3] {
            return Err(EngineError::backend(format!(
                "KV {name} append incompatible with cache: cached={c:?} new={n:?}"
            )));
        }
        Ok(())
    }
}

impl KvCache<CandleBackend> for CandleKvCache {
    fn new() -> Self {
        Self::default()
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }

    fn current_len(&self) -> usize {
        self.len
    }

    fn append(
        &mut self,
        k: &CandleTensor,
        v: &CandleTensor,
    ) -> EngineResult<(CandleTensor, CandleTensor)> {
        let seq_new = Self::validate_append(k, v)?;

        let (full_k, full_v) = match (&self.k, &self.v) {
            (None, None) => (k.clone(), v.clone()),
            (Some(prev_k), Some(prev_v)) => {
                Self::validate_against_cached(prev_k, k, "k")?;
                Self::validate_against_cached(prev_v, v, "v")?;
                let full_k = CandleTensor::cat(&[prev_k, k], 2)?;
                let full_v = CandleTensor::cat(&[prev_v, v], 2)?;
                (full_k, full_v)
            }
            _ => {
                return Err(EngineError::backend(
                    "KV cache in inconsistent state (one of k/v missing)",
                ));
            }
        };

        self.len += seq_new;
        self.k = Some(full_k.clone());
        self.v = Some(full_v.clone());
        Ok((full_k, full_v))
    }
}
