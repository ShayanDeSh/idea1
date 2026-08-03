//! Rotary positional embeddings (Llama2 / TinyLlama half-split RoPE).

use idea1_core::{BackendOps, ElementDType, EngineError, EngineResult, LlamaConfig, TensorOps};

/// Precomputed cos/sin tables for RoPE.
///
/// Tables are shaped `[max_seq_len, head_dim / 2]` — the layout expected by
/// Candle's half-split `rope` op (and our [`TensorOps::apply_rope`]).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding<B: BackendOps> {
    cos: B::Tensor,
    sin: B::Tensor,
    head_dim: usize,
    max_seq_len: usize,
}

impl<B: BackendOps> RotaryEmbedding<B> {
    /// Precompute cos/sin for `0..max_seq_len` with `rope_theta` from `cfg`.
    pub fn new(cfg: &LlamaConfig, device: &B::Device, dtype: ElementDType) -> EngineResult<Self> {
        let head_dim = cfg.head_dim;
        let max_seq_len = cfg.max_seq_len;
        let half = head_dim / 2;

        // inv_freq[i] = 1 / theta^(2i / head_dim) for i in 0..head_dim/2
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / cfg.rope_theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        // freqs[pos, i] = pos * inv_freq[i]
        let mut cos_data = vec![0f32; max_seq_len * half];
        let mut sin_data = vec![0f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }

        let cos = B::Tensor::from_f32(&cos_data, &[max_seq_len, half], device)?.to_dtype(dtype)?;
        let sin = B::Tensor::from_f32(&sin_data, &[max_seq_len, half], device)?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            head_dim,
            max_seq_len,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Slice cos/sin for positions `[index_pos, index_pos + seq_len)`.
    pub fn at(&self, index_pos: usize, seq_len: usize) -> EngineResult<(B::Tensor, B::Tensor)> {
        if index_pos + seq_len > self.max_seq_len {
            return Err(EngineError::model(format!(
                "RoPE position {}+{} exceeds max_seq_len {}",
                index_pos, seq_len, self.max_seq_len
            )));
        }
        let cos = self.cos.narrow(0, index_pos, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?.contiguous()?;
        Ok((cos, sin))
    }

    /// Apply RoPE to a `[b, n_heads, seq, head_dim]` tensor at `index_pos`.
    pub fn apply(&self, x: &B::Tensor, index_pos: usize) -> EngineResult<B::Tensor> {
        let seq_len = x.dim(2)?;
        let (cos, sin) = self.at(index_pos, seq_len)?;
        x.apply_rope(&cos, &sin)
    }
}
