//! Grouped-query causal self-attention with RoPE.

use idea1_core::{
    BackendOps, ElementDType, EngineResult, LinearModule, LlamaConfig, TensorOps, WeightSource,
};

use crate::ops::{causal_mask, dims3, masked_fill, repeat_kv};
use crate::rope::RotaryEmbedding;

/// Causal self-attention with GQA (`num_attention_heads` Q, `num_key_value_heads` KV).
///
/// Phase 3: full-sequence forward with causal mask (no KV cache yet).
#[derive(Debug, Clone)]
pub struct Attention<B: BackendOps> {
    q_proj: B::Linear,
    k_proj: B::Linear,
    v_proj: B::Linear,
    o_proj: B::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: BackendOps> Attention<B> {
    pub fn new(
        q_proj: B::Linear,
        k_proj: B::Linear,
        v_proj: B::Linear,
        o_proj: B::Linear,
        cfg: &LlamaConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    /// Load from HF names under `prefix` (e.g. `model.layers.0.self_attn`).
    pub fn load(ws: &impl WeightSource<B>, prefix: &str, cfg: &LlamaConfig) -> EngineResult<Self> {
        Ok(Self::new(
            ws.load_linear(&format!("{prefix}.q_proj.weight"))?,
            ws.load_linear(&format!("{prefix}.k_proj.weight"))?,
            ws.load_linear(&format!("{prefix}.v_proj.weight"))?,
            ws.load_linear(&format!("{prefix}.o_proj.weight"))?,
            cfg,
        ))
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Prefill-style forward over a full sequence (causal mask when `seq_len > 1`).
    ///
    /// `x`: `[b, seq, hidden]`  
    /// `rope`: shared rotary tables  
    /// `index_pos`: absolute start position (0 for a fresh prompt)
    pub fn forward(
        &self,
        x: &B::Tensor,
        rope: &RotaryEmbedding<B>,
        index_pos: usize,
    ) -> EngineResult<B::Tensor> {
        let (b, seq_len, hidden) = dims3::<B>(x)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape(&[b, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape(&[b, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        let q = rope.apply(&q, index_pos)?;
        let k = rope.apply(&k, index_pos)?;

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv::<B>(&k, n_rep)?;
        let v = repeat_kv::<B>(&v, n_rep)?;

        // Scaled dot-product attention in f32 for numerical stability.
        let in_dtype = q.dtype();
        let q = q.to_dtype(ElementDType::F32)?;
        let k = k.to_dtype(ElementDType::F32)?;
        let v = v.to_dtype(ElementDType::F32)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = q.matmul(&k.transpose(2, 3)?)?.mul_scalar(scale)?;

        let att = if seq_len == 1 {
            att
        } else {
            let mask = causal_mask::<B>(seq_len, index_pos, x.device())?;
            masked_fill::<B>(&att, &mask, f32::NEG_INFINITY)?
        };

        let att = att.softmax_dim(att.rank() - 1)?;
        let y = att
            .matmul(&v.contiguous()?)?
            .to_dtype(in_dtype)?
            .transpose(1, 2)?
            .reshape(&[b, seq_len, hidden])?;
        self.o_proj.forward(&y)
    }
}
