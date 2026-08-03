//! Llama decoder layer: pre-norm attention + residual + pre-norm MLP + residual.

use idea1_core::{
    BackendOps, EngineResult, LlamaConfig, RmsNormModule, TensorOps, WeightSource,
};

use crate::attention::Attention;
use crate::mlp::Mlp;
use crate::rope::RotaryEmbedding;

/// One Transformer decoder block (TinyLlama / Llama2 style).
#[derive(Debug, Clone)]
pub struct DecoderLayer<B: BackendOps> {
    input_layernorm: B::RmsNorm,
    self_attn: Attention<B>,
    post_attention_layernorm: B::RmsNorm,
    mlp: Mlp<B>,
}

impl<B: BackendOps> DecoderLayer<B> {
    pub fn new(
        input_layernorm: B::RmsNorm,
        self_attn: Attention<B>,
        post_attention_layernorm: B::RmsNorm,
        mlp: Mlp<B>,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// Load from HF names under `prefix` (e.g. `model.layers.0`).
    pub fn load(ws: &impl WeightSource<B>, prefix: &str, cfg: &LlamaConfig) -> EngineResult<Self> {
        Ok(Self {
            input_layernorm: ws.load_rms_norm(
                &format!("{prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )?,
            self_attn: Attention::load(ws, &format!("{prefix}.self_attn"), cfg)?,
            post_attention_layernorm: ws.load_rms_norm(
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?,
            mlp: Mlp::load(ws, &format!("{prefix}.mlp"), cfg)?,
        })
    }

    pub fn self_attn(&self) -> &Attention<B> {
        &self.self_attn
    }

    pub fn mlp(&self) -> &Mlp<B> {
        &self.mlp
    }

    /// `x`: `[b, seq, hidden]` → same shape.
    pub fn forward(
        &self,
        x: &B::Tensor,
        rope: &RotaryEmbedding<B>,
        index_pos: usize,
    ) -> EngineResult<B::Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, rope, index_pos)?;
        let h = h.add(&residual)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        h.add(&residual)
    }
}
