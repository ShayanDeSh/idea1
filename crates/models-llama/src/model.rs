//! Backend-generic Llama causal LM with explicit prefill / decode KV cache.

use idea1_core::{
    BackendOps, CausalLm, ElementDType, EmbeddingModule, EngineError, EngineResult, KvCache,
    LinearModule, LlamaConfig, RmsNormModule, TensorOps, WeightSource,
};

use crate::layer::DecoderLayer;
use crate::rope::RotaryEmbedding;

/// TinyLlama / Llama2-style causal LM.
///
/// Prefill / decode use per-layer [`KvCache`]. Load a full HF checkpoint via
/// [`Self::load`], or assemble modules in tests with [`Self::new`].
pub struct LlamaModel<B: BackendOps> {
    config: LlamaConfig,
    embed_tokens: B::Embedding,
    layers: Vec<DecoderLayer<B>>,
    norm: B::RmsNorm,
    lm_head: B::Linear,
    rope: RotaryEmbedding<B>,
    caches: Vec<B::KvCache>,
    device: B::Device,
}

impl<B: BackendOps> LlamaModel<B> {
    /// Assemble a model from modules (tests / custom weight sources).
    pub fn new(
        config: LlamaConfig,
        embed_tokens: B::Embedding,
        layers: Vec<DecoderLayer<B>>,
        norm: B::RmsNorm,
        lm_head: B::Linear,
        rope: RotaryEmbedding<B>,
        device: B::Device,
    ) -> EngineResult<Self> {
        if layers.len() != config.num_hidden_layers {
            return Err(EngineError::model(format!(
                "expected {} layers, got {}",
                config.num_hidden_layers,
                layers.len()
            )));
        }
        let caches = layers.iter().map(|l| l.new_kv_cache()).collect();
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            caches,
            device,
        })
    }

    /// Load embed / layers / norm / lm_head from an HF-style weight source.
    pub fn load(
        ws: &impl WeightSource<B>,
        config: LlamaConfig,
        device: &B::Device,
        dtype: ElementDType,
    ) -> EngineResult<Self> {
        let embed_tokens = ws.load_embedding("model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::load(
                ws,
                &format!("model.layers.{i}"),
                &config,
            )?);
        }
        let norm = ws.load_rms_norm("model.norm.weight", config.rms_norm_eps)?;
        let lm_head = if config.tie_word_embeddings {
            B::Linear::from_weight(embed_tokens.weight().clone())?
        } else {
            ws.load_linear("lm_head.weight")?
        };
        let rope = RotaryEmbedding::new(&config, device, dtype)?;
        Self::new(
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            device.clone(),
        )
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn cache_lens(&self) -> Vec<usize> {
        self.caches.iter().map(|c| c.current_len()).collect()
    }

    pub fn reset_caches(&mut self) {
        for c in &mut self.caches {
            c.reset();
        }
    }

    /// Full-sequence forward without touching the KV cache (recompute baseline).
    pub fn forward_recompute(&self, tokens: &[u32]) -> EngineResult<B::Tensor> {
        if tokens.is_empty() {
            return Err(EngineError::model("forward_recompute requires at least one token"));
        }
        let hidden = self.embed(tokens)?;
        let mut h = hidden;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope, 0, None)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    fn embed(&self, tokens: &[u32]) -> EngineResult<B::Tensor> {
        let ids = B::Tensor::from_u32(tokens, &[1, tokens.len()], &self.device)?;
        self.embed_tokens.forward(&ids)
    }

    fn logits_at_last(&self, hidden: &B::Tensor) -> EngineResult<B::Tensor> {
        let seq = hidden.dim(1)?;
        let last = hidden.narrow(1, seq - 1, 1)?;
        let normed = self.norm.forward(&last)?;
        // `[1, 1, vocab]` → `[1, vocab]` for sampling convenience.
        let logits = self.lm_head.forward(&normed)?;
        logits.reshape(&[logits.dim(0)?, logits.dim(2)?])
    }

    fn forward_chunks(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
    ) -> EngineResult<B::Tensor> {
        if tokens.is_empty() {
            return Err(EngineError::model("forward requires at least one token"));
        }
        let mut h = self.embed(tokens)?;
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            h = layer.forward(&h, &self.rope, index_pos, Some(cache))?;
        }
        self.logits_at_last(&h)
    }
}

impl<B: BackendOps> CausalLm<B> for LlamaModel<B> {
    fn prefill(&mut self, tokens: &[u32]) -> EngineResult<B::Tensor> {
        self.reset_caches();
        self.forward_chunks(tokens, 0)
    }

    fn decode_step(&mut self, token: u32, pos: usize) -> EngineResult<B::Tensor> {
        let expected = self.caches.first().map(|c| c.current_len()).unwrap_or(0);
        if pos != expected {
            return Err(EngineError::model(format!(
                "decode_step pos {pos} does not match cache length {expected}"
            )));
        }
        self.forward_chunks(&[token], pos)
    }
}
