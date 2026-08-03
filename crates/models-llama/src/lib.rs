//! Llama / TinyLlama model blocks, generic over [`idea1_core::BackendOps`].
//!
//! Phase 3 delivers RMSNorm (via backend), RoPE, GQA attention, SwiGLU MLP, and
//! a pre-norm decoder layer. Full CausalLm + KV cache arrive in later phases.

mod attention;
mod layer;
mod mlp;
mod ops;
mod rope;

pub use attention::Attention;
pub use layer::DecoderLayer;
pub use mlp::Mlp;
pub use ops::{causal_mask, dims3, dims4, masked_fill, repeat_kv};
pub use rope::RotaryEmbedding;

use idea1_core::LlamaConfig;

/// Placeholder for the eventual backend-generic TinyLlama / Llama causal LM.
#[derive(Debug, Clone)]
pub struct LlamaModel<B> {
    pub config: LlamaConfig,
    _backend: std::marker::PhantomData<B>,
}

impl<B> LlamaModel<B> {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests;
