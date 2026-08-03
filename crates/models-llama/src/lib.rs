//! Llama / TinyLlama model blocks, generic over [`idea1_core::BackendOps`].
//!
//! Phase 4: KV cache + prefill/decode split on [`LlamaModel`] / attention layers.
//! Phase 5 adds sampling + the generate CLI on top of this CausalLm.

mod attention;
mod layer;
mod mlp;
mod model;
mod ops;
mod rope;

pub use attention::Attention;
pub use layer::DecoderLayer;
pub use mlp::Mlp;
pub use model::LlamaModel;
pub use ops::{causal_mask, dims3, dims4, masked_fill, repeat_kv};
pub use rope::RotaryEmbedding;

#[cfg(test)]
mod tests;
