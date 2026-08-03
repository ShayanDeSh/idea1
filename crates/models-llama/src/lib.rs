//! Llama / TinyLlama model blocks, generic over [`idea1_core::BackendOps`].
//!
//! [`LlamaModel`] implements [`idea1_core::CausalLm`] with KV-cached prefill/decode.
//! Sampling and the generate CLI live in `idea1-core` / `cli`.

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
