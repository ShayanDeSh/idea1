//! Llama / TinyLlama model blocks, generic over `idea1_core::BackendOps`.
//!
//! Phase 0/1: crate scaffold only. Layers land in Phase 3–5.

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
