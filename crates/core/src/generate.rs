//! Autoregressive generate loop (prefill → decode/sample/detokenize).
//!
//! Stubbed in Phase 0/1; implemented in Phase 5.

use crate::backend::{BackendOps, CausalLm};
use crate::config::LlamaConfig;
use crate::error::EngineResult;
use crate::sampler::Sampler;
use crate::tokenizer::TextTokenizer;

/// Configuration for a generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
}

/// Placeholder generator; wired to `CausalLm` in Phase 5.
#[derive(Debug)]
pub struct Generator<'a, B: BackendOps, M: CausalLm<B>, S: Sampler> {
    pub config: &'a LlamaConfig,
    pub tokenizer: &'a TextTokenizer,
    pub model: &'a mut M,
    pub sampler: &'a mut S,
    _backend: std::marker::PhantomData<B>,
}

impl<'a, B: BackendOps, M: CausalLm<B>, S: Sampler> Generator<'a, B, M, S> {
    pub fn new(
        config: &'a LlamaConfig,
        tokenizer: &'a TextTokenizer,
        model: &'a mut M,
        sampler: &'a mut S,
    ) -> Self {
        Self {
            config,
            tokenizer,
            model,
            sampler,
            _backend: std::marker::PhantomData,
        }
    }

    /// Not implemented until Phase 5.
    pub fn generate(&mut self, _request: &GenerateRequest) -> EngineResult<String> {
        Err(crate::error::EngineError::Msg(
            "generate() is not implemented yet (Phase 5)".into(),
        ))
    }
}
