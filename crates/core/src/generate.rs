//! Autoregressive generate loop (prefill → decode/sample/detokenize).

use crate::backend::{BackendOps, CausalLm, TensorOps};
use crate::config::LlamaConfig;
use crate::error::{EngineError, EngineResult};
use crate::sampler::Sampler;
use crate::token_stream::TokenOutputStream;
use crate::tokenizer::TextTokenizer;

/// Configuration for a generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    /// When true, apply the tokenizer's special-token policy (typically BOS).
    pub add_special_tokens: bool,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<String>, max_new_tokens: usize) -> Self {
        Self {
            prompt: prompt.into(),
            max_new_tokens,
            add_special_tokens: true,
        }
    }
}

/// Result of a completed generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOutput {
    /// Detokenized completion text (prompt not included).
    pub text: String,
    /// Newly generated token ids (EOS not included).
    pub tokens: Vec<u32>,
    /// Prompt token ids (after encode).
    pub prompt_tokens: Vec<u32>,
}

/// Prefill → sample → decode loop over a [`CausalLm`].
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

    /// Generate completion text for `request`.
    pub fn generate(&mut self, request: &GenerateRequest) -> EngineResult<String> {
        Ok(self.generate_tokens(request)?.text)
    }

    /// Generate and return token ids plus text.
    pub fn generate_tokens(&mut self, request: &GenerateRequest) -> EngineResult<GenerateOutput> {
        self.generate_stream(request, |_| Ok(()))
    }

    /// Generate while invoking `on_text` for each stable decoded chunk.
    pub fn generate_stream<F>(
        &mut self,
        request: &GenerateRequest,
        on_text: F,
    ) -> EngineResult<GenerateOutput>
    where
        F: FnMut(&str) -> EngineResult<()>,
    {
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, request.add_special_tokens)?;
        self.generate_from_tokens_stream(&prompt_tokens, request.max_new_tokens, on_text)
    }

    /// Prefill/decode loop starting from already-encoded prompt tokens.
    pub fn generate_from_tokens(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> EngineResult<GenerateOutput> {
        self.generate_from_tokens_stream(prompt_tokens, max_new_tokens, |_| Ok(()))
    }

    /// Prefill/decode loop with a streaming text callback.
    pub fn generate_from_tokens_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        on_text: F,
    ) -> EngineResult<GenerateOutput>
    where
        F: FnMut(&str) -> EngineResult<()>,
    {
        self.generate_from_tokens_stream_with_prefill(
            prompt_tokens,
            max_new_tokens,
            on_text,
            |_| Ok(()),
        )
    }

    /// Like [`Self::generate_from_tokens_stream`], but invokes `on_prefill_logits`
    /// once after prefill with the flat host logits (for top-k dumps / debugging).
    pub fn generate_from_tokens_stream_with_prefill<F, P>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        mut on_text: F,
        mut on_prefill_logits: P,
    ) -> EngineResult<GenerateOutput>
    where
        F: FnMut(&str) -> EngineResult<()>,
        P: FnMut(&[f32]) -> EngineResult<()>,
    {
        if prompt_tokens.is_empty() {
            return Err(EngineError::model("prompt encoded to zero tokens"));
        }
        if prompt_tokens.len() >= self.config.max_seq_len {
            return Err(EngineError::model(format!(
                "prompt length {} exceeds max_seq_len {}",
                prompt_tokens.len(),
                self.config.max_seq_len
            )));
        }

        let mut stream = TokenOutputStream::new(self.tokenizer);
        let mut generated = Vec::new();
        let mut logits = self.model.prefill(prompt_tokens)?;
        let mut pos = prompt_tokens.len();

        {
            let flat = logits.to_vec_f32()?;
            on_prefill_logits(&flat)?;
        }

        for _ in 0..max_new_tokens {
            if pos >= self.config.max_seq_len {
                break;
            }
            let flat = logits.to_vec_f32()?;
            let token = self.sampler.sample(&flat)?;
            if self.config.is_eos(token) {
                break;
            }
            generated.push(token);
            if let Some(piece) = stream.next_token(token)? {
                on_text(&piece)?;
            }
            logits = self.model.decode_step(token, pos)?;
            pos += 1;
        }

        if let Some(rest) = stream.decode_rest()? {
            on_text(&rest)?;
        }

        Ok(GenerateOutput {
            text: stream.decode_all()?,
            tokens: generated,
            prompt_tokens: prompt_tokens.to_vec(),
        })
    }
}
