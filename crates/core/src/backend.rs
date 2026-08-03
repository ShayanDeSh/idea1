//! Backend abstraction surface.
//!
//! Model and generate-loop code depend on these traits only. Candle (and future
//! backends) implement them in separate crates. Hot paths use associated-type
//! generics (`B: BackendOps`) rather than `dyn Trait`.

use crate::error::EngineResult;

/// Where tensors live and how they are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda { device_id: usize },
}

/// Marker / ops bundle for a concrete backend implementation.
///
/// Associated types keep Candle (and later backends) zero-cost on hot paths.
pub trait BackendOps: Sized {
    type Device: Clone;
    type Tensor: TensorOps<Self>;
    type Linear: LinearModule<Self>;
    type Embedding: EmbeddingModule<Self>;
    type RmsNorm: RmsNormModule<Self>;
    type KvCache: KvCache<Self>;
}

/// Minimal tensor operations required by Llama layers.
///
/// Phase 0/1: stub only. Phase 2 fills in concrete ops.
pub trait TensorOps<B: BackendOps>: Clone {
    fn shape(&self) -> &[usize];
    fn device(&self) -> &B::Device;
}

pub trait LinearModule<B: BackendOps> {
    fn forward(&self, x: &B::Tensor) -> EngineResult<B::Tensor>;
}

pub trait EmbeddingModule<B: BackendOps> {
    fn forward(&self, token_ids: &B::Tensor) -> EngineResult<B::Tensor>;
}

pub trait RmsNormModule<B: BackendOps> {
    fn forward(&self, x: &B::Tensor) -> EngineResult<B::Tensor>;
}

/// Per-layer K/V cache used by prefill / decode.
pub trait KvCache<B: BackendOps> {
    fn reset(&mut self);
    fn current_len(&self) -> usize;
}

/// Autoregressive LM interface used by the generate loop.
pub trait CausalLm<B: BackendOps> {
    /// Run the full prompt; fill KV cache; return logits at the last position.
    fn prefill(&mut self, tokens: &[u32]) -> EngineResult<B::Tensor>;

    /// Decode one new token at absolute position `pos`; append to KV; return logits.
    fn decode_step(&mut self, token: u32, pos: usize) -> EngineResult<B::Tensor>;
}
