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

/// Element type used for model compute.
///
/// CPU correctness path uses [`ElementDType::F32`]. GPU may opt into lower
/// precision later behind the same policy helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementDType {
    F32,
    F16,
    BF16,
}

/// Default dtype policy for a device kind.
pub fn default_dtype(kind: DeviceKind) -> ElementDType {
    match kind {
        DeviceKind::Cpu => ElementDType::F32,
        // Prefer f32 for correctness until Phase 6+ mixed-precision work.
        DeviceKind::Cuda { .. } => ElementDType::F32,
    }
}

/// Marker / ops bundle for a concrete backend implementation.
///
/// Associated types keep Candle (and later backends) zero-cost on hot paths.
pub trait BackendOps: Sized {
    type Device: Clone;
    type Tensor: TensorOps<Self>;
    type Linear: LinearModule<Self> + Clone + std::fmt::Debug;
    type Embedding: EmbeddingModule<Self> + Clone + std::fmt::Debug;
    type RmsNorm: RmsNormModule<Self> + Clone + std::fmt::Debug;
    type KvCache: KvCache<Self>;

    /// Create a device handle for `kind`.
    fn new_device(kind: DeviceKind) -> EngineResult<Self::Device>;

    /// Dtype used when loading weights / running compute on `kind`.
    fn dtype_for(kind: DeviceKind) -> ElementDType {
        default_dtype(kind)
    }
}

/// Minimal tensor operations required by Llama layers.
pub trait TensorOps<B: BackendOps>: Clone + Sized {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> ElementDType;
    fn device(&self) -> &B::Device;

    fn rank(&self) -> usize {
        self.shape().len()
    }

    fn dim(&self, axis: usize) -> EngineResult<usize> {
        self.shape().get(axis).copied().ok_or_else(|| {
            crate::error::EngineError::backend(format!(
                "dim {axis} out of range for rank {}",
                self.rank()
            ))
        })
    }

    // --- constructors ---

    fn from_f32(data: &[f32], shape: &[usize], device: &B::Device) -> EngineResult<Self>;
    fn from_u32(data: &[u32], shape: &[usize], device: &B::Device) -> EngineResult<Self>;
    fn from_u8(data: &[u8], shape: &[usize], device: &B::Device) -> EngineResult<Self>;
    fn zeros(shape: &[usize], dtype: ElementDType, device: &B::Device) -> EngineResult<Self>;

    // --- shape / layout ---

    fn reshape(&self, shape: &[usize]) -> EngineResult<Self>;
    fn transpose(&self, dim1: usize, dim2: usize) -> EngineResult<Self>;
    fn narrow(&self, dim: usize, start: usize, len: usize) -> EngineResult<Self>;
    fn unsqueeze(&self, dim: usize) -> EngineResult<Self>;
    fn broadcast_as(&self, shape: &[usize]) -> EngineResult<Self>;
    fn contiguous(&self) -> EngineResult<Self>;
    fn cat(tensors: &[&Self], dim: usize) -> EngineResult<Self>;

    // --- math used by Llama ---

    fn matmul(&self, rhs: &Self) -> EngineResult<Self>;
    fn softmax_dim(&self, dim: usize) -> EngineResult<Self>;
    fn silu(&self) -> EngineResult<Self>;
    /// Apply Llama-style (half-split) RoPE. `self` is `[b, n_heads, seq, head_dim]`.
    fn apply_rope(&self, cos: &Self, sin: &Self) -> EngineResult<Self>;

    fn add(&self, rhs: &Self) -> EngineResult<Self>;
    fn mul(&self, rhs: &Self) -> EngineResult<Self>;
    fn broadcast_add(&self, rhs: &Self) -> EngineResult<Self>;
    fn broadcast_mul(&self, rhs: &Self) -> EngineResult<Self>;
    fn mul_scalar(&self, v: f64) -> EngineResult<Self>;
    /// Element-wise select: `self` is a 0/1 mask; where mask≠0 take `on_true`, else `on_false`.
    fn where_cond(&self, on_true: &Self, on_false: &Self) -> EngineResult<Self>;

    fn to_dtype(&self, dtype: ElementDType) -> EngineResult<Self>;

    /// Flatten to host f32 (for tests and sampling).
    fn to_vec_f32(&self) -> EngineResult<Vec<f32>>;
}

pub trait LinearModule<B: BackendOps>: Sized {
    /// `weight` shape is `[out_features, in_features]` (HF / Candle convention).
    fn from_weight(weight: B::Tensor) -> EngineResult<Self>;
    fn forward(&self, x: &B::Tensor) -> EngineResult<B::Tensor>;
    fn weight(&self) -> &B::Tensor;
}

pub trait EmbeddingModule<B: BackendOps>: Sized {
    /// `weight` shape is `[vocab_size, hidden_size]`.
    fn from_weight(weight: B::Tensor) -> EngineResult<Self>;
    fn forward(&self, token_ids: &B::Tensor) -> EngineResult<B::Tensor>;
    fn weight(&self) -> &B::Tensor;
}

pub trait RmsNormModule<B: BackendOps>: Sized {
    fn from_weight(weight: B::Tensor, eps: f64) -> EngineResult<Self>;
    fn forward(&self, x: &B::Tensor) -> EngineResult<B::Tensor>;
}

/// Per-layer K/V cache used by prefill / decode.
///
/// Append / read semantics arrive in Phase 4; Phase 2 only needs the stub surface.
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

/// Read named tensors from a weight store (safetensors, etc.).
pub trait WeightSource<B: BackendOps> {
    fn tensor_names(&self) -> EngineResult<Vec<String>>;
    fn contains(&self, name: &str) -> bool;
    fn get_tensor(&self, name: &str) -> EngineResult<B::Tensor>;

    fn load_linear(&self, name: &str) -> EngineResult<B::Linear> {
        B::Linear::from_weight(self.get_tensor(name)?)
    }

    fn load_embedding(&self, name: &str) -> EngineResult<B::Embedding> {
        B::Embedding::from_weight(self.get_tensor(name)?)
    }

    fn load_rms_norm(&self, name: &str, eps: f64) -> EngineResult<B::RmsNorm> {
        B::RmsNorm::from_weight(self.get_tensor(name)?, eps)
    }
}
