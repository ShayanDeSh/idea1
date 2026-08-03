//! Shared engine types: config, errors, backend traits, tokenizer, weight discovery.

pub mod backend;
pub mod config;
pub mod error;
pub mod generate;
pub mod sampler;
pub mod tokenizer;
pub mod weights;

pub use backend::{
    BackendOps, CausalLm, DeviceKind, EmbeddingModule, KvCache, LinearModule, RmsNormModule,
    TensorOps,
};
pub use config::{HfLlamaConfig, LlamaConfig};
pub use error::{EngineError, EngineResult};
pub use generate::Generator;
pub use sampler::{GreedySampler, SampleStrategy, Sampler};
pub use tokenizer::TextTokenizer;
pub use weights::{discover_weight_files, WeightFiles};
