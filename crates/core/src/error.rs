use std::path::PathBuf;

use thiserror::Error;

/// Top-level error type for the inference engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("missing model file: {0}")]
    MissingFile(PathBuf),

    #[error("invalid model directory `{path}`: {reason}")]
    InvalidModelDir { path: PathBuf, reason: String },

    #[error("weight discovery error: {0}")]
    Weights(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("{0}")]
    Msg(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

impl EngineError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }

    pub fn weights(msg: impl Into<String>) -> Self {
        Self::Weights(msg.into())
    }

    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }

    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }
}
