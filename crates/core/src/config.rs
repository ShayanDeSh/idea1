use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::Deserialize;

use crate::error::{EngineError, EngineResult};

/// Raw Hugging Face `config.json` fields for Llama-family models.
#[derive(Debug, Clone, Deserialize)]
pub struct HfLlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl HfLlamaConfig {
    pub fn from_path(path: impl AsRef<Path>) -> EngineResult<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let cfg: Self = serde_json::from_reader(reader)?;
        Ok(cfg)
    }

    pub fn into_llama_config(self) -> EngineResult<LlamaConfig> {
        if let Some(model_type) = &self.model_type {
            if model_type != "llama" {
                return Err(EngineError::config(format!(
                    "unsupported model_type `{model_type}` (expected `llama`)"
                )));
            }
        }

        let num_key_value_heads = self
            .num_key_value_heads
            .unwrap_or(self.num_attention_heads);

        if self.num_attention_heads == 0 {
            return Err(EngineError::config("num_attention_heads must be > 0"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(EngineError::config(format!(
                "hidden_size ({}) is not divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if num_key_value_heads == 0
            || self.num_attention_heads % num_key_value_heads != 0
        {
            return Err(EngineError::config(format!(
                "invalid GQA heads: num_attention_heads={}, num_key_value_heads={}",
                self.num_attention_heads, num_key_value_heads
            )));
        }

        let eos_token_ids = match self.eos_token_id {
            Some(LlamaEosToks::Single(id)) => vec![id],
            Some(LlamaEosToks::Multiple(ids)) => ids,
            None => Vec::new(),
        };

        Ok(LlamaConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_ids,
            max_seq_len: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        })
    }
}

/// Engine-facing Llama configuration (normalized from HF `config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    pub max_seq_len: usize,
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    pub fn from_hf_path(path: impl AsRef<Path>) -> EngineResult<Self> {
        HfLlamaConfig::from_path(path)?.into_llama_config()
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> EngineResult<Self> {
        let path = model_dir.as_ref().join("config.json");
        if !path.is_file() {
            return Err(EngineError::MissingFile(path));
        }
        Self::from_hf_path(path)
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_config() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tinyllama/config.json")
    }

    #[test]
    fn parses_tinyllama_config() {
        let cfg = LlamaConfig::from_hf_path(fixture_config()).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 5632);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 10000.0);
        assert_eq!(cfg.bos_token_id, Some(1));
        assert_eq!(cfg.eos_token_ids, vec![2]);
        assert_eq!(cfg.max_seq_len, 2048);
        assert!(!cfg.tie_word_embeddings);
    }
}
