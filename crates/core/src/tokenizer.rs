use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{EngineError, EngineResult};

/// Thin wrapper around Hugging Face `tokenizer.json`.
#[derive(Debug)]
pub struct TextTokenizer {
    inner: Tokenizer,
}

impl TextTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> EngineResult<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path).map_err(|err| {
            EngineError::tokenizer(format!("failed to load {}: {err}", path.display()))
        })?;
        Ok(Self { inner })
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> EngineResult<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        if !path.is_file() {
            return Err(EngineError::MissingFile(path));
        }
        Self::from_file(path)
    }

    /// Encode text to token ids.
    ///
    /// When `add_special_tokens` is true, the tokenizer applies its configured
    /// special-token policy (for TinyLlama / Llama this typically prepends BOS).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> EngineResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|err| EngineError::tokenizer(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids to a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> EngineResult<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|err| EngineError::tokenizer(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_tokenizer() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tinyllama/tokenizer.json")
    }

    #[test]
    fn tinyllama_hello_world_is_stable() {
        let tok = TextTokenizer::from_file(fixture_tokenizer()).unwrap();
        let ids = tok.encode("Hello world", true).unwrap();
        // TinyLlama / Llama tokenizer with BOS (id=1).
        assert_eq!(ids, vec![1, 15043, 3186]);
        let text = tok.decode(&ids, true).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn tinyllama_prompt_without_specials() {
        let tok = TextTokenizer::from_file(fixture_tokenizer()).unwrap();
        let ids = tok.encode("Hello world", false).unwrap();
        assert_eq!(ids, vec![15043, 3186]);
    }
}
