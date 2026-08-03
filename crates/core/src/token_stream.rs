//! Streaming detokenization (Candle `TokenOutputStream`-style).
//!
//! Buffers incomplete BPE / UTF-8 pieces so callers only see stable text chunks.

use crate::error::EngineResult;
use crate::tokenizer::TextTokenizer;

/// Incremental decoder that emits text only when a stable prefix is available.
#[derive(Debug)]
pub struct TokenOutputStream<'a> {
    tokenizer: &'a TextTokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    pub fn new(tokenizer: &'a TextTokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Push one generated token; return a newly available text chunk if any.
    
    pub fn next_token(&mut self, token: u32) -> EngineResult<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let last = text.chars().last();
            if last.is_some_and(|c| c.is_alphanumeric()) {
                let chunk = text[prev_text.len()..].to_string();
                self.prev_index = self.current_index;
                self.current_index = self.tokens.len();
                return Ok(Some(chunk));
            }
        }
        Ok(None)
    }

    /// Flush any remaining buffered text after generation stops.
    pub fn decode_rest(&self) -> EngineResult<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            Ok(Some(text[prev_text.len()..].to_string()))
        } else {
            Ok(None)
        }
    }

    /// Decode all accumulated tokens.
    pub fn decode_all(&self) -> EngineResult<String> {
        self.decode(&self.tokens)
    }

    fn decode(&self, tokens: &[u32]) -> EngineResult<String> {
        self.tokenizer.decode(tokens, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_tokenizer() -> TextTokenizer {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tinyllama/tokenizer.json");
        TextTokenizer::from_file(path).unwrap()
    }

    #[test]
    fn streams_hello_world_pieces() {
        let tok = fixture_tokenizer();
        // "Hello world" without BOS: [15043, 3186]
        let mut stream = TokenOutputStream::new(&tok);
        let mut out = String::new();
        for id in [15043u32, 3186] {
            if let Some(piece) = stream.next_token(id).unwrap() {
                out.push_str(&piece);
            }
        }
        if let Some(rest) = stream.decode_rest().unwrap() {
            out.push_str(&rest);
        }
        assert_eq!(out, "Hello world");
        assert_eq!(stream.decode_all().unwrap(), "Hello world");
    }
}
