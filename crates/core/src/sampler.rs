//! Host-side sampling over logits (backend-agnostic).
//!
//! Phase 0/1: trait + greedy stub only. Phase 5 wires temperature / top-k.

use crate::error::EngineResult;

/// Sampling strategy requested by the CLI / generate loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleStrategy {
    Greedy,
    Temperature { temperature: f32, top_k: Option<usize> },
}

/// Samples the next token id from a flat logits slice on the host.
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> EngineResult<u32>;
}

/// Argmax over logits.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> EngineResult<u32> {
        if logits.is_empty() {
            return Err(crate::error::EngineError::Msg(
                "cannot sample from empty logits".into(),
            ));
        }
        let mut best_idx = 0usize;
        let mut best_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&[-1.0, 3.5, 2.0]).unwrap(), 1);
    }
}
