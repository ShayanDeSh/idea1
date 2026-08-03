//! Host-side sampling over logits (backend-agnostic).

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::{EngineError, EngineResult};

/// Sampling strategy requested by the CLI / generate loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleStrategy {
    Greedy,
    Temperature {
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    },
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
        greedy_sample(logits)
    }
}

/// Strategy-driven sampler (greedy, or temperature + optional top-k / top-p).
#[derive(Debug)]
pub struct StrategySampler {
    strategy: SampleStrategy,
    rng: StdRng,
}

impl StrategySampler {
    pub fn new(strategy: SampleStrategy, seed: u64) -> Self {
        Self {
            strategy,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Build a sampler from CLI-style knobs. `temperature <= 0` → greedy.
    pub fn from_knobs(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        seed: u64,
    ) -> Self {
        let strategy = if temperature <= 0.0 {
            SampleStrategy::Greedy
        } else {
            SampleStrategy::Temperature {
                temperature,
                top_k,
                top_p,
            }
        };
        Self::new(strategy, seed)
    }

    pub fn strategy(&self) -> SampleStrategy {
        self.strategy
    }
}

impl Sampler for StrategySampler {
    fn sample(&mut self, logits: &[f32]) -> EngineResult<u32> {
        match self.strategy {
            SampleStrategy::Greedy => greedy_sample(logits),
            SampleStrategy::Temperature {
                temperature,
                top_k,
                top_p,
            } => temperature_sample(logits, temperature, top_k, top_p, &mut self.rng),
        }
    }
}

fn greedy_sample(logits: &[f32]) -> EngineResult<u32> {
    if logits.is_empty() {
        return Err(EngineError::Msg("cannot sample from empty logits".into()));
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

fn temperature_sample(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut StdRng,
) -> EngineResult<u32> {
    if logits.is_empty() {
        return Err(EngineError::Msg("cannot sample from empty logits".into()));
    }
    if temperature <= 0.0 {
        return greedy_sample(logits);
    }

    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();

    if let Some(k) = top_k {
        if k == 0 {
            return Err(EngineError::Msg("top_k must be > 0 when set".into()));
        }
        if k < scaled.len() {
            scaled.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scaled.truncate(k);
        }
    }

    // Softmax over the (possibly top-k filtered) set.
    let max_logit = scaled
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<(usize, f32)> = scaled
        .into_iter()
        .map(|(i, v)| (i, (v - max_logit).exp()))
        .collect();
    let sum: f32 = weights.iter().map(|(_, w)| *w).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(EngineError::Msg("invalid softmax weights during sampling".into()));
    }
    for (_, w) in &mut weights {
        *w /= sum;
    }

    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(EngineError::Msg(format!(
                "top_p must be in [0, 1], got {p}"
            )));
        }
        if p < 1.0 {
            weights.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cum = 0.0f32;
            let mut cutoff = weights.len();
            for (i, (_, w)) in weights.iter().enumerate() {
                cum += *w;
                if cum >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            weights.truncate(cutoff.max(1));
            let sum: f32 = weights.iter().map(|(_, w)| *w).sum();
            for (_, w) in &mut weights {
                *w /= sum;
            }
        }
    }

    let r: f32 = rng.random();
    let mut cum = 0.0f32;
    for (idx, w) in &weights {
        cum += *w;
        if r <= cum {
            return Ok(*idx as u32);
        }
    }
    Ok(weights.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&[-1.0, 3.5, 2.0]).unwrap(), 1);
    }

    #[test]
    fn temperature_zero_via_strategy_is_greedy() {
        let mut s = StrategySampler::from_knobs(0.0, None, None, 0);
        assert_eq!(s.strategy(), SampleStrategy::Greedy);
        assert_eq!(s.sample(&[-1.0, 3.5, 2.0]).unwrap(), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut s = StrategySampler::from_knobs(1.0, Some(1), None, 42);
        // Only the top logit survives top-k=1, so sampling is deterministic.
        assert_eq!(s.sample(&[1.0, 10.0, 2.0, 3.0]).unwrap(), 1);
    }

    #[test]
    fn top_p_near_zero_keeps_best() {
        let mut s = StrategySampler::from_knobs(1.0, None, Some(0.01), 7);
        assert_eq!(s.sample(&[0.0, 5.0, 0.1]).unwrap(), 1);
    }

    #[test]
    fn empty_logits_error() {
        let mut s = GreedySampler;
        assert!(s.sample(&[]).is_err());
    }
}
