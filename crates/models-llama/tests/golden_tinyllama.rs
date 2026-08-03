//! Golden greedy-token test against a real TinyLlama checkpoint.
//!
//! Skips unless `IDEA1_MODEL_DIR` points at a full-size checkpoint. Default
//! `cargo test` stays fast/green without the ~2GB download.

use std::fs;
use std::path::{Path, PathBuf};

use idea1_backend_candle::{CandleBackend, CandleWeights};
use idea1_core::{
    top_k_logits, BackendOps, CausalLm, DeviceKind, Generator, GreedySampler, LlamaConfig,
    TensorOps, TextTokenizer,
};
use idea1_models_llama::LlamaModel;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenGenerate {
    prompt: String,
    add_special_tokens: bool,
    max_new_tokens: usize,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    prefill_top5_token_ids: Vec<u32>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tinyllama")
        .join(name)
}

fn resolve_model_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("IDEA1_MODEL_DIR") else {
        eprintln!(
            "skipping golden TinyLlama test: set IDEA1_MODEL_DIR to a full checkpoint \
             (e.g. models/tinyllama)"
        );
        return None;
    };
    let dir = PathBuf::from(dir);
    if has_full_weights(&dir) {
        return Some(dir);
    }
    eprintln!(
        "IDEA1_MODEL_DIR={} set but no full-size safetensors found; skipping",
        dir.display()
    );
    None
}

fn has_full_weights(dir: &Path) -> bool {
    let st = dir.join("model.safetensors");
    match fs::metadata(&st) {
        Ok(meta) => meta.is_file() && meta.len() > 1_000_000,
        Err(_) => {
            // Sharded checkpoints: index present is enough to attempt the test.
            dir.join("model.safetensors.index.json").is_file()
        }
    }
}

fn load_golden() -> GoldenGenerate {
    let path = fixture_path("golden_generate.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("read {}: {e}", path.display());
    });
    serde_json::from_str(&raw).expect("parse golden_generate.json")
}

#[test]
fn tinyllama_greedy_matches_golden_tokens() {
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };

    let golden = load_golden();
    let config = LlamaConfig::from_model_dir(&model_dir).expect("load config");
    let tokenizer = TextTokenizer::from_model_dir(&model_dir).expect("load tokenizer");

    let prompt_ids = tokenizer
        .encode(&golden.prompt, golden.add_special_tokens)
        .expect("encode prompt");
    assert_eq!(
        prompt_ids, golden.prompt_token_ids,
        "prompt tokenization drifted; refresh golden_generate.json if intentional"
    );

    let device = CandleBackend::new_device(DeviceKind::Cpu).expect("cpu device");
    let dtype = CandleBackend::dtype_for(DeviceKind::Cpu);
    let weights = CandleWeights::from_model_dir(&model_dir, &device, dtype).expect("weights");
    let mut model = LlamaModel::<CandleBackend>::load(&weights, config.clone(), &device, dtype)
        .expect("load model");

    // Prefill top-5 smoke (also validates last-position logits shape).
    let prefill_logits = model.prefill(&prompt_ids).expect("prefill");
    let flat = prefill_logits.to_vec_f32().expect("logits host");
    let top5: Vec<u32> = top_k_logits(&flat, 5).into_iter().map(|(id, _)| id).collect();
    assert_eq!(
        top5, golden.prefill_top5_token_ids,
        "prefill top-5 token ids drifted"
    );

    let mut sampler = GreedySampler;
    let mut gen = Generator::new(&config, &tokenizer, &mut model, &mut sampler);
    let out = gen
        .generate_from_tokens(&prompt_ids, golden.max_new_tokens)
        .expect("generate");

    assert_eq!(
        out.tokens, golden.generated_token_ids,
        "greedy generated token ids drifted from checked-in golden"
    );
    assert_eq!(out.prompt_tokens, golden.prompt_token_ids);
}
