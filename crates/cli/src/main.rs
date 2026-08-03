use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use idea1_backend_candle::{CandleBackend, CandleWeights};
use idea1_core::{
    discover_weight_files, BackendOps, DeviceKind, GenerateRequest, Generator, LlamaConfig,
    StrategySampler, TextTokenizer,
};
use idea1_models_llama::LlamaModel;

#[derive(Debug, Parser)]
#[command(
    name = "idea1-infer",
    version,
    about = "TinyLlama / Llama decode loop (Candle backend)",
    long_about = "Load Hugging Face Llama-family models and run inference.\n\n\
Download TinyLlama first, then point --model-dir at the folder containing \
config.json, tokenizer.json, and model.safetensors. See README.md."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Tokenize a prompt with the model tokenizer (no weights loaded into tensors yet)
    Tokenize {
        /// Directory with config.json + tokenizer.json (+ safetensors)
        #[arg(long)]
        model_dir: PathBuf,

        /// Prompt text to encode
        #[arg(long)]
        prompt: String,

        /// Do not add special tokens (e.g. BOS)
        #[arg(long, default_value_t = false)]
        no_special_tokens: bool,

        /// Also print decoded text
        #[arg(long, default_value_t = false)]
        decode: bool,
    },

    /// Print parsed model config and discovered weight files
    Inspect {
        /// Directory with config.json + tokenizer.json (+ safetensors)
        #[arg(long)]
        model_dir: PathBuf,
    },

    /// Autoregressive text generation (prefill + decode)
    Generate {
        /// Directory with config.json + tokenizer.json + safetensors
        #[arg(long)]
        model_dir: PathBuf,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Maximum number of new tokens to generate
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        /// Sampling temperature; 0 = greedy (deterministic)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Keep only the top-k logits before sampling (temperature > 0)
        #[arg(long)]
        top_k: Option<usize>,

        /// Nucleus sampling cumulative probability cutoff (temperature > 0)
        #[arg(long)]
        top_p: Option<f32>,

        /// RNG seed for non-greedy sampling
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Do not add special tokens (e.g. BOS) when encoding the prompt
        #[arg(long, default_value_t = false)]
        no_special_tokens: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Tokenize {
            model_dir,
            prompt,
            no_special_tokens,
            decode,
        } => cmd_tokenize(model_dir, prompt, !no_special_tokens, decode),
        Commands::Inspect { model_dir } => cmd_inspect(model_dir),
        Commands::Generate {
            model_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            no_special_tokens,
        } => cmd_generate(
            model_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            !no_special_tokens,
        ),
    }
}

fn cmd_tokenize(
    model_dir: PathBuf,
    prompt: String,
    add_special_tokens: bool,
    also_decode: bool,
) -> Result<()> {
    ensure_model_dir(&model_dir)?;
    let tokenizer = TextTokenizer::from_model_dir(&model_dir)
        .with_context(|| format!("load tokenizer from {}", model_dir.display()))?;

    let ids = tokenizer
        .encode(&prompt, add_special_tokens)
        .context("encode prompt")?;

    println!("{}", serde_json::to_string(&ids)?);
    eprintln!("tokens: {}", ids.len());

    if also_decode {
        let text = tokenizer.decode(&ids, true).context("decode tokens")?;
        println!("decoded: {text}");
    }

    Ok(())
}

fn cmd_inspect(model_dir: PathBuf) -> Result<()> {
    ensure_model_dir(&model_dir)?;

    let config = LlamaConfig::from_model_dir(&model_dir)
        .with_context(|| format!("load config from {}", model_dir.display()))?;
    let weights = discover_weight_files(&model_dir)
        .with_context(|| format!("discover weights in {}", model_dir.display()))?;
    let _tokenizer = TextTokenizer::from_model_dir(&model_dir)
        .with_context(|| format!("load tokenizer from {}", model_dir.display()))?;

    println!("model_dir: {}", model_dir.display());
    println!("hidden_size: {}", config.hidden_size);
    println!("intermediate_size: {}", config.intermediate_size);
    println!("vocab_size: {}", config.vocab_size);
    println!("num_hidden_layers: {}", config.num_hidden_layers);
    println!("num_attention_heads: {}", config.num_attention_heads);
    println!("num_key_value_heads: {}", config.num_key_value_heads);
    println!("head_dim: {}", config.head_dim);
    println!("rms_norm_eps: {}", config.rms_norm_eps);
    println!("rope_theta: {}", config.rope_theta);
    println!("max_seq_len: {}", config.max_seq_len);
    println!("bos_token_id: {:?}", config.bos_token_id);
    println!("eos_token_ids: {:?}", config.eos_token_ids);
    println!("tie_word_embeddings: {}", config.tie_word_embeddings);
    println!("weight_files: {}", weights.paths.len());
    for path in &weights.paths {
        println!("  {}", path.display());
    }
    if let Some(map) = &weights.weight_map {
        println!("weight_map_entries: {}", map.len());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_generate(
    model_dir: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: u64,
    add_special_tokens: bool,
) -> Result<()> {
    ensure_model_dir(&model_dir)?;

    let config = LlamaConfig::from_model_dir(&model_dir)
        .with_context(|| format!("load config from {}", model_dir.display()))?;
    let tokenizer = TextTokenizer::from_model_dir(&model_dir)
        .with_context(|| format!("load tokenizer from {}", model_dir.display()))?;

    let device_kind = DeviceKind::Cpu;
    let device = CandleBackend::new_device(device_kind).context("create Candle CPU device")?;
    let dtype = CandleBackend::dtype_for(device_kind);

    eprintln!(
        "loading weights from {} (dtype={dtype:?}, device=cpu)…",
        model_dir.display()
    );
    let weights = CandleWeights::from_model_dir(&model_dir, &device, dtype)
        .with_context(|| format!("mmap safetensors in {}", model_dir.display()))?;
    let mut model = LlamaModel::<CandleBackend>::load(&weights, config.clone(), &device, dtype)
        .context("load LlamaModel from weights")?;
    eprintln!(
        "loaded TinyLlama-style model: {} layers, hidden {}, vocab {}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );

    let mut sampler = StrategySampler::from_knobs(temperature, top_k, top_p, seed);
    let request = GenerateRequest {
        prompt,
        max_new_tokens: max_tokens,
        add_special_tokens,
    };

    let mut generator = Generator::new(&config, &tokenizer, &mut model, &mut sampler);
    let mut stdout = io::stdout();
    let output = generator
        .generate_stream(&request, |piece| {
            print!("{piece}");
            stdout.flush().map_err(|err| {
                idea1_core::EngineError::Msg(format!("stdout flush failed: {err}"))
            })?;
            Ok(())
        })
        .context("generate")?;

    if !output.text.is_empty() && !output.text.ends_with('\n') {
        println!();
    }
    eprintln!("generated_tokens: {}", output.tokens.len());

    Ok(())
}

fn ensure_model_dir(model_dir: &PathBuf) -> Result<()> {
    if !model_dir.is_dir() {
        bail!(
            "model directory does not exist: {}\n\n\
Download TinyLlama 1.1B Chat, for example:\n  \
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\\n    \
--local-dir models/tinyllama\n\n\
Then pass --model-dir models/tinyllama",
            model_dir.display()
        );
    }
    Ok(())
}
