use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use idea1_core::{discover_weight_files, LlamaConfig, TextTokenizer};

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
