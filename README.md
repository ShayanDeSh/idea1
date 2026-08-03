# idea1 — Llama decode loop

Greenfield Rust workspace that implements a Llama-family autoregressive decode loop, targeting **TinyLlama 1.1B Chat** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) via Hugging Face `config.json` + `tokenizer.json` + `model.safetensors`.

First compute backend: [Candle](https://github.com/huggingface/candle) (CPU). Model code depends on traits in `idea1-core`, not on `candle_*` directly.

## Crates

| Crate | Role |
|-------|------|
| `idea1-core` | Backend traits, `LlamaConfig`, tokenizer, weight discovery, sampler/generate stubs |
| `idea1-models-llama` | Llama blocks (backend-generic; layers in later phases) |
| `idea1-backend-candle` | Candle implementation of backend traits |
| `cli` (`idea1-infer`) | CLI binary |

## Build

```bash
cargo build -p cli
cargo test --workspace
```

Requires a recent stable/nightly Rust toolchain. CPU only for now.

## Download TinyLlama

You need a local directory with at least:

- `config.json`
- `tokenizer.json`
- `model.safetensors` (~2.2 GB; single file for this model)

### Option A — Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir models/tinyllama
```

### Option B — `hf` / git LFS

```bash
mkdir -p models
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama
```

**Memory note:** Phase 5 will load weights in f32 on CPU for correctness (~2 GB+ RAM). Keep that in mind before running generate.

## CLI (current)

```bash
# Help
cargo run -p cli -- --help

# Tokenize a prompt (Phase 1)
cargo run -p cli -- tokenize \
  --model-dir models/tinyllama \
  --prompt "Hello world"

# Inspect config + weight file discovery
cargo run -p cli -- inspect --model-dir models/tinyllama
```

`generate` arrives in a later phase.

## Fixtures

`tests/fixtures/tinyllama/` is a HF-layout stub (`config.json`, `tokenizer.json`, dummy `model.safetensors`) for unit tests and CLI smoke checks without downloading the full model.
