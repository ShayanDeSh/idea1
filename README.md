# idea1 — Llama decode loop

Greenfield Rust workspace that implements a Llama-family autoregressive decode loop, targeting **TinyLlama 1.1B Chat** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) via Hugging Face `config.json` + `tokenizer.json` + `model.safetensors`.

First compute backend: [Candle](https://github.com/huggingface/candle) (CPU). Model and generate-loop code depend on traits in `idea1-core`, not on `candle_*` directly.

## Crates

| Crate | Role |
|-------|------|
| `idea1-core` | Backend traits, `LlamaConfig`, tokenizer, sampler, generate loop |
| `idea1-models-llama` | Llama blocks + `LlamaModel` (backend-generic) |
| `idea1-backend-candle` | Candle implementation of backend traits |
| `cli` (`idea1-infer`) | CLI binary |

```
cli → core + models-llama + backend-candle
models-llama → core only
backend-candle → core + candle-*
```

## Build

```bash
cargo build -p cli --release
cargo test --workspace
```

Requires a recent stable Rust toolchain. CPU only for now.

Unit tests use the tiny stub under `tests/fixtures/tinyllama/` and do **not** need the full checkpoint. The golden TinyLlama test runs only when `IDEA1_MODEL_DIR` is set (see below).

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

### Option B — `hf`

```bash
mkdir -p models
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama
```

**Memory note:** `generate` / golden tests load weights in f32 on CPU for correctness (~2 GB+ RAM).

## CLI

```bash
# Help
cargo run -p cli --release -- --help

# Tokenize a prompt
cargo run -p cli --release -- tokenize \
  --model-dir models/tinyllama \
  --prompt "Hello world"

# Inspect config + weight file discovery
cargo run -p cli --release -- inspect --model-dir models/tinyllama

# Greedy generation (temperature 0)
cargo run -p cli --release -- generate \
  --model-dir models/tinyllama \
  --prompt $'<|user|>\nHello</s>\n<|assistant|>\n' \
  --max-tokens 64 \
  --temperature 0

# Debug: dump last-prefill top-5 logits + print generated token ids
cargo run -p cli --release -- generate \
  --model-dir models/tinyllama \
  --prompt $'<|user|>\nHello</s>\n<|assistant|>\n' \
  --max-tokens 32 \
  --temperature 0 \
  --dump-prefill-top 5 \
  --print-token-ids

# Prefill-only top-k dump
cargo run -p cli --release -- logits \
  --model-dir models/tinyllama \
  --prompt $'<|user|>\nHello</s>\n<|assistant|>\n' \
  --top 5

# Informal CPU tok/s baseline (prefill vs decode)
cargo run -p cli --release -- bench \
  --model-dir models/tinyllama \
  --prompt $'<|user|>\nHello</s>\n<|assistant|>\n' \
  --max-tokens 32
```

## Golden test

Checked-in expectation: [`tests/fixtures/tinyllama/golden_generate.json`](tests/fixtures/tinyllama/golden_generate.json).

With weights at `models/tinyllama` (or `IDEA1_MODEL_DIR`):

```bash
IDEA1_MODEL_DIR=models/tinyllama cargo test -p idea1-models-llama --test golden_tinyllama -- --nocapture
```

Without weights the test skips and the workspace stays green.

## Fixtures

`tests/fixtures/tinyllama/` is a HF-layout stub (`config.json`, `tokenizer.json`, dummy `model.safetensors`) for unit tests without downloading the full model. `golden_generate.json` holds the real-checkpoint greedy token vector.

## Adding a backend

1. Create a new crate, e.g. `crates/backend-foo`, depending on `idea1-core` (and your native libs).
2. Implement the traits in `idea1_core::backend`:
   - `BackendOps` (associated `Device`, `Tensor`, `Linear`, `Embedding`, `RmsNorm`, `KvCache`)
   - `TensorOps` — shapes, matmul, softmax, SiLU, RoPE, cat/narrow, host `to_vec_f32`
   - `LinearModule` / `EmbeddingModule` / `RmsNormModule`
   - `KvCache` — append along sequence + reset
   - `WeightSource` — load named tensors from safetensors (or your format)
3. Keep `idea1-models-llama` and the generate loop unchanged: they are generic over `B: BackendOps`.
4. Wire the new backend in `cli` (device selection + `LlamaModel::<FooBackend>::load`).
5. Reuse the golden test / parity tests with the new backend once CPU f32 numerics match.

Keep the trait surface thin — only ops Llama needs — rather than a full tensor framework.
