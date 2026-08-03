//! SwiGLU feed-forward block (`gate_proj` / `up_proj` / `down_proj`).

use idea1_core::{
    BackendOps, EngineResult, LinearModule, LlamaConfig, TensorOps, WeightSource,
};

/// Llama MLP: `down(silu(gate(x)) * up(x))`.
#[derive(Debug, Clone)]
pub struct Mlp<B: BackendOps> {
    gate_proj: B::Linear,
    up_proj: B::Linear,
    down_proj: B::Linear,
}

impl<B: BackendOps> Mlp<B> {
    pub fn new(gate_proj: B::Linear, up_proj: B::Linear, down_proj: B::Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Load from HF names under `prefix` (e.g. `model.layers.0.mlp`).
    pub fn load(ws: &impl WeightSource<B>, prefix: &str, _cfg: &LlamaConfig) -> EngineResult<Self> {
        Ok(Self {
            gate_proj: ws.load_linear(&format!("{prefix}.gate_proj.weight"))?,
            up_proj: ws.load_linear(&format!("{prefix}.up_proj.weight"))?,
            down_proj: ws.load_linear(&format!("{prefix}.down_proj.weight"))?,
        })
    }

    /// `x`: `[..., hidden]` → `[..., hidden]`.
    pub fn forward(&self, x: &B::Tensor) -> EngineResult<B::Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}
