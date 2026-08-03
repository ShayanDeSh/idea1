//! Shared tensor helpers used by Llama layers (backend-generic).

use idea1_core::{BackendOps, EngineError, EngineResult, TensorOps};

/// Unpack a rank-3 tensor as `(b, seq, hidden)`.
pub fn dims3<B: BackendOps>(t: &B::Tensor) -> EngineResult<(usize, usize, usize)> {
    let s = t.shape();
    if s.len() != 3 {
        return Err(EngineError::model(format!(
            "expected rank-3 tensor, got {:?}",
            s
        )));
    }
    Ok((s[0], s[1], s[2]))
}

/// Unpack a rank-4 tensor as `(b, heads, seq, head_dim)`.
pub fn dims4<B: BackendOps>(t: &B::Tensor) -> EngineResult<(usize, usize, usize, usize)> {
    let s = t.shape();
    if s.len() != 4 {
        return Err(EngineError::model(format!(
            "expected rank-4 tensor, got {:?}",
            s
        )));
    }
    Ok((s[0], s[1], s[2], s[3]))
}

/// Repeat KV heads for GQA: `[b, n_kv, seq, d]` → `[b, n_kv * n_rep, seq, d]`.
pub fn repeat_kv<B: BackendOps>(x: &B::Tensor, n_rep: usize) -> EngineResult<B::Tensor> {
    if n_rep == 0 {
        return Err(EngineError::model("repeat_kv n_rep must be > 0"));
    }
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, seq, head_dim) = dims4::<B>(x)?;
    x.unsqueeze(2)?
        .broadcast_as(&[b, n_kv, n_rep, seq, head_dim])?
        .reshape(&[b, n_kv * n_rep, seq, head_dim])
}

/// Causal attention mask: `1` marks positions that must be masked (future tokens).
///
/// Shape `[seq_len, index_pos + seq_len]`. Entry `(i, j)` is 1 iff `j > i + index_pos`.
pub fn causal_mask<B: BackendOps>(
    seq_len: usize,
    index_pos: usize,
    device: &B::Device,
) -> EngineResult<B::Tensor> {
    let kv_len = index_pos + seq_len;
    let mut data = vec![0u8; seq_len * kv_len];
    for i in 0..seq_len {
        for j in 0..kv_len {
            if j > i + index_pos {
                data[i * kv_len + j] = 1;
            }
        }
    }
    B::Tensor::from_u8(&data, &[seq_len, kv_len], device)
}

/// Replace masked positions in `values` with `fill` (typically `-inf` for attention).
pub fn masked_fill<B: BackendOps>(
    values: &B::Tensor,
    mask: &B::Tensor,
    fill: f32,
) -> EngineResult<B::Tensor> {
    let fill_t = B::Tensor::from_f32(&[fill], &[], values.device())?.broadcast_as(values.shape())?;
    let mask = mask.broadcast_as(values.shape())?;
    mask.where_cond(&fill_t, values)
}
