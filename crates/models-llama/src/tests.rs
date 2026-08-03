//! Shape and numeric smoke tests for Llama building blocks (Candle CPU).

use idea1_backend_candle::{CandleBackend, CandleLinear, CandleRmsNorm, CandleTensor};
use idea1_core::{
    BackendOps, DeviceKind, ElementDType, LinearModule, LlamaConfig, RmsNormModule, TensorOps,
};

use crate::{Attention, DecoderLayer, Mlp, RotaryEmbedding, causal_mask, repeat_kv};

fn cpu() -> candle_core::Device {
    CandleBackend::new_device(DeviceKind::Cpu).unwrap()
}

/// Tiny GQA config: hidden=32, 4 Q heads / 2 KV heads, head_dim=8, intermediate=64.
fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 32,
        intermediate_size: 64,
        vocab_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: Some(1),
        eos_token_ids: vec![2],
        max_seq_len: 32,
        tie_word_embeddings: false,
    }
}

fn linear(out: usize, inn: usize, device: &candle_core::Device, seed: f32) -> CandleLinear {
    let n = out * inn;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect();
    let w = CandleTensor::from_f32(&data, &[out, inn], device).unwrap();
    CandleLinear::from_weight(w).unwrap()
}

fn rms(dim: usize, device: &candle_core::Device) -> CandleRmsNorm {
    let w = CandleTensor::from_f32(&vec![1f32; dim], &[dim], device).unwrap();
    CandleRmsNorm::from_weight(w, 1e-5).unwrap()
}

#[test]
fn rope_tables_have_expected_shape() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    assert_eq!(rope.max_seq_len(), 32);
    assert_eq!(rope.head_dim(), 8);

    let (cos, sin) = rope.at(0, 5).unwrap();
    assert_eq!(cos.shape(), &[5, 4]); // head_dim/2
    assert_eq!(sin.shape(), &[5, 4]);

    let (cos2, _) = rope.at(3, 4).unwrap();
    assert_eq!(cos2.shape(), &[4, 4]);
}

#[test]
fn rope_position_zero_is_identity_rotation() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();

    // At pos 0, angles are 0 → cos=1, sin=0 → RoPE is identity.
    let (cos, sin) = rope.at(0, 1).unwrap();
    let cos_v = cos.to_vec_f32().unwrap();
    let sin_v = sin.to_vec_f32().unwrap();
    for c in &cos_v {
        assert!((c - 1.0).abs() < 1e-5, "cos={c}");
    }
    for s in &sin_v {
        assert!(s.abs() < 1e-5, "sin={s}");
    }

    let x_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    // [1, 1, 1, 8] then expand conceptually — use [1, 2, 1, 8]
    let x = CandleTensor::from_f32(&x_data[..16], &[1, 2, 1, 8], &device)
        .unwrap()
        .contiguous()
        .unwrap();
    let y = rope.apply(&x, 0).unwrap();
    let xv = x.to_vec_f32().unwrap();
    let yv = y.to_vec_f32().unwrap();
    for (a, b) in xv.iter().zip(yv.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn causal_mask_upper_triangle() {
    let device = cpu();
    let mask = causal_mask::<CandleBackend>(3, 0, &device).unwrap();
    assert_eq!(mask.shape(), &[3, 3]);
    // from_u8 → need to check via where / cast; convert by using as attention logic
    // Build expected: row i masks j > i
    // We'll verify via masked_fill on zeros.
    let zeros = CandleTensor::zeros(&[1, 1, 3, 3], ElementDType::F32, &device).unwrap();
    let filled = crate::masked_fill::<CandleBackend>(&zeros, &mask, -1.0).unwrap();
    let v = filled.to_vec_f32().unwrap();
    // row0: keep 0, mask 1,2
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], -1.0);
    assert_eq!(v[2], -1.0);
    // row1: keep 0,1 mask 2
    assert_eq!(v[3], 0.0);
    assert_eq!(v[4], 0.0);
    assert_eq!(v[5], -1.0);
    // row2: keep all
    assert_eq!(v[6], 0.0);
    assert_eq!(v[7], 0.0);
    assert_eq!(v[8], 0.0);
}

#[test]
fn repeat_kv_expands_heads() {
    let device = cpu();
    // [1, 2, 3, 4] → n_rep=2 → [1, 4, 3, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = CandleTensor::from_f32(&data, &[1, 2, 3, 4], &device).unwrap();
    let y = repeat_kv::<CandleBackend>(&x, 2).unwrap();
    assert_eq!(y.shape(), &[1, 4, 3, 4]);
    let yv = y.to_vec_f32().unwrap();
    // head0 repeated as heads 0 and 1
    assert_eq!(&yv[0..12], &data[0..12]);
    assert_eq!(&yv[12..24], &data[0..12]);
    // head1 repeated as heads 2 and 3
    assert_eq!(&yv[24..36], &data[12..24]);
    assert_eq!(&yv[36..48], &data[12..24]);
}

#[test]
fn mlp_swiglu_shape() {
    let device = cpu();
    let cfg = tiny_cfg();
    let mlp = Mlp::<CandleBackend>::new(
        linear(cfg.intermediate_size, cfg.hidden_size, &device, 1.0),
        linear(cfg.intermediate_size, cfg.hidden_size, &device, 2.0),
        linear(cfg.hidden_size, cfg.intermediate_size, &device, 3.0),
    );
    let x = CandleTensor::from_f32(&vec![0.1f32; 2 * 3 * 32], &[2, 3, 32], &device).unwrap();
    let y = mlp.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 3, 32]);
    let vals = y.to_vec_f32().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn attention_gqa_shape() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let attn = Attention::<CandleBackend>::new(
        linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, &device, 0.0),
        linear(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, &device, 1.0),
        linear(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, &device, 2.0),
        linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, &device, 3.0),
        &cfg,
    );
    assert_eq!(attn.num_heads(), 4);
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.head_dim(), 8);

    let seq = 5;
    let x = CandleTensor::from_f32(&vec![0.05f32; seq * cfg.hidden_size], &[1, seq, 32], &device)
        .unwrap();
    let y = attn.forward(&x, &rope, 0).unwrap();
    assert_eq!(y.shape(), &[1, seq, 32]);
    assert!(y.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn decoder_layer_shape_and_residual_smoke() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let layer = DecoderLayer::<CandleBackend>::new(
        rms(cfg.hidden_size, &device),
        Attention::new(
            linear(32, 32, &device, 0.0),
            linear(16, 32, &device, 1.0), // 2 kv heads * 8
            linear(16, 32, &device, 2.0),
            linear(32, 32, &device, 3.0),
            &cfg,
        ),
        rms(cfg.hidden_size, &device),
        Mlp::new(
            linear(64, 32, &device, 4.0),
            linear(64, 32, &device, 5.0),
            linear(32, 64, &device, 6.0),
        ),
    );

    let seq = 4;
    let x = CandleTensor::from_f32(&vec![0.02f32; seq * 32], &[1, seq, 32], &device).unwrap();
    let y = layer.forward(&x, &rope, 0).unwrap();
    assert_eq!(y.shape(), &[1, seq, 32]);
    let yv = y.to_vec_f32().unwrap();
    assert!(yv.iter().all(|v| v.is_finite()));
    // Residual path: output should not be all zeros with non-zero input.
    assert!(yv.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn single_token_attention_skips_mask() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let attn = Attention::<CandleBackend>::new(
        linear(32, 32, &device, 0.0),
        linear(16, 32, &device, 1.0),
        linear(16, 32, &device, 2.0),
        linear(32, 32, &device, 3.0),
        &cfg,
    );
    let x = CandleTensor::from_f32(&vec![0.1f32; 32], &[1, 1, 32], &device).unwrap();
    let y = attn.forward(&x, &rope, 0).unwrap();
    assert_eq!(y.shape(), &[1, 1, 32]);
    assert!(y.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}
