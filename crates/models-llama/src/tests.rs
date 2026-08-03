//! Shape, KV-cache, and prefill/decode parity tests (Candle CPU).

use idea1_backend_candle::{CandleBackend, CandleEmbedding, CandleLinear, CandleRmsNorm, CandleTensor};
use idea1_core::{
    BackendOps, CausalLm, DeviceKind, ElementDType, EmbeddingModule, KvCache, LinearModule,
    LlamaConfig, RmsNormModule, TensorOps,
};

use crate::{Attention, DecoderLayer, LlamaModel, Mlp, RotaryEmbedding, causal_mask, repeat_kv};

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

fn embedding(vocab: usize, hidden: usize, device: &candle_core::Device) -> CandleEmbedding {
    let n = vocab * hidden;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017).sin() * 0.05)
        .collect();
    let w = CandleTensor::from_f32(&data, &[vocab, hidden], device).unwrap();
    CandleEmbedding::from_weight(w).unwrap()
}

fn tiny_attention(device: &candle_core::Device, cfg: &LlamaConfig) -> Attention<CandleBackend> {
    Attention::new(
        linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, device, 0.0),
        linear(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, device, 1.0),
        linear(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size, device, 2.0),
        linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, device, 3.0),
        cfg,
    )
}

fn tiny_layer(device: &candle_core::Device, cfg: &LlamaConfig) -> DecoderLayer<CandleBackend> {
    DecoderLayer::new(
        rms(cfg.hidden_size, device),
        tiny_attention(device, cfg),
        rms(cfg.hidden_size, device),
        Mlp::new(
            linear(cfg.intermediate_size, cfg.hidden_size, device, 4.0),
            linear(cfg.intermediate_size, cfg.hidden_size, device, 5.0),
            linear(cfg.hidden_size, cfg.intermediate_size, device, 6.0),
        ),
    )
}

fn tiny_model(device: &candle_core::Device) -> LlamaModel<CandleBackend> {
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, device, ElementDType::F32).unwrap();
    LlamaModel::new(
        cfg.clone(),
        embedding(cfg.vocab_size, cfg.hidden_size, device),
        vec![tiny_layer(device, &cfg)],
        rms(cfg.hidden_size, device),
        linear(cfg.vocab_size, cfg.hidden_size, device, 7.0),
        rope,
        device.clone(),
    )
    .unwrap()
}

fn assert_all_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= tol,
            "mismatch at {i}: {x} vs {y} (diff={diff}, tol={tol})"
        );
    }
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
    let zeros = CandleTensor::zeros(&[1, 1, 3, 3], ElementDType::F32, &device).unwrap();
    let filled = crate::masked_fill::<CandleBackend>(&zeros, &mask, -1.0).unwrap();
    let v = filled.to_vec_f32().unwrap();
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], -1.0);
    assert_eq!(v[2], -1.0);
    assert_eq!(v[3], 0.0);
    assert_eq!(v[4], 0.0);
    assert_eq!(v[5], -1.0);
    assert_eq!(v[6], 0.0);
    assert_eq!(v[7], 0.0);
    assert_eq!(v[8], 0.0);
}

#[test]
fn repeat_kv_expands_heads() {
    let device = cpu();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = CandleTensor::from_f32(&data, &[1, 2, 3, 4], &device).unwrap();
    let y = repeat_kv::<CandleBackend>(&x, 2).unwrap();
    assert_eq!(y.shape(), &[1, 4, 3, 4]);
    let yv = y.to_vec_f32().unwrap();
    assert_eq!(&yv[0..12], &data[0..12]);
    assert_eq!(&yv[12..24], &data[0..12]);
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
    let attn = tiny_attention(&device, &cfg);
    assert_eq!(attn.num_heads(), 4);
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.head_dim(), 8);

    let seq = 5;
    let x = CandleTensor::from_f32(&vec![0.05f32; seq * cfg.hidden_size], &[1, seq, 32], &device)
        .unwrap();
    let y = attn.forward(&x, &rope, 0, None).unwrap();
    assert_eq!(y.shape(), &[1, seq, 32]);
    assert!(y.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn decoder_layer_shape_and_residual_smoke() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let layer = tiny_layer(&device, &cfg);

    let seq = 4;
    let x = CandleTensor::from_f32(&vec![0.02f32; seq * 32], &[1, seq, 32], &device).unwrap();
    let y = layer.forward(&x, &rope, 0, None).unwrap();
    assert_eq!(y.shape(), &[1, seq, 32]);
    let yv = y.to_vec_f32().unwrap();
    assert!(yv.iter().all(|v| v.is_finite()));
    assert!(yv.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn single_token_attention_skips_mask() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let attn = tiny_attention(&device, &cfg);
    let x = CandleTensor::from_f32(&vec![0.1f32; 32], &[1, 1, 32], &device).unwrap();
    let y = attn.forward(&x, &rope, 0, None).unwrap();
    assert_eq!(y.shape(), &[1, 1, 32]);
    assert!(y.to_vec_f32().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn kv_cache_append_grows_seq_len() {
    let device = cpu();
    let mut cache = <CandleBackend as BackendOps>::KvCache::new();
    assert_eq!(cache.current_len(), 0);

    let k1 = CandleTensor::from_f32(&vec![1f32; 1 * 2 * 3 * 4], &[1, 2, 3, 4], &device).unwrap();
    let v1 = CandleTensor::from_f32(&vec![2f32; 1 * 2 * 3 * 4], &[1, 2, 3, 4], &device).unwrap();
    let (k, v) = cache.append(&k1, &v1).unwrap();
    assert_eq!(cache.current_len(), 3);
    assert_eq!(k.shape(), &[1, 2, 3, 4]);
    assert_eq!(v.shape(), &[1, 2, 3, 4]);

    let k2 = CandleTensor::from_f32(&vec![3f32; 1 * 2 * 1 * 4], &[1, 2, 1, 4], &device).unwrap();
    let v2 = CandleTensor::from_f32(&vec![4f32; 1 * 2 * 1 * 4], &[1, 2, 1, 4], &device).unwrap();
    let (k, v) = cache.append(&k2, &v2).unwrap();
    assert_eq!(cache.current_len(), 4);
    assert_eq!(k.shape(), &[1, 2, 4, 4]);
    assert_eq!(v.shape(), &[1, 2, 4, 4]);

    cache.reset();
    assert_eq!(cache.current_len(), 0);
}

#[test]
fn attention_prefill_decode_matches_recompute() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let attn = tiny_attention(&device, &cfg);

    let seq = 5;
    let data: Vec<f32> = (0..seq * cfg.hidden_size)
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();
    let x = CandleTensor::from_f32(&data, &[1, seq, cfg.hidden_size], &device).unwrap();

    let full = attn.forward(&x, &rope, 0, None).unwrap();
    let full_last = full.narrow(1, seq - 1, 1).unwrap().to_vec_f32().unwrap();

    let mut cache = attn.new_kv_cache();
    let prompt = x.narrow(1, 0, seq - 1).unwrap();
    let _ = attn.forward(&prompt, &rope, 0, Some(&mut cache)).unwrap();
    assert_eq!(cache.current_len(), seq - 1);

    let last_tok = x.narrow(1, seq - 1, 1).unwrap();
    let decoded = attn
        .forward(&last_tok, &rope, seq - 1, Some(&mut cache))
        .unwrap();
    assert_eq!(cache.current_len(), seq);
    assert_eq!(decoded.shape(), &[1, 1, cfg.hidden_size]);

    let decoded_v = decoded.to_vec_f32().unwrap();
    assert_all_close(&full_last, &decoded_v, 1e-5);
}

#[test]
fn decoder_layer_prefill_decode_matches_recompute() {
    let device = cpu();
    let cfg = tiny_cfg();
    let rope = RotaryEmbedding::<CandleBackend>::new(&cfg, &device, ElementDType::F32).unwrap();
    let layer = tiny_layer(&device, &cfg);

    let seq = 4;
    let data: Vec<f32> = (0..seq * cfg.hidden_size)
        .map(|i| ((i as f32) * 0.02).cos() * 0.1)
        .collect();
    let x = CandleTensor::from_f32(&data, &[1, seq, cfg.hidden_size], &device).unwrap();

    let full = layer.forward(&x, &rope, 0, None).unwrap();
    let full_last = full.narrow(1, seq - 1, 1).unwrap().to_vec_f32().unwrap();

    let mut cache = layer.new_kv_cache();
    let _ = layer
        .forward(&x.narrow(1, 0, seq - 1).unwrap(), &rope, 0, Some(&mut cache))
        .unwrap();
    let decoded = layer
        .forward(
            &x.narrow(1, seq - 1, 1).unwrap(),
            &rope,
            seq - 1,
            Some(&mut cache),
        )
        .unwrap();

    assert_eq!(cache.current_len(), seq);
    assert_all_close(&full_last, &decoded.to_vec_f32().unwrap(), 1e-5);
}

#[test]
fn causal_lm_prefill_decode_logits_match_recompute() {
    let device = cpu();
    let mut model = tiny_model(&device);
    let tokens: Vec<u32> = vec![3, 7, 11, 19, 23];

    let recomputed = model.forward_recompute(&tokens).unwrap();
    let seq = tokens.len();
    let recompute_last = recomputed
        .narrow(1, seq - 1, 1)
        .unwrap()
        .reshape(&[1, model.config().vocab_size])
        .unwrap()
        .to_vec_f32()
        .unwrap();

    let prefill_logits = model.prefill(&tokens[..seq - 1]).unwrap();
    assert_eq!(model.cache_lens(), vec![seq - 1]);
    assert_eq!(prefill_logits.shape(), &[1, model.config().vocab_size]);

    let decode_logits = model.decode_step(tokens[seq - 1], seq - 1).unwrap();
    assert_eq!(model.cache_lens(), vec![seq]);
    assert_eq!(decode_logits.shape(), &[1, model.config().vocab_size]);

    assert_all_close(
        &recompute_last,
        &decode_logits.to_vec_f32().unwrap(),
        1e-4,
    );

    // Prefill of the full prompt should match the last-position recompute too.
    let full_prefill = model.prefill(&tokens).unwrap();
    assert_eq!(model.cache_lens(), vec![seq]);
    assert_all_close(
        &recompute_last,
        &full_prefill.to_vec_f32().unwrap(),
        1e-4,
    );
}

#[test]
fn decode_grows_cache_not_full_recompute_length() {
    let device = cpu();
    let mut model = tiny_model(&device);
    let prompt = [5u32, 6, 7, 8];
    let _ = model.prefill(&prompt).unwrap();
    assert_eq!(model.cache_lens(), vec![4]);

    for (i, &tok) in [9u32, 10, 11].iter().enumerate() {
        let pos = prompt.len() + i;
        let _ = model.decode_step(tok, pos).unwrap();
        // Cache length grows by one each decode — O(seq) attention context, not
        // a fresh full-prompt recompute buffer.
        assert_eq!(model.cache_lens(), vec![pos + 1]);
    }
}
