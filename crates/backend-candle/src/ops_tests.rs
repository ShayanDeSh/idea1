//! Numeric and weight-loading smoke tests for the Candle backend.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use idea1_core::{
    BackendOps, DeviceKind, ElementDType, EmbeddingModule, LinearModule, RmsNormModule,
    TensorOps, WeightSource,
};

use crate::{CandleBackend, CandleLinear, CandleRmsNorm, CandleTensor, CandleWeights};

fn cpu() -> Device {
    CandleBackend::new_device(DeviceKind::Cpu).unwrap()
}

fn assert_allclose(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "mismatch at {i}: got {g}, expected {e}, tol {tol}"
        );
    }
}

#[test]
fn matmul_matches_hand_computed() {
    let device = cpu();
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = CandleTensor::from_f32(&[1., 2., 3., 4.], &[2, 2], &device).unwrap();
    let b = CandleTensor::from_f32(&[5., 6., 7., 8.], &[2, 2], &device).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_allclose(&c.to_vec_f32().unwrap(), &[19., 22., 43., 50.], 1e-5);
}

#[test]
fn linear_is_x_times_w_t() {
    let device = cpu();
    // W [out=3, in=2], x [1, 2] → y [1, 3]
    let w = CandleTensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[3, 2], &device).unwrap();
    let x = CandleTensor::from_f32(&[10., 100.], &[1, 2], &device).unwrap();
    let linear = CandleLinear::from_weight(w).unwrap();
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.shape(), &[1, 3]);
    assert_allclose(&y.to_vec_f32().unwrap(), &[210., 430., 650.], 1e-5);
}

#[test]
fn rms_norm_matches_hand_computed() {
    let device = cpu();
    let x = CandleTensor::from_f32(&[1., 2., 3., 4.], &[1, 4], &device).unwrap();
    let weight = CandleTensor::from_f32(&[1., 1., 1., 1.], &[4], &device).unwrap();
    let eps = 1e-5_f64;
    let rms = CandleRmsNorm::from_weight(weight, eps).unwrap();
    let out = rms.forward(&x).unwrap();

    let mean_sq = (1.0f32 + 4. + 9. + 16.) / 4.0;
    let scale = 1.0 / (mean_sq + eps as f32).sqrt();
    let expected = [1. * scale, 2. * scale, 3. * scale, 4. * scale];
    assert_allclose(&out.to_vec_f32().unwrap(), &expected, 1e-5);
}

#[test]
fn silu_and_softmax_smoke() {
    let device = cpu();
    let x = CandleTensor::from_f32(&[0., 1., -1.], &[3], &device).unwrap();
    let silu = x.silu().unwrap().to_vec_f32().unwrap();
    // silu(x) = x * sigmoid(x)
    let expected_silu = [
        0.0,
        1.0 / (1.0 + (-1.0f32).exp()),
        -1.0 / (1.0 + 1.0f32.exp()),
    ];
    assert_allclose(&silu, &expected_silu, 1e-5);

    let logits = CandleTensor::from_f32(&[1., 2., 3.], &[1, 3], &device).unwrap();
    let probs = logits.softmax_dim(1).unwrap().to_vec_f32().unwrap();
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(probs[2] > probs[1] && probs[1] > probs[0]);
}

#[test]
fn loads_embedding_and_linear_from_hf_named_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let device = Device::Cpu;

    // Tiny shapes with TinyLlama / HF tensor names.
    let embed = Tensor::from_slice(
        &[1f32, 2., 3., 4., 5., 6.],
        (3, 2), // vocab=3, hidden=2
        &device,
    )
    .unwrap();
    let lm_head = Tensor::from_slice(
        &[0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5],
        (3, 2), // vocab=3, hidden=2
        &device,
    )
    .unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("model.embed_tokens.weight".to_string(), embed);
    tensors.insert("lm_head.weight".to_string(), lm_head);
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let weights =
        CandleWeights::from_paths(&[path], &device, ElementDType::F32).unwrap();

    assert!(weights.contains("model.embed_tokens.weight"));
    assert!(weights.contains("lm_head.weight"));
    assert!(!weights.contains("missing.weight"));

    let names = weights.tensor_names().unwrap();
    assert_eq!(
        names,
        vec![
            "lm_head.weight".to_string(),
            "model.embed_tokens.weight".to_string()
        ]
    );

    let emb = weights
        .load_embedding("model.embed_tokens.weight")
        .unwrap();
    assert_eq!(emb.weight().shape(), &[3, 2]);

    let ids = CandleTensor::from_u32(&[2, 0], &[2], &device).unwrap();
    let looked_up = emb.forward(&ids).unwrap();
    assert_eq!(looked_up.shape(), &[2, 2]);
    assert_allclose(&looked_up.to_vec_f32().unwrap(), &[5., 6., 1., 2.], 1e-5);

    let linear = weights.load_linear("lm_head.weight").unwrap();
    assert_eq!(linear.weight().shape(), &[3, 2]);
    let x = CandleTensor::from_f32(&[1., 0.], &[1, 2], &device).unwrap();
    let logits = linear.forward(&x).unwrap();
    // rows of W dotted with [1,0] → first column of W
    assert_allclose(&logits.to_vec_f32().unwrap(), &[0.5, 2.5, 4.5], 1e-5);
}

/// Optional: load real TinyLlama weights when `IDEA1_MODEL_DIR` is set.
#[test]
fn load_tinyllama_embed_and_lm_head_if_present() {
    let Some(model_dir) = std::env::var_os("IDEA1_MODEL_DIR") else {
        eprintln!("skipping: set IDEA1_MODEL_DIR to run TinyLlama weight load");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    if !model_dir.join("model.safetensors").is_file()
        && !model_dir.join("model.safetensors.index.json").is_file()
    {
        eprintln!(
            "skipping: no safetensors under {}",
            model_dir.display()
        );
        return;
    }

    let device = cpu();
    let weights =
        CandleWeights::from_model_dir(&model_dir, &device, CandleBackend::dtype_for(DeviceKind::Cpu))
            .unwrap();

    let emb = weights
        .load_embedding("model.embed_tokens.weight")
        .unwrap();
    assert_eq!(emb.weight().shape(), &[32000, 2048]);

    let linear = weights.load_linear("lm_head.weight").unwrap();
    assert_eq!(linear.weight().shape(), &[32000, 2048]);

    let ids = CandleTensor::from_u32(&[1], &[1], &device).unwrap();
    let hidden = emb.forward(&ids).unwrap();
    assert_eq!(hidden.shape(), &[1, 2048]);
    let logits = linear.forward(&hidden).unwrap();
    assert_eq!(logits.shape(), &[1, 32000]);
}
