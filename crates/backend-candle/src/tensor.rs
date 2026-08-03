use candle_core::{Device, Tensor};
use candle_nn::ops;
use idea1_core::{ElementDType, EngineError, EngineResult, TensorOps};

use crate::device::{candle_dtype, from_candle_dtype};
use crate::CandleBackend;

/// Newtype around Candle [`Tensor`].
#[derive(Debug, Clone)]
pub struct CandleTensor(pub Tensor);

impl CandleTensor {
    pub fn inner(&self) -> &Tensor {
        &self.0
    }

    pub fn into_inner(self) -> Tensor {
        self.0
    }

    pub fn from_candle(t: Tensor) -> Self {
        Self(t)
    }
}

fn map_err(err: candle_core::Error) -> EngineError {
    EngineError::backend(err.to_string())
}

impl TensorOps<CandleBackend> for CandleTensor {
    fn shape(&self) -> &[usize] {
        self.0.dims()
    }

    fn dtype(&self) -> ElementDType {
        from_candle_dtype(self.0.dtype()).expect("tensor created with supported dtype")
    }

    fn device(&self) -> &Device {
        self.0.device()
    }

    fn from_f32(data: &[f32], shape: &[usize], device: &Device) -> EngineResult<Self> {
        Tensor::from_slice(data, shape, device)
            .map(Self)
            .map_err(map_err)
    }

    fn from_u32(data: &[u32], shape: &[usize], device: &Device) -> EngineResult<Self> {
        Tensor::from_slice(data, shape, device)
            .map(Self)
            .map_err(map_err)
    }

    fn zeros(shape: &[usize], dtype: ElementDType, device: &Device) -> EngineResult<Self> {
        Tensor::zeros(shape, candle_dtype(dtype), device)
            .map(Self)
            .map_err(map_err)
    }

    fn reshape(&self, shape: &[usize]) -> EngineResult<Self> {
        self.0.reshape(shape).map(Self).map_err(map_err)
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> EngineResult<Self> {
        self.0.transpose(dim1, dim2).map(Self).map_err(map_err)
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> EngineResult<Self> {
        self.0.narrow(dim, start, len).map(Self).map_err(map_err)
    }

    fn contiguous(&self) -> EngineResult<Self> {
        self.0.contiguous().map(Self).map_err(map_err)
    }

    fn cat(tensors: &[&Self], dim: usize) -> EngineResult<Self> {
        let refs: Vec<&Tensor> = tensors.iter().map(|t| &t.0).collect();
        Tensor::cat(&refs, dim).map(Self).map_err(map_err)
    }

    fn matmul(&self, rhs: &Self) -> EngineResult<Self> {
        self.0.matmul(&rhs.0).map(Self).map_err(map_err)
    }

    fn softmax_dim(&self, dim: usize) -> EngineResult<Self> {
        ops::softmax(&self.0, dim).map(Self).map_err(map_err)
    }

    fn silu(&self) -> EngineResult<Self> {
        ops::silu(&self.0).map(Self).map_err(map_err)
    }

    fn apply_rope(&self, cos: &Self, sin: &Self) -> EngineResult<Self> {
        // Llama / TinyLlama use half-split RoPE (not interleaved).
        candle_nn::rotary_emb::rope(&self.0, &cos.0, &sin.0)
            .map(Self)
            .map_err(map_err)
    }

    fn add(&self, rhs: &Self) -> EngineResult<Self> {
        (&self.0 + &rhs.0).map(Self).map_err(map_err)
    }

    fn mul(&self, rhs: &Self) -> EngineResult<Self> {
        (&self.0 * &rhs.0).map(Self).map_err(map_err)
    }

    fn broadcast_add(&self, rhs: &Self) -> EngineResult<Self> {
        self.0.broadcast_add(&rhs.0).map(Self).map_err(map_err)
    }

    fn broadcast_mul(&self, rhs: &Self) -> EngineResult<Self> {
        self.0.broadcast_mul(&rhs.0).map(Self).map_err(map_err)
    }

    fn mul_scalar(&self, v: f64) -> EngineResult<Self> {
        (&self.0 * v).map(Self).map_err(map_err)
    }

    fn to_dtype(&self, dtype: ElementDType) -> EngineResult<Self> {
        self.0
            .to_dtype(candle_dtype(dtype))
            .map(Self)
            .map_err(map_err)
    }

    fn to_vec_f32(&self) -> EngineResult<Vec<f32>> {
        let t = if self.0.dtype() == candle_core::DType::F32 {
            self.0.clone()
        } else {
            self.0.to_dtype(candle_core::DType::F32).map_err(map_err)?
        };
        let flat = t.flatten_all().map_err(map_err)?;
        flat.to_vec1::<f32>().map_err(map_err)
    }
}
