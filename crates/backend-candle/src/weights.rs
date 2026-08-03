use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use idea1_core::{
    discover_weight_files, ElementDType, EngineError, EngineResult, WeightFiles, WeightSource,
};

use crate::device::candle_dtype;
use crate::tensor::CandleTensor;
use crate::CandleBackend;

fn map_err(err: candle_core::Error) -> EngineError {
    EngineError::backend(err.to_string())
}

/// Memory-mapped safetensors weight store for Candle.
///
/// Tensors are loaded on demand via [`WeightSource::get_tensor`] and converted
/// to the configured compute dtype (f32 on CPU).
pub struct CandleWeights {
    mmap: MmapedSafetensors,
    device: Device,
    dtype: DType,
}

impl CandleWeights {
    /// mmap all shards discovered under `model_dir`.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        device: &Device,
        dtype: ElementDType,
    ) -> EngineResult<Self> {
        let files = discover_weight_files(model_dir)?;
        Self::from_weight_files(&files, device, dtype)
    }

    /// mmap the given weight files.
    pub fn from_weight_files(
        files: &WeightFiles,
        device: &Device,
        dtype: ElementDType,
    ) -> EngineResult<Self> {
        Self::from_paths(&files.paths, device, dtype)
    }

    /// mmap `paths` without materializing tensors yet.
    pub fn from_paths(
        paths: &[impl AsRef<Path>],
        device: &Device,
        dtype: ElementDType,
    ) -> EngineResult<Self> {
        if paths.is_empty() {
            return Err(EngineError::weights("no safetensors paths provided"));
        }
        // Safety: inherited from Candle / memmap2 — files must not be mutated
        // while mapped. Model dirs are treated as immutable inputs.
        let mmap = unsafe { MmapedSafetensors::multi(paths) }.map_err(map_err)?;
        Ok(Self {
            mmap,
            device: device.clone(),
            dtype: candle_dtype(dtype),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl WeightSource<CandleBackend> for CandleWeights {
    fn tensor_names(&self) -> EngineResult<Vec<String>> {
        let mut names: Vec<String> = self
            .mmap
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        names.sort();
        Ok(names)
    }

    fn contains(&self, name: &str) -> bool {
        self.mmap.get(name).is_ok()
    }

    fn get_tensor(&self, name: &str) -> EngineResult<CandleTensor> {
        let t = self
            .mmap
            .load(name, &self.device)
            .map_err(|err| EngineError::weights(format!("load `{name}`: {err}")))?
            .to_dtype(self.dtype)
            .map_err(map_err)?;
        Ok(CandleTensor::from_candle(t))
    }
}
