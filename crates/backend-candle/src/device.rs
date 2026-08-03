use candle_core::{DType, Device};
use idea1_core::{DeviceKind, ElementDType, EngineError, EngineResult};

/// Select a Candle [`Device`] from an idea1 [`DeviceKind`].
pub fn candle_device(kind: DeviceKind) -> EngineResult<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Cuda { device_id } => Device::new_cuda(device_id)
            .map_err(|err| EngineError::backend(format!("CUDA device {device_id}: {err}"))),
    }
}

/// Map idea1 [`ElementDType`] to Candle [`DType`].
pub fn candle_dtype(dtype: ElementDType) -> DType {
    match dtype {
        ElementDType::F32 => DType::F32,
        ElementDType::F16 => DType::F16,
        ElementDType::BF16 => DType::BF16,
    }
}

/// Map Candle [`DType`] to idea1 [`ElementDType`].
pub fn from_candle_dtype(dtype: DType) -> EngineResult<ElementDType> {
    match dtype {
        DType::F32 => Ok(ElementDType::F32),
        DType::F16 => Ok(ElementDType::F16),
        DType::BF16 => Ok(ElementDType::BF16),
        other => Err(EngineError::backend(format!(
            "unsupported candle dtype {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_available() {
        let device = candle_device(DeviceKind::Cpu).unwrap();
        assert!(device.is_cpu());
    }
}
