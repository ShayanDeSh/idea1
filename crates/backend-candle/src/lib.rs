//! Candle backend for idea1.
//!
//! Phase 0/1: crate scaffold and device helper only. Tensor/module/KV
//! implementations arrive in Phase 2.

use candle_core::Device;
use idea1_core::{DeviceKind, EngineError, EngineResult};

/// Select a Candle [`Device`] from an idea1 [`DeviceKind`].
pub fn candle_device(kind: DeviceKind) -> EngineResult<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Cuda { device_id } => Device::new_cuda(device_id)
            .map_err(|err| EngineError::backend(format!("CUDA device {device_id}: {err}"))),
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
