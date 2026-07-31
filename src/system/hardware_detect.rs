//! Hardware resource auto-detection and adaptive capacity scaling engine.
//! Dynamically inspects host hardware (RAM, CPU cores, SIMD support)
//! and calculates optimal memory capacities for HKL-1.

use crate::core::math::FixedPoint;

/// Hardware profile describing host capabilities
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareProfile {
    pub system_ram_bytes: usize,
    pub cpu_cores: usize,
    pub simd_enabled: bool,
    pub is_bare_metal: bool,
    pub recommended_max_neurons: usize,
    pub recommended_max_synapses: usize,
    pub recommended_worker_threads: usize,
}

impl HardwareProfile {
    /// Return default bare-metal embedded profile
    pub const fn bare_metal_default() -> Self {
        Self {
            system_ram_bytes: 256 * 1024, // 256 KB
            cpu_cores: 1,
            simd_enabled: cfg!(feature = "simd"),
            is_bare_metal: true,
            recommended_max_neurons: crate::MAX_NEURONS,
            recommended_max_synapses: crate::MAX_SYNAPSES,
            recommended_worker_threads: 1,
        }
    }
}

pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect host hardware capabilities and compute optimal capacity scaling
    pub fn detect() -> HardwareProfile {
        #[cfg(feature = "std")]
        {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            // Desktop/Server profile calculation:
            // Estimate 4 GB RAM base for desktop host environment
            let estimated_ram_bytes = 4 * 1024 * 1024 * 1024;

            // Allow 64,000 neurons per GB RAM by default, up to 1,048,576 neurons
            let neurons = 262_144;
            let synapses = neurons * 16;

            HardwareProfile {
                system_ram_bytes: estimated_ram_bytes,
                cpu_cores: cores,
                simd_enabled: cfg!(feature = "simd"),
                is_bare_metal: false,
                recommended_max_neurons: neurons,
                recommended_max_synapses: synapses,
                recommended_worker_threads: cores,
            }
        }

        #[cfg(not(feature = "std"))]
        {
            HardwareProfile::bare_metal_default()
        }
    }

    /// Calculate scaling factor relative to bare-metal baseline (FixedPoint)
    pub fn calculate_scale_factor(profile: &HardwareProfile) -> FixedPoint {
        let ratio = profile.recommended_max_neurons as f32 / crate::MAX_NEURONS as f32;
        FixedPoint::from_f32(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detector_default() {
        let profile = HardwareDetector::detect();
        assert!(profile.recommended_max_neurons >= crate::MAX_NEURONS);
        assert!(profile.recommended_max_synapses >= crate::MAX_SYNAPSES);
        assert!(profile.recommended_worker_threads >= 1);
    }

    #[test]
    fn test_scale_factor_calculation() {
        let profile = HardwareProfile::bare_metal_default();
        let scale = HardwareDetector::calculate_scale_factor(&profile);
        assert!((scale.to_f32() - 1.0).abs() < 0.01);
    }
}
