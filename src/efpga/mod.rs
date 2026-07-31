//! eFPGA Bio-Compilation & Hardware Logic Acceleration Module for HKL-1.
//! Identifies stable SNN sub-networks, generates synthesizable Verilog HDL,
//! compiles binary eFPGA bitstreams (LUT4/LUT6), and evaluates hardware logic in sub-nanosecond time.

pub mod bitstream;
pub mod hdl_gen;
pub mod simulator;
pub mod stability;

use core::mem::MaybeUninit;
pub use bitstream::{BitstreamConfig, BitstreamEncoder};
pub use hdl_gen::{HdlGenerator, MAX_VERILOG_BUFFER_LEN};
pub use simulator::{EfpgaHardwareSimulator, HardwareBenchmark};
pub use stability::{FrozenSubnetwork, SubnetworkStabilityAnalyzer};
use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;

/// Unified eFPGA Bio-Compilation Engine
pub struct EfpgaEngine {
    pub stability_analyzer: SubnetworkStabilityAnalyzer,
    pub last_frozen_subnetwork: Option<FrozenSubnetwork>,
    pub last_bitstream: Option<BitstreamConfig>,
    pub last_benchmark: Option<HardwareBenchmark>,
    pub verilog_buffer: [u8; MAX_VERILOG_BUFFER_LEN],
    pub verilog_len: usize,
}

impl EfpgaEngine {
    pub fn new() -> Self {
        Self {
            stability_analyzer: SubnetworkStabilityAnalyzer::new(),
            last_frozen_subnetwork: None,
            last_bitstream: None,
            last_benchmark: None,
            verilog_buffer: [0u8; MAX_VERILOG_BUFFER_LEN],
            verilog_len: 0,
        }
    }

    /// Run full eFPGA bio-compilation pipeline on candidate SNN synapses
    pub fn compile_and_accelerate_subnetwork(
        &mut self,
        synapse_data: &[(NeuronId, NeuronId, FixedPoint, u32, FixedPoint, u32)],
        subnetwork_id: u32,
    ) -> (bool, HardwareBenchmark) {
        // 1. Analyze stability & freeze sub-network
        let subnetwork = self.stability_analyzer.analyze_and_freeze_subnetwork(synapse_data, subnetwork_id);

        if subnetwork.count == 0 {
            let empty_bm = HardwareBenchmark {
                subnetwork_id,
                latency_picoseconds: 0,
                clock_cycles: 0,
                speedup_vs_software: 1.0,
                active_luts: 0,
            };
            return (false, empty_bm);
        }

        // 2. Generate synthesizable Verilog HDL
        self.verilog_len = HdlGenerator::generate_verilog_hdl(&subnetwork, &mut self.verilog_buffer);

        // 3. Encode LUT4/LUT6 eFPGA Bitstream
        let bitstream = BitstreamEncoder::encode_bitstream(&subnetwork);

        // 4. Simulate Hardware Execution & Nanosecond Benchmark
        let (_spikes, benchmark) = EfpgaHardwareSimulator::simulate_hardware_execution(&bitstream);

        self.last_frozen_subnetwork = Some(subnetwork);
        self.last_bitstream = Some(bitstream);
        self.last_benchmark = Some(benchmark);

        (true, benchmark)
    }
}

// ---------------------------------------------------------------------------
// Global Instance
// ---------------------------------------------------------------------------
pub static mut EFPGA_ENGINE: MaybeUninit<EfpgaEngine> = MaybeUninit::uninit();

static INITIALIZED_EFPGA_ENGINE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_efpga_engine() {
    unsafe {
        if !INITIALIZED_EFPGA_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            EFPGA_ENGINE.write(EfpgaEngine::new());
            INITIALIZED_EFPGA_ENGINE.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn efpga_engine() -> &'static mut EfpgaEngine {
    unsafe {
        if !INITIALIZED_EFPGA_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            init_efpga_engine();
        }
        &mut *EFPGA_ENGINE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;
    use crate::core::memory::NeuronId;

    #[test]
    fn efpga_engine_new_state() {
        let engine = EfpgaEngine::new();
        assert_eq!(engine.verilog_len, 0);
        assert!(engine.last_frozen_subnetwork.is_none());
        assert!(engine.last_bitstream.is_none());
    }

    #[test]
    fn efpga_engine_rejects_empty_synapses() {
        let mut engine = EfpgaEngine::new();
        let (success, bm) = engine.compile_and_accelerate_subnetwork(&[], 1);
        assert!(!success);
        assert_eq!(bm.speedup_vs_software, 1.0);
    }

    #[test]
    fn efpga_stability_analyzer_new() {
        let sa = SubnetworkStabilityAnalyzer::new();
        assert!(sa.variance_threshold > FixedPoint::ZERO);
        assert_eq!(sa.min_age_cycles, 100);
    }

    #[test]
    fn efpga_stability_analyzer_freeze_empty() {
        let sa = SubnetworkStabilityAnalyzer::new();
        let sub = sa.analyze_and_freeze_subnetwork(&[], 0);
        assert_eq!(sub.count, 0);
        assert_eq!(sub.id, 0);
    }

    #[test]
    fn efpga_stability_analyzer_freeze_stable() {
        let sa = SubnetworkStabilityAnalyzer::new();
        let data = [
            (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200),
        ];
        let sub = sa.analyze_and_freeze_subnetwork(&data, 1);
        assert_eq!(sub.count, 1);
        assert!(sub.avg_variance > FixedPoint::ZERO);
    }

    #[test]
    fn efpga_global_singleton() {
        init_efpga_engine();
        let engine = efpga_engine();
        assert_eq!(engine.verilog_len, 0);
    }
}
