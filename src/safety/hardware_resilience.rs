use crate::core::math::FixedPoint;
use crate::core::memory::{NEURON_COUNT, NeuronId, SynapseId, neuron_state};
use core::sync::atomic::Ordering;

const REDUNDANCY_GROUPS: usize = 32;
const ECC_BLOCK_SIZE: usize = 8;
const SENESCENCE_MAX_AGE: u32 = 100_000_000;

#[derive(Clone, Copy)]
pub struct EccBlock {
    pub data: [i32; ECC_BLOCK_SIZE],
    pub parity: i32,
    pub syndrome: i32,
    pub errors_detected: u32,
    pub errors_corrected: u32,
}

impl EccBlock {
    pub const fn new() -> Self {
        Self {
            data: [0; ECC_BLOCK_SIZE],
            parity: 0,
            syndrome: 0,
            errors_detected: 0,
            errors_corrected: 0,
        }
    }

    pub fn compute_parity(data: &[i32]) -> i32 {
        data.iter().fold(0i32, |acc, &x| acc ^ x)
    }

    pub fn write_block(&mut self, data: &[i32; ECC_BLOCK_SIZE]) {
        self.data = *data;
        self.parity = Self::compute_parity(data);
        self.syndrome = 0;
    }

    pub fn verify_and_correct(&mut self) -> bool {
        let current_parity = Self::compute_parity(&self.data);
        self.syndrome = self.parity ^ current_parity;
        if self.syndrome == 0 {
            return true;
        }
        self.errors_detected += 1;

        let flipped = self.syndrome.trailing_zeros() as usize;
        if flipped < ECC_BLOCK_SIZE {
            self.data[flipped] ^= 1;
            self.parity = Self::compute_parity(&self.data);
            self.syndrome = self.parity ^ Self::compute_parity(&self.data);
            if self.syndrome == 0 {
                self.errors_corrected += 1;
                return true;
            }
        }
        false
    }
}

pub struct BitFlipDetector {
    pub parity_errors: [u32; 16],
    pub memory_region_checksums: [u32; 128],
    pub last_check: u32,
    pub soft_errors_detected: u32,
    pub ecc_blocks: [EccBlock; REDUNDANCY_GROUPS],
    pub redundant_region: [u16; 1024],
}

impl BitFlipDetector {
    pub const fn new() -> Self {
        Self {
            parity_errors: [0; 16],
            memory_region_checksums: [0; 128],
            last_check: 0,
            soft_errors_detected: 0,
            ecc_blocks: [EccBlock::new(); REDUNDANCY_GROUPS],
            redundant_region: [0; 1024],
        }
    }

    pub fn check_parity(&mut self, region: usize, data: &[u8]) -> bool {
        let expected = self.memory_region_checksums[region];
        let actual = data.iter().fold(0u32, |acc, &x| acc.wrapping_add(x as u32));
        if expected != 0 && expected != actual {
            self.parity_errors[region % 16] += 1;
            self.soft_errors_detected += 1;
            return false;
        }
        self.memory_region_checksums[region] = actual;
        true
    }

    pub fn has_too_many_errors(&self, threshold: u32) -> bool {
        self.soft_errors_detected > threshold
    }

    pub fn verify_all_ecc(&mut self) -> u32 {
        let mut corrected = 0;
        for i in 0..REDUNDANCY_GROUPS {
            if !self.ecc_blocks[i].verify_and_correct() {
                self.soft_errors_detected += 1;
            } else if self.ecc_blocks[i].errors_corrected > 0 {
                corrected += 1;
            }
        }
        corrected
    }

    pub fn repair_neuron_state(&mut self, neuron_id: NeuronId) -> bool {
        let state = neuron_state(neuron_id);
        let data: [i32; ECC_BLOCK_SIZE] = [
            state.membrane_potential.to_bits(),
            state.threshold.to_bits(),
            state.leak.to_bits(),
            state.bias_current.to_bits(),
            state.last_spike_time as i32,
            state.refractory_remaining as i32,
            state.layer as i32,
            state.neuron_type as i32,
        ];
        let block_idx = (neuron_id.index()) % REDUNDANCY_GROUPS;
        self.ecc_blocks[block_idx].write_block(&data);
        true
    }
}

pub struct MemoryDiagnostics {
    pub ping_responses: [u32; 64],
    pub failing_blocks: u32,
    pub bad_sectors: [u16; 16],
    pub bad_sector_count: u8,
    pub migration_map: [SynapseId; 256],
    pub migration_count: u16,
}

impl MemoryDiagnostics {
    pub const fn new() -> Self {
        Self {
            ping_responses: [0; 64],
            failing_blocks: 0,
            bad_sectors: [0; 16],
            bad_sector_count: 0,
            migration_map: [SynapseId::INVALID; 256],
            migration_count: 0,
        }
    }

    pub fn ping_block(&mut self, block: usize) -> u32 {
        let start = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() };
        unsafe {
            core::ptr::read_volatile(block as *const u32);
        }
        let elapsed = (unsafe { crate::core::time::METABOLIC_CLOCK.now_us() } - start) as u32;
        self.ping_responses[block % 64] = elapsed;
        elapsed
    }

    pub fn is_degrading(&self, block: usize, baseline: u32, threshold: u32) -> bool {
        let current = self.ping_responses[block % 64];
        current > baseline + threshold
    }

    pub fn mark_bad_sector(&mut self, sector: u16) {
        if self.bad_sector_count < 16 {
            self.bad_sectors[self.bad_sector_count as usize] = sector;
            self.bad_sector_count += 1;
        }
        self.failing_blocks += 1;
    }

    pub fn migrate_synapse(&mut self, old_id: SynapseId, new_id: SynapseId) {
        if self.migration_count < 256 {
            self.migration_map[self.migration_count as usize] = old_id;
            self.migration_count += 1;
        }
        let src = crate::snn::synapse::synapse_ref(old_id);
        let dst = crate::snn::synapse::synapse(new_id);
        dst.pre = src.pre;
        dst.post = src.post;
        dst.weight = src.weight;
        dst.plasticity_enabled = src.plasticity_enabled;

        let old_s = crate::snn::synapse::synapse(old_id);
        old_s.pre = crate::core::memory::NeuronId::INVALID;
        old_s.weight = crate::core::math::Weight::ZERO;
    }

    pub fn has_bad_sector(&self, sector: u16) -> bool {
        for i in 0..self.bad_sector_count as usize {
            if self.bad_sectors[i] == sector {
                return true;
            }
        }
        false
    }

    pub fn senescence_score(&self, age: u32) -> FixedPoint {
        let ratio = (age as f32) / (SENESCENCE_MAX_AGE as f32);
        FixedPoint::from_f32(ratio.min(1.0))
    }

    pub fn check_senescence(&self, age: u32) -> SenescenceStage {
        let score = self.senescence_score(age).to_f32();
        if score < 0.3 {
            SenescenceStage::Healthy
        } else if score < 0.6 {
            SenescenceStage::Aging
        } else if score < 0.85 {
            SenescenceStage::Degraded
        } else {
            SenescenceStage::EndOfLife
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SenescenceStage {
    Healthy,
    Aging,
    Degraded,
    EndOfLife,
}

impl Default for SenescenceStage {
    fn default() -> Self {
        Self::Healthy
    }
}

pub fn run_diagnostics(now: u32) {
    if now % 1000 != 0 {
        return;
    }
    let count = NEURON_COUNT.load(Ordering::Relaxed);
    for i in 0..count.min(16) as u16 {
        let id = NeuronId::new(i);
        let _state = neuron_state(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;

    #[test]
    fn test_ecc_block_new() {
        let b = EccBlock::new();
        assert_eq!(b.parity, 0);
        assert_eq!(b.errors_detected, 0);
    }

    #[test]
    fn test_ecc_write_and_verify() {
        let mut b = EccBlock::new();
        let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
        b.write_block(&data);
        assert!(b.verify_and_correct());
    }

    #[test]
    fn test_bitflip_detector_new() {
        let d = BitFlipDetector::new();
        assert_eq!(d.soft_errors_detected, 0);
    }

    #[test]
    fn test_parity_check() {
        let mut d = BitFlipDetector::new();
        let data = [1u8, 2, 3, 4];
        assert!(d.check_parity(0, &data));
        assert!(!d.check_parity(0, &[5, 6, 7, 8]));
    }

    #[test]
    fn test_too_many_errors() {
        let d = BitFlipDetector::new();
        assert!(!d.has_too_many_errors(10));
    }

    #[test]
    fn test_memory_diagnostics_new() {
        let md = MemoryDiagnostics::new();
        assert_eq!(md.migration_count, 0);
        assert_eq!(md.bad_sector_count, 0);
    }

    #[test]
    fn test_mark_bad_sector() {
        let mut md = MemoryDiagnostics::new();
        md.mark_bad_sector(42);
        assert_eq!(md.bad_sector_count, 1);
        assert!(md.has_bad_sector(42));
    }

    #[test]
    fn test_senescence_score() {
        let md = MemoryDiagnostics::new();
        let score = md.senescence_score(0);
        assert_eq!(score, FixedPoint::ZERO);
        let score = md.senescence_score(SENESCENCE_MAX_AGE);
        assert_eq!(score, FixedPoint::ONE);
    }

    #[test]
    fn test_senescence_stage_healthy() {
        let md = MemoryDiagnostics::new();
        let stage = md.check_senescence(0);
        assert_eq!(stage, SenescenceStage::Healthy);
    }

    #[test]
    fn test_senescence_stage_eol() {
        let md = MemoryDiagnostics::new();
        let stage = md.check_senescence(SENESCENCE_MAX_AGE + 1);
        assert_eq!(stage, SenescenceStage::EndOfLife);
    }
}
