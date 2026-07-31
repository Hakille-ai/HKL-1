//! Neuromodulatory systems: dopamine, serotonin, noradrenaline, acetylcholine. Regulates learning, mood, and attention.
use crate::core::math::FixedPoint;

/// Neuromodulation system (Section 21)
/// Artificial hormones that regulate global network behavior

#[repr(C, align(8))]
pub struct CognitiveNeuromodulators {
    pub noradrenaline: FixedPoint, // Crisis/arousal: 0.0-1.0
    pub serotonin: FixedPoint,     // Stability/contentment: 0.0-1.0
    pub dopamine: FixedPoint,      // Reward/learning: 0.0-1.0
    pub acetylcholine: FixedPoint, // Attention/plasticity: 0.0-1.0
}

impl CognitiveNeuromodulators {
    pub const fn new() -> Self {
        Self {
            noradrenaline: FixedPoint::ZERO,
            serotonin: FixedPoint::from_f32(0.5),
            dopamine: FixedPoint::from_f32(0.3),
            acetylcholine: FixedPoint::from_f32(0.5),
        }
    }

    pub fn crisis_mode(&mut self) {
        self.noradrenaline = FixedPoint::ONE;
        self.serotonin = FixedPoint::ZERO;
        self.dopamine = FixedPoint::from_f32(0.5);
        self.acetylcholine = FixedPoint::ONE;
    }

    pub fn stability_mode(&mut self) {
        self.noradrenaline = FixedPoint::ZERO;
        self.serotonin = FixedPoint::ONE;
        self.dopamine = FixedPoint::from_f32(0.2);
        self.acetylcholine = FixedPoint::from_f32(0.3);
    }

    pub fn exploration_mode(&mut self) {
        self.noradrenaline = FixedPoint::from_f32(0.3);
        self.serotonin = FixedPoint::from_f32(0.3);
        self.dopamine = FixedPoint::ONE;
        self.acetylcholine = FixedPoint::ONE;
    }

    pub fn decay(&mut self, rate: FixedPoint) {
        let one_minus_rate = FixedPoint::ONE - rate;
        self.noradrenaline *= one_minus_rate;
        self.serotonin = self.serotonin * one_minus_rate + rate;
        self.dopamine *= one_minus_rate;
        self.acetylcholine *= one_minus_rate;
    }

    pub fn init_defaults(&mut self) {
        self.noradrenaline = FixedPoint::ZERO;
        self.serotonin = FixedPoint::ONE;
        self.dopamine = FixedPoint::from_f32(0.3);
        self.acetylcholine = FixedPoint::from_f32(0.5);
    }
}

impl Default for CognitiveNeuromodulators {
    fn default() -> Self {
        Self::new()
    }
}

pub static mut COGNITIVE_NEUROMODULATORS: CognitiveNeuromodulators =
    CognitiveNeuromodulators::new();

/// Sync COGNITIVE_NEUROMODULATORS → GLOBAL_NEUROMODULATORS.
/// Call after any cognitive-mode change so the SNN sees the updated values.
pub fn sync_to_snn() {
    unsafe {
        let snn_nm = &mut *crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
        snn_nm.noradrenaline = COGNITIVE_NEUROMODULATORS.noradrenaline;
        snn_nm.serotonin = COGNITIVE_NEUROMODULATORS.serotonin;
        snn_nm.dopamine = COGNITIVE_NEUROMODULATORS.dopamine;
        snn_nm.acetylcholine = COGNITIVE_NEUROMODULATORS.acetylcholine;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_baseline() {
        let nm = CognitiveNeuromodulators::new();
        assert_eq!(nm.noradrenaline, FixedPoint::ZERO);
        assert_eq!(nm.serotonin, FixedPoint::from_f32(0.5));
        assert_eq!(nm.dopamine, FixedPoint::from_f32(0.3));
        assert_eq!(nm.acetylcholine, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_crisis_mode() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.crisis_mode();
        assert_eq!(nm.noradrenaline, FixedPoint::ONE);
        assert_eq!(nm.serotonin, FixedPoint::ZERO);
        assert_eq!(nm.acetylcholine, FixedPoint::ONE);
    }

    #[test]
    fn test_stability_mode() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.stability_mode();
        assert_eq!(nm.noradrenaline, FixedPoint::ZERO);
        assert_eq!(nm.serotonin, FixedPoint::ONE);
        assert_eq!(nm.dopamine, FixedPoint::from_f32(0.2));
    }

    #[test]
    fn test_exploration_mode() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.exploration_mode();
        assert_eq!(nm.dopamine, FixedPoint::ONE);
        assert_eq!(nm.acetylcholine, FixedPoint::ONE);
        assert_eq!(nm.noradrenaline, FixedPoint::from_f32(0.3));
        assert_eq!(nm.serotonin, FixedPoint::from_f32(0.3));
    }

    #[test]
    fn test_decay_reduces_values() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.dopamine = FixedPoint::ONE;
        nm.acetylcholine = FixedPoint::ONE;
        nm.decay(FixedPoint::from_f32(0.1));
        assert!(nm.dopamine < FixedPoint::ONE);
        assert!(nm.noradrenaline == FixedPoint::ZERO);
    }

    #[test]
    fn test_decay_serotonin_increases() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.serotonin = FixedPoint::ZERO;
        nm.decay(FixedPoint::from_f32(0.1));
        assert!(nm.serotonin > FixedPoint::ZERO);
    }

    #[test]
    fn test_init_defaults() {
        let mut nm = CognitiveNeuromodulators::new();
        nm.crisis_mode();
        nm.init_defaults();
        assert_eq!(nm.noradrenaline, FixedPoint::ZERO);
        assert_eq!(nm.serotonin, FixedPoint::ONE);
        assert_eq!(nm.dopamine, FixedPoint::from_f32(0.3));
        assert_eq!(nm.acetylcholine, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_default_trait() {
        let nm: CognitiveNeuromodulators = Default::default();
        assert_eq!(nm.noradrenaline, FixedPoint::ZERO);
        assert_eq!(nm.serotonin, FixedPoint::from_f32(0.5));
    }
}
