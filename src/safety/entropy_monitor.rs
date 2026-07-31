use crate::core::math::FixedPoint;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntropyState {
    Healthy,
    HighEntropy,
    LowEntropy,
}

pub struct EntropyMonitor {
    pub weight_entropy: FixedPoint,
    pub activity_entropy: FixedPoint,
    pub cognitive_entropy: FixedPoint,
    pub target_min: FixedPoint,
    pub target_max: FixedPoint,
    pub high_threshold: u32,
    pub consecutive_high: u32,
    pub consecutive_low: u32,
    pub last_state: EntropyState,
    adapt_cooldown: u32,
}

impl EntropyMonitor {
    pub const fn new() -> Self {
        Self {
            weight_entropy: FixedPoint::ZERO,
            activity_entropy: FixedPoint::ZERO,
            cognitive_entropy: FixedPoint::ZERO,
            target_min: FixedPoint::from_f32(0.1),
            target_max: FixedPoint::from_f32(0.8),
            high_threshold: 10,
            consecutive_high: 0,
            consecutive_low: 0,
            last_state: EntropyState::Healthy,
            adapt_cooldown: 0,
        }
    }

    pub fn check_health(&self) -> EntropyState {
        if self.weight_entropy > self.target_max {
            EntropyState::HighEntropy
        } else if self.weight_entropy < self.target_min {
            EntropyState::LowEntropy
        } else {
            EntropyState::Healthy
        }
    }

    /// Record an entropy sample and adapt thresholds periodically.
    /// Delegates adaptive computation to core entropy monitor.
    pub fn record_entropy(&mut self, entropy: FixedPoint) {
        self.weight_entropy = entropy;
        let core = unsafe { &mut crate::core::entropy::ENTROPY_MONITOR };
        core.record_entropy(entropy);
        if self.adapt_cooldown == 0 {
            core.adapt_thresholds();
            self.adapt_targets(core);
            self.adapt_cooldown = 20;
        } else {
            self.adapt_cooldown -= 1;
        }
        self.update_consecutive();
    }

    /// Pull adaptive thresholds from the core entropy monitor
    fn adapt_targets(&mut self, core: &crate::core::entropy::EntropyMonitor) {
        let core_min = core.target_entropy_min.to_f32();
        let core_max = core.target_entropy_max.to_f32();
        let safety_min = core_min / 8.0;
        let safety_max = core_max / 8.0;
        let alpha = FixedPoint::from_f32(0.2);
        let oma = FixedPoint::ONE - alpha;
        self.target_min = self.target_min * oma + FixedPoint::from_f32(safety_min) * alpha;
        self.target_max = self.target_max * oma + FixedPoint::from_f32(safety_max) * alpha;
        self.target_min = self
            .target_min
            .clamp(FixedPoint::from_f32(0.01), FixedPoint::from_f32(0.5));
        self.target_max = self
            .target_max
            .clamp(FixedPoint::from_f32(0.3), FixedPoint::from_f32(1.5));
    }

    fn update_consecutive(&mut self) {
        let state = self.check_health();
        match state {
            EntropyState::HighEntropy => {
                self.consecutive_high += 1;
                self.consecutive_low = 0;
            }
            EntropyState::LowEntropy => {
                self.consecutive_low += 1;
                self.consecutive_high = 0;
            }
            EntropyState::Healthy => {
                self.consecutive_high = 0;
                self.consecutive_low = 0;
            }
        }
        self.last_state = state;
    }

    pub fn force_crystallization(&self, network: &mut crate::snn::network::Network) {
        network.neuromodulators.serotonin = FixedPoint::ONE;
        network.neuromodulators.noradrenaline = FixedPoint::ZERO;
    }

    pub fn inject_stochastic_noise(&self, network: &mut crate::snn::network::Network) {
        network.neuromodulators.dopamine = FixedPoint::ONE;
        network.neuromodulators.acetylcholine = FixedPoint::ONE;
        unsafe {
            crate::cognitive::curiosity::CURIOSITY_ENGINE.activate_dreaming();
        }
    }

    /// Returns true if entropy has been abnormal for longer than the threshold
    pub fn is_persistent_anomaly(&self) -> bool {
        self.consecutive_high > self.high_threshold || self.consecutive_low > self.high_threshold
    }
}

pub static mut ENTROPY_MONITOR: EntropyMonitor = EntropyMonitor::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let m = EntropyMonitor::new();
        assert_eq!(m.target_min, FixedPoint::from_f32(0.1));
        assert_eq!(m.target_max, FixedPoint::from_f32(0.8));
        assert_eq!(m.last_state, EntropyState::Healthy);
    }

    #[test]
    fn test_check_healthy_in_range() {
        let m = EntropyMonitor {
            weight_entropy: FixedPoint::from_f32(0.5),
            ..EntropyMonitor::new()
        };
        assert_eq!(m.check_health(), EntropyState::Healthy);
    }

    #[test]
    fn test_check_high_entropy() {
        let m = EntropyMonitor {
            weight_entropy: FixedPoint::from_f32(0.9),
            ..EntropyMonitor::new()
        };
        assert_eq!(m.check_health(), EntropyState::HighEntropy);
    }

    #[test]
    fn test_check_low_entropy() {
        let m = EntropyMonitor {
            weight_entropy: FixedPoint::from_f32(0.05),
            ..EntropyMonitor::new()
        };
        assert_eq!(m.check_health(), EntropyState::LowEntropy);
    }

    #[test]
    fn test_record_entropy_updates_weight() {
        let mut m = EntropyMonitor::new();
        m.record_entropy(FixedPoint::from_f32(0.5));
        assert_eq!(m.weight_entropy, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_consecutive_counting() {
        let mut m = EntropyMonitor::new();
        m.target_min = FixedPoint::from_f32(0.3);
        for _ in 0..5 {
            m.weight_entropy = FixedPoint::from_f32(0.1);
            m.update_consecutive();
        }
        assert_eq!(m.consecutive_low, 5);
        assert_eq!(m.consecutive_high, 0);
    }

    #[test]
    fn test_persistent_anomaly() {
        let mut m = EntropyMonitor::new();
        m.high_threshold = 3;
        m.target_min = FixedPoint::from_f32(0.3);
        for _ in 0..5 {
            m.weight_entropy = FixedPoint::from_f32(0.9);
            m.update_consecutive();
        }
        assert!(m.is_persistent_anomaly());
    }

    #[test]
    fn test_adapt_thresholds_narrow_on_stable() {
        let mut m = EntropyMonitor::new();
        for _ in 0..30 {
            m.record_entropy(FixedPoint::from_f32(0.5));
        }
        assert!(m.target_min > FixedPoint::from_f32(0.01));
        assert!(m.target_max < FixedPoint::from_f32(1.5));
    }

    #[test]
    fn test_force_crystallization_sets_serotonin() {
        let mut net = crate::snn::network::Network::new();
        let m = EntropyMonitor::new();
        net.neuromodulators.serotonin = FixedPoint::ZERO;
        m.force_crystallization(&mut net);
        assert_eq!(net.neuromodulators.serotonin, FixedPoint::ONE);
    }

    #[test]
    fn test_is_persistent_anomaly_false_by_default() {
        let m = EntropyMonitor::new();
        assert!(!m.is_persistent_anomaly());
    }
}
