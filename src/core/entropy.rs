//! Entropy monitoring for cognitive health (Shannon entropy of weight
//! distributions), cognitive entropy for action/prediction diversity, and
//! stochastic noise sources (PRNG, hardware TRNG, thermal) for curiosity
//! and dreaming.

use crate::core::math::{FixedPoint, XorShift64Star};
use core::sync::atomic::{AtomicU32, Ordering};

/// Cognitive mode derived from entropy
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CognitiveMode {
    Crisis,   // High entropy — stabilize, reduce LR
    Explore,  // Moderate entropy — encourage exploration
    Stable,   // Low entropy — maintain, consolidate
    Balanced, // Healthy range — normal operation
}

/// Entropy monitor for cognitive health (Section 32)
pub struct EntropyMonitor {
    // Sliding window of weight distributions
    weight_histogram: [u32; 256], // 8-bit weight histogram
    histogram_age: u32,
    // Entropy estimates
    current_entropy: FixedPoint,
    _entropy_critical: AtomicU32, // 0=normal, 1=high, 2=low
    _last_alert_time: AtomicU32,
    // Adaptive threshold tracking
    entropy_history: [FixedPoint; 64],
    entropy_idx: u8,
    entropy_samples: u8,
    entropy_mean: FixedPoint,
    entropy_std: FixedPoint,
    adapt_cooldown: u32,
    /// Derived cognitive mode (set by adapt_thresholds)
    pub cognitive_mode: CognitiveMode,
    /// Adaptive min/max thresholds (public for safety monitor delegation)
    pub target_entropy_min: FixedPoint,
    pub target_entropy_max: FixedPoint,
}

impl EntropyMonitor {
    pub const fn new() -> Self {
        Self {
            weight_histogram: [0; 256],
            histogram_age: 0,
            current_entropy: FixedPoint::ZERO,
            target_entropy_min: FixedPoint::from_f32(3.0),
            target_entropy_max: FixedPoint::from_f32(7.0),
            _entropy_critical: AtomicU32::new(0),
            _last_alert_time: AtomicU32::new(0),
            entropy_history: [FixedPoint::ZERO; 64],
            entropy_idx: 0,
            entropy_samples: 0,
            entropy_mean: FixedPoint::ZERO,
            entropy_std: FixedPoint::ZERO,
            adapt_cooldown: 0,
            cognitive_mode: CognitiveMode::Balanced,
        }
    }

    /// Update histogram with current weight distribution
    pub fn sample_weights(&mut self, weights: &[crate::core::math::Weight]) {
        // Clear histogram periodically
        if self.histogram_age > 10000 {
            self.weight_histogram = [0; 256];
            self.histogram_age = 0;
        }

        for &w in weights {
            let idx = ((w.0 as i32 + 32768) >> 8) as usize; // Map i16 to 0-255
            if idx < 256 {
                self.weight_histogram[idx] = self.weight_histogram[idx].saturating_add(1);
            }
        }
        self.histogram_age += 1;
    }

    /// Calculate Shannon entropy of weight distribution
    pub fn compute_entropy(&mut self, total_synapses: usize) -> FixedPoint {
        if total_synapses == 0 {
            return FixedPoint::ZERO;
        }

        let total = FixedPoint::from_int(total_synapses as i32);
        let mut entropy = FixedPoint::ZERO;

        for &count in &self.weight_histogram {
            if count > 0 {
                let p = FixedPoint::from_int(count as i32).div(total);
                entropy = entropy - p * p.ln();
            }
        }

        self.current_entropy = entropy;
        self.current_entropy
    }

    /// Check if entropy is in healthy range
    pub fn check_health(&self) -> EntropyState {
        if self.histogram_age == 0 || self.current_entropy < FixedPoint::from_f32(0.01) {
            return EntropyState::Healthy; // Not enough data yet
        }
        if self.current_entropy > self.target_entropy_max {
            EntropyState::HighEntropy // Too noisy - crystallize
        } else if self.current_entropy < self.target_entropy_min {
            EntropyState::LowEntropy // Stagnation - inject noise
        } else {
            EntropyState::Healthy
        }
    }

    /// Record entropy sample and adapt thresholds
    pub fn record_entropy(&mut self, entropy: FixedPoint) {
        self.current_entropy = entropy;
        self.entropy_history[self.entropy_idx as usize] = entropy;
        self.entropy_idx = (self.entropy_idx + 1) % 64;
        if self.entropy_samples < 64 {
            self.entropy_samples += 1;
        }
    }

    /// Recompute mean + std from history, then adapt thresholds
    pub fn adapt_thresholds(&mut self) {
        let n = self.entropy_samples.max(2) as usize;
        if n < 2 {
            return;
        }

        // Compute mean
        let mut sum = FixedPoint::ZERO;
        for i in 0..n {
            sum += self.entropy_history[i];
        }
        self.entropy_mean = sum / FixedPoint::from_int(n as i32);

        // Compute std
        let mut var_sum = FixedPoint::ZERO;
        for i in 0..n {
            let diff = self.entropy_history[i] - self.entropy_mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / FixedPoint::from_int(n as i32);
        self.entropy_std = variance.sqrt();

        // Adaptive thresholds: mean ± 2*std, clamped to reasonable bounds
        let two_sigma = self.entropy_std * FixedPoint::from_int(2);
        let new_min = (self.entropy_mean - two_sigma)
            .clamp(FixedPoint::from_f32(1.0), FixedPoint::from_f32(6.0));
        let new_max = (self.entropy_mean + two_sigma)
            .clamp(FixedPoint::from_f32(4.0), FixedPoint::from_f32(7.5));

        // Smooth update
        let alpha = FixedPoint::from_f32(0.3);
        let one_minus_alpha = FixedPoint::ONE - alpha;
        self.target_entropy_min = self.target_entropy_min * one_minus_alpha + new_min * alpha;
        self.target_entropy_max = self.target_entropy_max * one_minus_alpha + new_max * alpha;

        // Derive cognitive mode
        self.cognitive_mode = self.derive_mode();
    }

    /// Map current entropy + history to a cognitive mode
    pub fn derive_mode(&self) -> CognitiveMode {
        let e = self.current_entropy;
        let min = self.target_entropy_min;
        let max = self.target_entropy_max;

        if e > max {
            CognitiveMode::Crisis
        } else if e < min {
            CognitiveMode::Stable
        } else {
            let mid = (min + max) / FixedPoint::from_int(2);
            if e > mid {
                CognitiveMode::Explore
            } else {
                CognitiveMode::Balanced
            }
        }
    }

    /// Apply cognitive mode to neuromodulators
    pub fn apply_cognitive_mode(&self) {
        match self.cognitive_mode {
            CognitiveMode::Crisis => unsafe {
                crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.crisis_mode();
            },
            CognitiveMode::Explore => unsafe {
                crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.exploration_mode();
            },
            CognitiveMode::Stable => unsafe {
                crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.stability_mode();
            },
            CognitiveMode::Balanced => {
                // Let natural dynamics play out
            }
        }
        crate::cognitive::neuromodulation::sync_to_snn();
    }

    /// Full update: sample → record → adapt → apply
    pub fn cognitive_update(&mut self, entropy: FixedPoint) {
        self.record_entropy(entropy);
        if self.adapt_cooldown == 0 {
            self.adapt_thresholds();
            self.apply_cognitive_mode();
            self.adapt_cooldown = 10; // Recompute every 10 calls
        } else {
            self.adapt_cooldown -= 1;
        }
    }

    /// Trigger crystallization (high entropy -> stabilize)
    pub fn force_crystallization(&self, _network: &mut crate::snn::network::Network) {
        // Reduce learning rates, increase thresholds
        unsafe {
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.serotonin +=
                crate::core::math::FixedPoint::from_f32(0.5);
        }
        crate::cognitive::neuromodulation::sync_to_snn();
    }

    /// Inject controlled noise (low entropy -> explore)
    pub fn inject_stochastic_noise(&self, _network: &mut crate::snn::network::Network) {
        // Increase curiosity, add random spikes
        unsafe {
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.noradrenaline +=
                crate::core::math::FixedPoint::from_f32(0.3);
            crate::cognitive::curiosity::CURIOSITY_ENGINE.activate_dreaming();
        }
        crate::cognitive::neuromodulation::sync_to_snn();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntropyState {
    Healthy,
    HighEntropy, // Crystallization needed
    LowEntropy,  // Noise injection needed
}

/// Cognitive entropy - higher level behavioral entropy
pub struct CognitiveEntropy {
    // Action distribution entropy
    action_counts: [u32; 256],
    // Prediction error entropy
    prediction_errors: [u16; 1024],
    error_head: AtomicU32,
    // Entropy metrics
    action_entropy: FixedPoint,
    prediction_entropy: FixedPoint,
}

impl CognitiveEntropy {
    pub const fn new() -> Self {
        Self {
            action_counts: [0; 256],
            prediction_errors: [0; 1024],
            error_head: AtomicU32::new(0),
            action_entropy: FixedPoint::ZERO,
            prediction_entropy: FixedPoint::ZERO,
        }
    }

    pub fn record_action(&mut self, action_id: u8) {
        self.action_counts[action_id as usize] =
            self.action_counts[action_id as usize].saturating_add(1);
    }

    pub fn record_prediction_error(&mut self, error: u16) {
        let head = self.error_head.fetch_add(1, Ordering::AcqRel) % 1024;
        self.prediction_errors[head as usize] = error;
    }

    pub fn compute_action_entropy(&mut self, total_actions: u32) -> FixedPoint {
        if total_actions == 0 {
            return FixedPoint::ZERO;
        }
        let total = FixedPoint::from_int(total_actions as i32);
        let mut entropy = FixedPoint::ZERO;
        for &count in &self.action_counts {
            if count > 0 {
                let p = FixedPoint::from_int(count as i32).div(total);
                entropy = entropy - p * p.ln();
            }
        }
        self.action_entropy = entropy;
        self.action_entropy
    }

    /// Map action entropy to curiosity level
    pub fn entropy_to_curiosity(&self) -> FixedPoint {
        let max_entropy = FixedPoint::from_f32(5.55); // ln(256)
        let ratio = self.action_entropy / max_entropy;
        (ratio * FixedPoint::from_f32(0.8) + FixedPoint::from_f32(0.1))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE)
    }

    /// Map prediction entropy to acetylcholine drive
    pub fn entropy_to_acetylcholine(&self) -> FixedPoint {
        let max_entropy = FixedPoint::from_f32(2.77); // ln(16)
        let ratio = self.prediction_entropy / max_entropy;
        (ratio * FixedPoint::from_f32(0.7) + FixedPoint::from_f32(0.2))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE)
    }

    pub fn compute_prediction_entropy(&mut self) -> FixedPoint {
        let mut hist = [0u32; 16];
        for &e in &self.prediction_errors {
            let bin = (e >> 12) as usize;
            hist[bin] = hist[bin].saturating_add(1);
        }

        let total = FixedPoint::from_int(self.prediction_errors.len() as i32);
        let mut entropy = FixedPoint::ZERO;
        for &count in &hist {
            if count > 0 {
                let p = FixedPoint::from_int(count as i32).div(total);
                entropy = entropy - p * p.ln();
            }
        }
        self.prediction_entropy = entropy;
        self.prediction_entropy
    }
}

/// Stochastic noise generator for curiosity/dreaming (Section 29)
pub struct StochasticNoiseSource {
    rng: XorShift64Star,
    // Hardware TRNG if available
    trng_available: bool,
    trng_reg: *mut u32,
    // Thermal noise from CPU temp sensor
    temp_sensor_reg: *mut u32,
}

impl StochasticNoiseSource {
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: XorShift64Star::new(seed),
            trng_available: false,
            trng_reg: core::ptr::null_mut(),
            temp_sensor_reg: core::ptr::null_mut(),
        }
    }

    pub fn enable_trng(&mut self, reg: *mut u32) {
        self.trng_reg = reg;
        self.trng_available = true;
    }

    pub fn enable_thermal_noise(&mut self, reg: *mut u32) {
        self.temp_sensor_reg = reg;
    }

    /// Generate spike noise for dreaming
    pub fn generate_spike_noise(
        &mut self,
        rate_hz: f32,
        duration_ms: u32,
        out: &mut [SpikeEvent],
    ) -> usize {
        let interval_us = (1_000_000.0 / rate_hz) as u32;
        let num_spikes = (rate_hz * duration_ms as f32 / 1000.0) as usize;
        let count = num_spikes.min(out.len());

        for i in 0..count {
            let neuron_id = self.rng.next_u32() as u16 % 4096;
            let time_offset = (i as u32 * interval_us) + (self.rng.next_u32() % (interval_us / 10));
            out[i] = SpikeEvent {
                neuron_id: crate::core::memory::NeuronId::new(neuron_id),
                timestamp: time_offset,
                source: SpikeSource::Noise,
            };
        }
        count
    }

    pub fn thermal_sample(&mut self) -> u32 {
        if !self.temp_sensor_reg.is_null() {
            unsafe { *self.temp_sensor_reg }
        } else {
            self.rng.next_u32()
        }
    }

    pub fn trng_sample(&mut self) -> u32 {
        if self.trng_available {
            unsafe { *self.trng_reg }
        } else {
            self.rng.next_u32()
        }
    }
}

#[derive(Clone, Copy)]
pub struct SpikeEvent {
    pub neuron_id: crate::core::memory::NeuronId,
    pub timestamp: u32,
    pub source: SpikeSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpikeSource {
    Sensor,
    Noise,
    Dream,
    Reflex,
    InterNetwork,
}

/// Global entropy monitor instance
pub static mut ENTROPY_MONITOR: EntropyMonitor = EntropyMonitor::new();
pub static mut COGNITIVE_ENTROPY: CognitiveEntropy = CognitiveEntropy::new();
pub static mut NOISE_SOURCE: StochasticNoiseSource = StochasticNoiseSource::new(0xDEADBEEF);

pub fn init_entropy(seed: u64) {
    unsafe {
        ENTROPY_MONITOR = EntropyMonitor::new();
        COGNITIVE_ENTROPY = CognitiveEntropy::new();
        NOISE_SOURCE = StochasticNoiseSource::new(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_monitor_new_default() {
        let em = EntropyMonitor::new();
        assert_eq!(em.current_entropy, FixedPoint::ZERO);
        assert_eq!(em.target_entropy_min.to_f32(), 3.0);
        assert_eq!(em.target_entropy_max.to_f32(), 7.0);
        assert_eq!(em.cognitive_mode, CognitiveMode::Balanced);
    }

    #[test]
    fn compute_entropy_returns_non_zero() {
        let mut em = EntropyMonitor::new();
        // Use diverse weights to get non-zero entropy
        for i in 0..256 {
            let raw = ((i as i32) << 8) - 32768;
            let w = crate::core::math::Weight(raw as i16);
            for _ in 0..10 {
                em.sample_weights(&[w]);
            }
        }
        let e = em.compute_entropy(2560);
        assert!(e > FixedPoint::from_f32(2.0), "entropy={}", e.to_f32());
    }

    #[test]
    fn compute_entropy_zero_for_empty() {
        let mut em = EntropyMonitor::new();
        assert_eq!(em.compute_entropy(0), FixedPoint::ZERO);
    }

    #[test]
    fn record_entropy_samples() {
        let mut em = EntropyMonitor::new();
        em.record_entropy(FixedPoint::from_f32(4.0));
        em.record_entropy(FixedPoint::from_f32(5.0));
        assert_eq!(em.entropy_samples, 2);
    }

    #[test]
    fn adapt_thresholds_moves_toward_data() {
        let mut em = EntropyMonitor::new();
        // Record a bunch of consistent entropy readings near 5.0
        for _ in 0..20 {
            em.record_entropy(FixedPoint::from_f32(5.0));
        }
        em.adapt_thresholds();
        // Mean should be near 5.0
        let diff = (em.entropy_mean - FixedPoint::from_f32(5.0)).abs();
        assert!(
            diff < FixedPoint::from_f32(1.0),
            "mean={} expected~5.0",
            em.entropy_mean.to_f32()
        );
    }

    #[test]
    fn derive_mode_crisis_for_high_entropy() {
        let mut em = EntropyMonitor::new();
        for _ in 0..20 {
            em.record_entropy(FixedPoint::from_f32(9.0));
        }
        em.adapt_thresholds();
        assert_eq!(
            em.cognitive_mode,
            CognitiveMode::Crisis,
            "got {:?} mean={} min={} max={}",
            em.cognitive_mode,
            em.entropy_mean.to_f32(),
            em.target_entropy_min.to_f32(),
            em.target_entropy_max.to_f32()
        );
    }

    #[test]
    fn derive_mode_stable_for_low_entropy() {
        let mut em = EntropyMonitor::new();
        for _ in 0..20 {
            em.record_entropy(FixedPoint::from_f32(1.0));
        }
        em.adapt_thresholds();
        assert_eq!(em.cognitive_mode, CognitiveMode::Stable);
    }

    #[test]
    fn apply_cognitive_mode_crisis_sets_noradrenaline() {
        let mut em = EntropyMonitor::new();
        em.cognitive_mode = CognitiveMode::Crisis;
        // Reset neuromodulators first
        unsafe {
            crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS =
                crate::cognitive::neuromodulation::CognitiveNeuromodulators::new();
        }
        em.apply_cognitive_mode();
        unsafe {
            assert_eq!(
                crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS.noradrenaline,
                FixedPoint::ONE
            );
        }
    }

    #[test]
    fn cognitive_update_integration() {
        let mut em = EntropyMonitor::new();
        em.cognitive_update(FixedPoint::from_f32(4.0));
        // Mode should be set
        assert!(em.entropy_samples > 0);
    }

    #[test]
    fn cognitive_entropy_new_default() {
        let ce = CognitiveEntropy::new();
        assert_eq!(ce.action_entropy, FixedPoint::ZERO);
    }

    #[test]
    fn cognitive_entropy_record_action() {
        let mut ce = CognitiveEntropy::new();
        ce.record_action(5);
        assert_eq!(ce.action_counts[5], 1);
    }

    #[test]
    fn cognitive_entropy_compute_action_entropy() {
        let mut ce = CognitiveEntropy::new();
        ce.record_action(0);
        ce.record_action(1);
        ce.record_action(2);
        let e = ce.compute_action_entropy(3);
        assert!(e > FixedPoint::ZERO);
    }

    #[test]
    fn entropy_to_curiosity_in_range() {
        let ce = CognitiveEntropy::new();
        let c = ce.entropy_to_curiosity();
        assert!(c >= FixedPoint::ZERO && c <= FixedPoint::ONE);
    }

    #[test]
    fn entropy_to_acetylcholine_in_range() {
        let ce = CognitiveEntropy::new();
        let a = ce.entropy_to_acetylcholine();
        assert!(a >= FixedPoint::ZERO && a <= FixedPoint::ONE);
    }

    #[test]
    fn check_health_healthy_initially() {
        let em = EntropyMonitor::new();
        assert_eq!(em.check_health(), EntropyState::Healthy);
    }
}
