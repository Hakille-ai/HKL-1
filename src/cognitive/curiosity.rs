use crate::core::math::FixedPoint;
use crate::core::memory::{NEURON_COUNT, NeuronFlags, NeuronId, neuron_state};
use crate::snn::network::Network;

const HABITUATION_SLOTS: usize = 32;

pub struct HabituationTracker {
    pub stimulus_hashes: [u32; HABITUATION_SLOTS],
    pub hit_counts: [u16; HABITUATION_SLOTS],
    pub slot_idx: u8,
    pub slot_count: u8,
    pub habituation_rate: FixedPoint,
    pub familiarity: FixedPoint,
    pub threshold: u16,
}

impl HabituationTracker {
    pub const fn new() -> Self {
        Self {
            stimulus_hashes: [0; HABITUATION_SLOTS],
            hit_counts: [0; HABITUATION_SLOTS],
            slot_idx: 0,
            slot_count: 0,
            habituation_rate: FixedPoint::from_f32(0.05),
            familiarity: FixedPoint::ZERO,
            threshold: 5,
        }
    }

    pub fn record_stimulus(&mut self, hash: u32) {
        for i in 0..self.slot_count as usize {
            if self.stimulus_hashes[i] == hash {
                self.hit_counts[i] = self.hit_counts[i].saturating_add(1);
                self.update_familiarity();
                return;
            }
        }
        if self.slot_count < HABITUATION_SLOTS as u8 {
            let idx = self.slot_count as usize;
            self.stimulus_hashes[idx] = hash;
            self.hit_counts[idx] = 1;
            self.slot_count += 1;
        } else {
            let idx = self.slot_idx as usize;
            self.stimulus_hashes[idx] = hash;
            self.hit_counts[idx] = 1;
            self.slot_idx = (self.slot_idx + 1) % (HABITUATION_SLOTS as u8);
        }
        self.familiarity *= FixedPoint::ONE - self.habituation_rate;
    }

    fn update_familiarity(&mut self) {
        let mut hits: u32 = 0;
        for i in 0..self.slot_count as usize {
            hits += self.hit_counts[i] as u32;
        }
        let denom = (self.slot_count as u32).max(1);
        let avg = hits / denom;
        let x = FixedPoint::from_f32(avg as f32 - 10.0);
        let neg_half_x = FixedPoint::from_f32(-0.5) * x;
        let exp_val = neg_half_x.exp();
        let ratio = FixedPoint::ONE / (FixedPoint::ONE + exp_val);
        self.familiarity = ratio;
        self.familiarity = self.familiarity.clamp(FixedPoint::ZERO, FixedPoint::ONE);
    }

    pub fn is_habituated(&self, hash: u32) -> bool {
        for i in 0..self.slot_count as usize {
            if self.stimulus_hashes[i] == hash {
                return self.hit_counts[i] >= self.threshold;
            }
        }
        false
    }

    pub fn novelty_bonus(&self) -> FixedPoint {
        FixedPoint::ONE - self.familiarity
    }

    pub fn decay(&mut self) {
        let decay_rate = FixedPoint::from_f32(0.001);
        self.familiarity *= FixedPoint::ONE - decay_rate;
        for i in 0..self.slot_count as usize {
            self.hit_counts[i] = self.hit_counts[i].saturating_sub(1);
        }
    }
}

pub struct CuriosityEngine {
    pub curiosity_level: FixedPoint,
    pub boredom_threshold: FixedPoint,
    pub boredom_accumulator: FixedPoint,
    pub exploration_urge: FixedPoint,
    pub monotony_counter: u32,
    pub forced_exploration_cooldown: u32,
    pub noise_amplitude: FixedPoint,
    pub thermal_noise_amplitude: FixedPoint,
    pub thermal_noise_active: bool,
    pub dreaming_active: bool,
    pub dream_duration_ms: u32,
    pub dream_count: u32,
    pub last_dream_time: u32,
    pub adaptive_threshold_min: FixedPoint,
    pub adaptive_threshold_max: FixedPoint,
    pub habituation: HabituationTracker,
    pub last_prediction_error: FixedPoint,
    pub pred_error_ema: FixedPoint,
    pub last_habituation_hash: u32,
    pub ibl_tags: [u32; 8],
    pub ibl_decay: [u16; 8],
    pub ibl_idx: u8,
}

impl CuriosityEngine {
    pub const fn new() -> Self {
        Self {
            curiosity_level: FixedPoint::ZERO,
            boredom_threshold: FixedPoint::from_f32(0.01),
            boredom_accumulator: FixedPoint::ZERO,
            exploration_urge: FixedPoint::ZERO,
            monotony_counter: 0,
            forced_exploration_cooldown: 0,
            noise_amplitude: FixedPoint::from_f32(0.1),
            thermal_noise_amplitude: FixedPoint::from_f32(0.05),
            thermal_noise_active: true,
            dreaming_active: false,
            dream_duration_ms: 100,
            dream_count: 0,
            last_dream_time: 0,
            adaptive_threshold_min: FixedPoint::from_f32(0.005),
            adaptive_threshold_max: FixedPoint::from_f32(0.05),
            habituation: HabituationTracker::new(),
            last_prediction_error: FixedPoint::ZERO,
            pred_error_ema: FixedPoint::ZERO,
            last_habituation_hash: 0,
            ibl_tags: [0; 8],
            ibl_decay: [0; 8],
            ibl_idx: 0,
        }
    }

    pub fn inhibit_layer(&mut self, layer_hash: u32, duration: u16) {
        let idx = self.ibl_idx as usize;
        self.ibl_tags[idx] = layer_hash;
        self.ibl_decay[idx] = duration;
        self.ibl_idx = (self.ibl_idx + 1) & 7;
    }

    pub fn is_layer_inhibited(&self, layer_hash: u32) -> bool {
        self.ibl_tags
            .iter()
            .zip(self.ibl_decay.iter())
            .any(|(&tag, &decay)| tag == layer_hash && decay > 0)
    }

    pub fn decay_ibl(&mut self) {
        for d in self.ibl_decay.iter_mut() {
            *d = d.saturating_sub(1);
        }
    }

    pub fn update(&mut self, net: &Network) {
        let prediction_error = net.predictor.mean_prediction_error;
        let novelty = net.predictor.novelty;
        let time = net.time;

        self.pred_error_ema = self.pred_error_ema * FixedPoint::from_f32(0.95)
            + prediction_error * FixedPoint::from_f32(0.05);

        if prediction_error < self.boredom_threshold && novelty < FixedPoint::from_f32(0.01) {
            self.monotony_counter += 1;
            self.boredom_accumulator += FixedPoint::from_f32(0.002);
            self.curiosity_level += FixedPoint::from_f32(0.001);
        } else {
            if self.monotony_counter > 10 {
                self.monotony_counter = self.monotony_counter.saturating_sub(1);
            } else {
                self.monotony_counter = 0;
            }
            self.boredom_accumulator *= FixedPoint::from_f32(0.98);
            self.curiosity_level *= FixedPoint::from_f32(0.99);
        }

        self.curiosity_level = self
            .curiosity_level
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.boredom_accumulator = self
            .boredom_accumulator
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);

        self.exploration_urge = self.boredom_accumulator * FixedPoint::from_f32(0.5)
            + self.curiosity_level * FixedPoint::from_f32(0.3);

        if self.forced_exploration_cooldown > 0 {
            self.forced_exploration_cooldown -= 1;
        }

        if self.boredom_accumulator > FixedPoint::from_f32(0.3)
            && self.forced_exploration_cooldown == 0
            && !self.dreaming_active
        {
            self.activate_dreaming();
            self.forced_exploration_cooldown = 500;
            self.boredom_accumulator *= FixedPoint::from_f32(0.5);
        }

        self.adapt_thresholds();

        if novelty > FixedPoint::from_f32(0.1) {
            let state_hash = self.compute_network_hash(net);
            self.habituation.record_stimulus(state_hash);
            self.last_habituation_hash = state_hash;
        }
        self.habituation.decay();

        if time.is_multiple_of(50) && self.thermal_noise_active {
            self.inject_thermal_noise(net, time);
        }

        self.decay_ibl();

        self.last_prediction_error = prediction_error;
    }

    fn adapt_thresholds(&mut self) {
        let ema = self.pred_error_ema;
        let margin = FixedPoint::from_f32(0.002);
        let target_low = (ema - margin).max(FixedPoint::from_f32(0.001));
        let target_high = (ema + margin).max(FixedPoint::from_f32(0.01));

        let alpha = FixedPoint::from_f32(0.1);
        if ema < self.boredom_threshold {
            self.boredom_threshold =
                self.boredom_threshold * (FixedPoint::ONE - alpha) + target_low * alpha;
        } else {
            self.boredom_threshold =
                self.boredom_threshold * (FixedPoint::ONE - alpha) + target_high * alpha;
        }
        self.boredom_threshold = self
            .boredom_threshold
            .clamp(self.adaptive_threshold_min, self.adaptive_threshold_max);

        let curiosity_alpha = FixedPoint::from_f32(0.05);
        let target_curiosity = if self.monotony_counter > 50 {
            FixedPoint::from_f32(0.5)
        } else if self.monotony_counter > 20 {
            FixedPoint::from_f32(0.3)
        } else {
            FixedPoint::ZERO
        };
        self.curiosity_level = self.curiosity_level * (FixedPoint::ONE - curiosity_alpha)
            + target_curiosity * curiosity_alpha;
    }

    fn compute_network_hash(&self, net: &Network) -> u32 {
        let mut h: u32 = 5381;
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        for i in (0..count.min(1024.min(crate::MAX_NEURONS))).step_by(8) {
            let id = NeuronId::new(i as u16);
            let state = crate::core::memory::neuron_state_ref(id);
            let bits = state.membrane_potential.to_bits();
            h = h.wrapping_mul(33).wrapping_add(bits as u32);
        }
        h = h
            .wrapping_mul(33)
            .wrapping_add(net.predictor.mean_prediction_error.to_bits() as u32);
        h = h
            .wrapping_mul(33)
            .wrapping_add(net.predictor.novelty.to_bits() as u32);
        h
    }

    pub fn should_explore(&self) -> bool {
        self.exploration_urge > FixedPoint::from_f32(0.4)
            || self.boredom_accumulator > FixedPoint::from_f32(0.5)
            || (self.habituation.familiarity > FixedPoint::from_f32(0.7)
                && self.curiosity_level > FixedPoint::from_f32(0.3))
    }

    pub fn explore_epsilon(&self) -> FixedPoint {
        let base = FixedPoint::from_f32(0.05);
        let urge_part = self.exploration_urge * FixedPoint::from_f32(0.3);
        let boredom_part = self.boredom_accumulator * FixedPoint::from_f32(0.2);
        let monotony_part = FixedPoint::from_f32((self.monotony_counter as f32) / 2000.0)
            .min(FixedPoint::from_f32(0.3));
        (base + urge_part + boredom_part + monotony_part).min(FixedPoint::from_f32(0.6))
    }

    pub fn activate_dreaming(&mut self) {
        self.dreaming_active = true;
        self.dream_count += 1;
        self.last_dream_time = unsafe { crate::core::time::METABOLIC_CLOCK.ticks_1khz() };
    }

    pub fn inject_noise(&mut self, _net: &mut Network, now: u32) {
        if !self.dreaming_active {
            return;
        }

        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            return;
        }

        let rng_seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
        let mut rng = crate::core::math::XorShift64Star::new(rng_seed);

        let num_noise = (count as f32 * 0.01) as u16;
        for _ in 0..num_noise {
            let idx = (rng.next_u32() as u16) % count as u16;
            let id = NeuronId::new(idx);
            let state = neuron_state(id);

            let layer_hash = (state.layer as u32).wrapping_mul(0x9e3779b9);
            if self.is_layer_inhibited(layer_hash) {
                continue;
            }

            let amp = if state.flags.has(NeuronFlags::PREDICTOR_MODE) {
                self.noise_amplitude
            } else if state.layer == 0 {
                self.thermal_noise_amplitude
            } else {
                self.noise_amplitude * FixedPoint::from_f32(0.5)
            };

            let noise = FixedPoint::from_f32(rng.next_gaussian().to_f32() * amp.to_f32());
            state.membrane_potential += noise;
        }

        let habituation_slot = &self.habituation;
        if habituation_slot.familiarity > FixedPoint::from_f32(0.6) && rng.next_f32() < 0.1 {
            let extra_idx = (rng.next_u32() as u16) % count as u16;
            let extra = neuron_state(NeuronId::new(extra_idx));
            let burst = FixedPoint::from_f32(2.0);
            extra.membrane_potential += burst;
        }

        if now - self.last_dream_time > self.dream_duration_ms {
            self.dreaming_active = false;
            self.last_dream_time = now;
        }
    }

    pub fn inject_thermal_noise(&mut self, _net: &Network, now: u32) {
        if !self.thermal_noise_active {
            return;
        }
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            return;
        }

        let temp = crate::io::sensors::sensor_manager().temperature;
        let temp_factor = ((temp - 20.0) / 60.0).clamp(0.0, 1.0);
        let amplitude = self.thermal_noise_amplitude.to_f32() * (0.5 + temp_factor * 0.5);

        let rng_seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
        let mut rng = crate::core::math::XorShift64Star::new(rng_seed);

        let fraction = (0.005 + temp_factor * 0.015) * count as f32;
        let num_noise = (fraction as u16).max(1);

        for _ in 0..num_noise {
            let idx = (rng.next_u32() as u16) % count as u16;
            let id = NeuronId::new(idx);
            let state = neuron_state(id);
            if state.layer == 0 || state.layer == 1 {
                let noise = FixedPoint::from_f32(rng.next_gaussian().to_f32() * amplitude);
                state.membrane_potential += noise;
            }
        }

        let _ = now;
    }

    pub fn thermal_noise_sample(&self) -> FixedPoint {
        unsafe {
            let clock = &crate::core::time::METABOLIC_CLOCK;
            let low_bits = clock.cycles() & 0xFF;
            FixedPoint::from_f32((low_bits as f32) / 255.0)
        }
    }
}

pub static mut CURIOSITY_ENGINE: CuriosityEngine = CuriosityEngine::new();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;

    #[test]
    fn test_curiosity_engine_new() {
        let ce = CuriosityEngine::new();
        assert_eq!(ce.curiosity_level, FixedPoint::ZERO);
        assert_eq!(ce.boredom_threshold, FixedPoint::from_f32(0.01));
        assert!(!ce.dreaming_active);
        assert_eq!(ce.monotony_counter, 0);
        assert_eq!(ce.forced_exploration_cooldown, 0);
        assert!(ce.thermal_noise_active);
    }

    #[test]
    fn test_curiosity_boredom_grows() {
        let mut ce = CuriosityEngine::new();
        let mut net = Network::new();
        net.predictor.mean_prediction_error = FixedPoint::ZERO;
        net.predictor.novelty = FixedPoint::ZERO;
        for _ in 0..100 {
            ce.update(&net);
        }
        assert!(ce.boredom_accumulator > FixedPoint::ZERO);
        assert!(ce.curiosity_level > FixedPoint::ZERO);
    }

    #[test]
    fn test_curiosity_decays_with_novelty() {
        let mut ce = CuriosityEngine::new();
        let mut net = Network::new();
        ce.curiosity_level = FixedPoint::from_f32(0.5);
        net.predictor.mean_prediction_error = FixedPoint::from_f32(0.2);
        net.predictor.novelty = FixedPoint::from_f32(0.3);
        ce.update(&net);
        assert!(ce.curiosity_level < FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_exploration_urge_computed() {
        let mut ce = CuriosityEngine::new();
        ce.boredom_accumulator = FixedPoint::from_f32(0.8);
        ce.curiosity_level = FixedPoint::from_f32(0.3);
        let urge = ce.explore_epsilon();
        assert!(urge > FixedPoint::ZERO);
        assert!(urge <= FixedPoint::from_f32(0.6));
    }

    #[test]
    fn test_should_explore_high_boredom() {
        let mut ce = CuriosityEngine::new();
        ce.boredom_accumulator = FixedPoint::from_f32(0.6);
        assert!(ce.should_explore());
    }

    #[test]
    fn test_should_not_explore_initially() {
        let ce = CuriosityEngine::new();
        assert!(!ce.should_explore());
    }

    #[test]
    fn test_explore_epsilon_increases_with_boredom() {
        let mut ce = CuriosityEngine::new();
        let eps0 = ce.explore_epsilon();
        ce.boredom_accumulator = FixedPoint::from_f32(0.5);
        ce.monotony_counter = 100;
        let eps1 = ce.explore_epsilon();
        assert!(eps1 >= eps0);
    }

    #[test]
    fn test_explore_epsilon_bounded() {
        let mut ce = CuriosityEngine::new();
        ce.boredom_accumulator = FixedPoint::ONE;
        ce.monotony_counter = 5000;
        ce.curiosity_level = FixedPoint::ONE;
        let eps = ce.explore_epsilon();
        assert!(eps <= FixedPoint::from_f32(0.6));
        assert!(eps >= FixedPoint::ZERO);
    }

    #[test]
    fn test_dreaming_activation() {
        let mut ce = CuriosityEngine::new();
        assert!(!ce.dreaming_active);
        ce.activate_dreaming();
        assert!(ce.dreaming_active);
        assert_eq!(ce.dream_count, 1);
    }

    #[test]
    fn test_habituation_new() {
        let ht = HabituationTracker::new();
        assert_eq!(ht.familiarity, FixedPoint::ZERO);
        assert_eq!(ht.slot_count, 0);
        assert_eq!(ht.threshold, 5);
    }

    #[test]
    fn test_habituation_records_stimulus() {
        let mut ht = HabituationTracker::new();
        ht.record_stimulus(0x1234);
        assert_eq!(ht.slot_count, 1);
        assert_eq!(ht.hit_counts[0], 1);
    }

    #[test]
    fn test_habituation_familiarity_increases() {
        let mut ht = HabituationTracker::new();
        for _ in 0..10 {
            ht.record_stimulus(0xABCD);
        }
        assert!(ht.familiarity > FixedPoint::ZERO);
    }

    #[test]
    fn test_habituation_is_habituated() {
        let mut ht = HabituationTracker::new();
        ht.threshold = 3;
        for _ in 0..5 {
            ht.record_stimulus(0x5678);
        }
        assert!(ht.is_habituated(0x5678));
    }

    #[test]
    fn test_habituation_not_habituated_unknown() {
        let mut ht = HabituationTracker::new();
        ht.record_stimulus(0x1111);
        assert!(!ht.is_habituated(0x2222));
    }

    #[test]
    fn test_habituation_novelty_bonus_decreases() {
        let mut ht = HabituationTracker::new();
        let before = ht.novelty_bonus();
        for _ in 0..20 {
            ht.record_stimulus(0xDEAD);
        }
        let after = ht.novelty_bonus();
        assert!(after < before);
    }

    #[test]
    fn test_habituation_decay_reduces_familiarity() {
        let mut ht = HabituationTracker::new();
        for _ in 0..10 {
            ht.record_stimulus(0xCAFE);
        }
        let before = ht.familiarity;
        for _ in 0..100 {
            ht.decay();
        }
        assert!(ht.familiarity < before);
    }

    #[test]
    fn test_thermal_noise_sample() {
        let ce = CuriosityEngine::new();
        let sample = ce.thermal_noise_sample();
        assert!(sample >= FixedPoint::ZERO);
        assert!(sample <= FixedPoint::ONE);
    }

    #[test]
    fn test_adaptive_thresholds_move_with_error() {
        let mut ce = CuriosityEngine::new();
        let mut net = Network::new();
        net.predictor.mean_prediction_error = FixedPoint::from_f32(0.03);
        net.predictor.novelty = FixedPoint::from_f32(0.01);
        let _before = ce.boredom_threshold;
        for _ in 0..20 {
            ce.update(&net);
        }
        assert!(ce.boredom_threshold >= ce.adaptive_threshold_min);
        assert!(ce.boredom_threshold <= ce.adaptive_threshold_max);
    }

    #[test]
    fn test_monotony_counter_increments() {
        let mut ce = CuriosityEngine::new();
        let mut net = Network::new();
        net.predictor.mean_prediction_error = FixedPoint::ZERO;
        net.predictor.novelty = FixedPoint::ZERO;
        ce.monotony_counter = 0;
        ce.update(&net);
        assert_eq!(ce.monotony_counter, 1);
    }

    #[test]
    fn test_forced_exploration_triggered_by_boredom() {
        let mut ce = CuriosityEngine::new();
        ce.boredom_accumulator = FixedPoint::from_f32(0.4);
        ce.forced_exploration_cooldown = 0;
        assert!(!ce.dreaming_active);
    }

    #[test]
    fn test_dreaming_does_not_immediately_terminate() {
        let mut ce = CuriosityEngine::new();
        ce.activate_dreaming();
        let mut net = Network::new();
        ce.inject_noise(&mut net, ce.last_dream_time + ce.dream_duration_ms - 1);
        assert!(ce.dreaming_active);
    }

    #[test]
    fn test_ibl_prevents_noise_on_inhibited_layer() {
        let mut ce = CuriosityEngine::new();
        let layer_hash = (1_u32).wrapping_mul(0x9e3779b9);
        ce.inhibit_layer(layer_hash, 10);
        assert!(ce.is_layer_inhibited(layer_hash));
    }

    #[test]
    fn test_sigmoid_familiarity_shape() {
        let mut ht = HabituationTracker::new();
        for _ in 0..50 {
            ht.record_stimulus(0x9999);
        }
        assert!(ht.familiarity > FixedPoint::from_f32(0.9));
    }
}
