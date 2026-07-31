//! Synapse storage with R-STDP plasticity, eligibility traces, and global
//! reward-modulated learning. Supports homeostatic scaling, pruning of silent
//! synapses, and neuromodulator-gated plasticity.

use crate::core::math::{FixedPoint, Weight, XorShift64Star};
use crate::core::memory::{MAX_SYNAPSES, NeuronId, SynapseId};
use crate::snn::neuron::Neuromodulators;
use core::sync::atomic::{AtomicU32, Ordering};

/// Synapse with R-STDP plasticity (Section 5.3)
/// dw/dt = R * (A+ * exp(-dt/tau+) - A- * exp(dt/tau-))
/// where R is global reward/modulation signal

#[derive(Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct Synapse {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub weight: Weight,
    pub delay: u8, // Axonal conduction delay (1-255 steps)
    pub plasticity_enabled: bool,
    // STDP state
    pub pre_trace: FixedPoint,  // Pre-synaptic eligibility trace
    pub post_trace: FixedPoint, // Post-synaptic eligibility trace
    // Plasticity parameters
    pub a_plus: FixedPoint,    // LTP amplitude
    pub a_minus: FixedPoint,   // LTD amplitude
    pub tau_plus: FixedPoint,  // LTP time constant
    pub tau_minus: FixedPoint, // LTD time constant
    // R-STDP modulation
    pub reward_sensitivity: FixedPoint,
    // Structural plasticity
    pub age: u32,         // Synapse age for pruning
    pub last_active: u32, // Last time pre or post spiked
    // Pruning
    pub silence_count: u16, // Consecutive inactive cycles
}

impl Synapse {
    pub fn new(pre: NeuronId, post: NeuronId, weight: Weight, delay: u8) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
            plasticity_enabled: true,
            pre_trace: FixedPoint::ZERO,
            post_trace: FixedPoint::ZERO,
            a_plus: FixedPoint::from_f32(0.01),
            a_minus: FixedPoint::from_f32(0.012),
            tau_plus: FixedPoint::from_f32(20.0),
            tau_minus: FixedPoint::from_f32(20.0),
            reward_sensitivity: FixedPoint::ONE,
            age: 0,
            last_active: 0,
            silence_count: 0,
        }
    }

    /// Called when pre-synaptic neuron spikes
    #[inline(always)]
    pub fn on_pre_spike(&mut self, time: u32, reward_signal: FixedPoint) {
        self.last_active = time;
        self.silence_count = 0;

        if !self.plasticity_enabled {
            return;
        }

        // Update pre-synaptic trace: trace = trace * exp(-dt/tau+) + 1
        let dt = (time - self.pre_trace_time(time)) as i32;
        if dt > 0 {
            let decay = FixedPoint::exp(-FixedPoint::from_int(dt) / self.tau_plus);
            self.pre_trace = self.pre_trace * decay;
        }
        self.pre_trace += FixedPoint::ONE;

        // LTD: post_trace * A- * R
        if self.post_trace > FixedPoint::ZERO {
            let dw = self.post_trace * self.a_minus * reward_signal * self.reward_sensitivity;
            self.weight = self.weight.saturating_sub(Weight::from_f32(dw.to_f32()));
        }

        // Clamp weight
        self.clamp_weight();
    }

    /// Called when post-synaptic neuron spikes
    #[inline(always)]
    pub fn on_post_spike(&mut self, time: u32, reward_signal: FixedPoint) {
        self.last_active = time;
        self.silence_count = 0;

        if !self.plasticity_enabled {
            return;
        }

        // Update post-synaptic trace
        let dt = (time - self.post_trace_time(time)) as i32;
        if dt > 0 {
            let decay = FixedPoint::exp(-FixedPoint::from_int(dt) / self.tau_minus);
            self.post_trace = self.post_trace * decay;
        }
        self.post_trace += FixedPoint::ONE;

        // LTP: pre_trace * A+ * R
        if self.pre_trace > FixedPoint::ZERO {
            let dw = self.pre_trace * self.a_plus * reward_signal * self.reward_sensitivity;
            self.weight = self.weight.saturating_add(Weight::from_f32(dw.to_f32()));
        }

        self.clamp_weight();
    }

    /// Decay traces each timestep (call from network step)
    #[inline(always)]
    pub fn decay_traces(&mut self) {
        let decay_plus = FixedPoint::ONE - FixedPoint::ONE / self.tau_plus;
        let decay_minus = FixedPoint::ONE - FixedPoint::ONE / self.tau_minus;
        self.pre_trace *= decay_plus;
        self.post_trace *= decay_minus;

        // Increment silence counter
        self.silence_count = self.silence_count.saturating_add(1);
    }

    /// Get weight as fixed point for computation
    #[inline(always)]
    pub fn weight_fixed(&self) -> FixedPoint {
        self.weight.to_fixed()
    }

    #[inline(always)]
    fn clamp_weight(&mut self) {
        const MAX_W: i16 = 30000; // ~117 in Q8.8
        const MIN_W: i16 = -30000;
        self.weight.0 = self.weight.0.clamp(MIN_W, MAX_W);
    }

    #[inline(always)]
    fn pre_trace_time(&self, _time: u32) -> u32 {
        // Simplified - would track actual trace update time
        self.last_active
    }

    #[inline(always)]
    fn post_trace_time(&self, _time: u32) -> u32 {
        self.last_active
    }

    /// Check if synapse should be pruned (Section 12)
    pub fn should_prune(&self, min_weight: Weight, max_silence: u16) -> bool {
        !self.plasticity_enabled
            || (self.weight.0 < min_weight.0 && self.silence_count > max_silence)
    }
}

const UNINIT_SYNAPSE: core::mem::MaybeUninit<Synapse> = core::mem::MaybeUninit::uninit();
pub static mut SYNAPSE_ARRAY: [core::mem::MaybeUninit<Synapse>; crate::MAX_SYNAPSES] =
    [UNINIT_SYNAPSE; crate::MAX_SYNAPSES];
pub static SYNAPSE_COUNT: AtomicU32 = AtomicU32::new(0);

/// Get synapse by ID
#[inline(always)]
pub fn synapse(id: SynapseId) -> &'static mut Synapse {
    unsafe { &mut *SYNAPSE_ARRAY[id.index()].as_mut_ptr() }
}

#[inline(always)]
pub fn synapse_ref(id: SynapseId) -> &'static Synapse {
    unsafe { &*SYNAPSE_ARRAY[id.index()].as_ptr() }
}

/// Create new synapse (neurogenesis).
/// Tries the free pool first (from pruned synapses), then allocates a new ID.
pub fn create_synapse(
    pre: NeuronId,
    post: NeuronId,
    weight: Weight,
    delay: u8,
) -> Option<SynapseId> {
    // Try free pool first
    if let Some(id) = crate::snn::neurogenesis::free_pool_pop() {
        let s = synapse(id);
        *s = Synapse::new(pre, post, weight, delay);
        return Some(id);
    }

    let count = SYNAPSE_COUNT.fetch_add(1, Ordering::AcqRel);
    if count >= MAX_SYNAPSES as u32 {
        return None;
    }

    let id = SynapseId::new(count as u16);
    let s = synapse(id);
    *s = Synapse::new(pre, post, weight, delay);
    Some(id)
}

/// Initialize synapse population with sparse random connectivity (Section 25)
pub fn init_connectivity(rng: &mut XorShift64Star, sparsity: f32) {
    use crate::core::memory::{NEURON_COUNT, neuron_state_ref};

    let neuron_count = NEURON_COUNT.load(Ordering::Relaxed) as u16;
    if neuron_count == 0 {
        return;
    }

    // Connection probability per layer pair
    let layer_probs = [
        // L0:Sensory  L1:Inhib  L2:Adapt  L3:Pred  L4:Motor  L5:Pace  L6:Reflex L7:Curio
        [0.02, 0.10, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00], // L0 -> *
        [0.15, 0.05, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00], // L1 -> *
        [0.05, 0.10, 0.08, 0.03, 0.02, 0.00, 0.00, 0.05], // L2 -> *
        [0.01, 0.00, 0.03, 0.05, 0.02, 0.00, 0.00, 0.10], // L3 -> *
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // L4 -> *
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], // L5 -> *
        [0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00], // L6 -> *
        [0.00, 0.05, 0.05, 0.20, 0.00, 0.00, 0.00, 0.05], // L7 -> *
    ];

    for pre_id in 0..neuron_count {
        let pre = NeuronId::new(pre_id);
        let pre_state = neuron_state_ref(pre);

        for post_id in 0..neuron_count {
            let post = NeuronId::new(post_id);
            if pre == post {
                continue;
            }

            let post_state = neuron_state_ref(post);
            let prob = layer_probs[pre_state.layer as usize][post_state.layer as usize];

            if rng.next_f32() < prob * sparsity {
                // Distance-dependent delay
                let delay = if pre_state.layer == post_state.layer {
                    1
                } else {
                    2
                };
                let weight = Weight::from_f32(rng.next_gaussian().to_f32() * 0.1);
                let _ = create_synapse(pre, post, weight, delay);
            }
        }
    }

    // Add reflex arcs (Section 19) - hard-coded, no plasticity
    init_reflex_arcs();
}

/// Hard-coded reflex arcs (Section 19)
/// Maps sensor neurons (L0) to reflex neurons (L6) to actuator neurons (L4)
fn init_reflex_arcs() {
    use crate::core::memory::{NEURON_COUNT, neuron_state_ref};

    let count = NEURON_COUNT.load(Ordering::Relaxed) as u16;
    let mut reflex_ids = [NeuronId::INVALID; 16];
    let mut motor_ids = [NeuronId::INVALID; 16];
    let mut reflex_count = 0u8;
    let mut motor_count = 0u8;
    for i in 0..count {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        match state.layer {
            6 => {
                // Layer 6 = reflex
                if reflex_count < 16 {
                    reflex_ids[reflex_count as usize] = id;
                    reflex_count += 1;
                }
            }
            4 => {
                // Layer 4 = motor/actuator
                if motor_count < 16 {
                    motor_ids[motor_count as usize] = id;
                    motor_count += 1;
                }
            }
            _ => {}
        }
    }
    // Connect reflex → motor with fixed weights, no plasticity
    for i in 0..reflex_count.min(motor_count) {
        let r_id = reflex_ids[i as usize];
        let m_id = motor_ids[i as usize];
        if r_id != NeuronId::INVALID && m_id != NeuronId::INVALID {
            if let Some(syn_id) = create_synapse(r_id, m_id, Weight::from_f32(0.8), 1) {
                let s = synapse(syn_id);
                s.plasticity_enabled = false;
            }
        }
    }
    // Connect first sensor (L0) → first reflex
    for i in 0..count {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        if state.layer == 0 && reflex_count > 0 {
            if let Some(syn_id) = create_synapse(id, reflex_ids[0], Weight::from_f32(0.5), 1) {
                let s = synapse(syn_id);
                s.plasticity_enabled = false;
            }
            break;
        }
    }
}

/// Get all synapses from a neuron
pub fn outgoing_synapses(pre: NeuronId) -> impl Iterator<Item = SynapseId> {
    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    (0..count as u16).filter_map(move |i| {
        let id = SynapseId::new(i);
        let s = synapse_ref(id);
        if s.pre == pre { Some(id) } else { None }
    })
}

/// Get all synapses to a neuron
pub fn incoming_synapses(post: NeuronId) -> impl Iterator<Item = SynapseId> {
    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    (0..count as u16).filter_map(move |i| {
        let id = SynapseId::new(i);
        let s = synapse_ref(id);
        if s.post == post { Some(id) } else { None }
    })
}

/// R-STDP global reward signal computation
pub fn compute_reward_signal(
    prediction_error: FixedPoint,
    novelty: FixedPoint,
    homeostasis_error: FixedPoint,
) -> FixedPoint {
    // R = base + novelty - |prediction_error| - |homeostasis_error|
    let base = FixedPoint::from_f32(0.1); // Baseline learning
    let reward = base + novelty - prediction_error.abs() - homeostasis_error.abs();
    reward.clamp(FixedPoint::ZERO, FixedPoint::ONE)
}

/// Global plasticity modulation (Section 21)
pub fn modulate_plasticity(nm: &Neuromodulators) -> (FixedPoint, FixedPoint) {
    // Noradrenaline: increases both LTP and LTD (crisis learning)
    // Serotonin: decreases plasticity (consolidation)
    // Dopamine: gates LTP specifically (reward learning)
    // Acetylcholine: increases LTP, decreases LTD (attention)

    let ltp_mult = FixedPoint::ONE + nm.noradrenaline * FixedPoint::from_f32(2.0)
        - nm.serotonin * FixedPoint::from_f32(0.5)
        + nm.dopamine * FixedPoint::from_f32(1.0)
        + nm.acetylcholine * FixedPoint::from_f32(0.5);

    let ltd_mult = FixedPoint::ONE + nm.noradrenaline * FixedPoint::from_f32(2.0)
        - nm.serotonin * FixedPoint::from_f32(0.5)
        - nm.dopamine * FixedPoint::from_f32(0.5)
        - nm.acetylcholine * FixedPoint::from_f32(0.3);

    (
        ltp_mult.clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(5.0)),
        ltd_mult.clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(5.0)),
    )
}

/// Apply global plasticity modulation to all synapses.
/// Sets a_plus/a_minus from base rate × multiplier (no compounding).
pub fn apply_plasticity_modulation(ltp_mult: FixedPoint, ltd_mult: FixedPoint) {
    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse(id);
        if s.plasticity_enabled {
            s.a_plus = (FixedPoint::from_f32(0.01) * ltp_mult)
                .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
            s.a_minus = (FixedPoint::from_f32(0.012) * ltd_mult)
                .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
        }
    }
}

/// Prune silent synapses (Section 12 - Neurogenesis/Recycling)
pub fn prune_silent_synapses(min_weight: Weight, max_silence: u16) -> usize {
    let mut pruned = 0;
    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);

    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse(id);
        if s.should_prune(min_weight, max_silence) {
            // Mark for recycling (weight = 0, plasticity disabled)
            s.weight = Weight::ZERO;
            s.plasticity_enabled = false;
            s.pre = NeuronId::INVALID;
            s.post = NeuronId::INVALID;
            pruned += 1;
        }
    }
    pruned
}

/// Apply age-related synaptic senescence (Section 12 - Synaptic Aging)
/// Weight decays to 50% at max_age, plasticity to 20%.
/// Synapses exceeding max_age are pruned.
/// Returns number of synapses pruned due to old age.
pub fn apply_senescence(max_age: u32) -> usize {
    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    let mut pruned = 0;

    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse(id);
        if !s.plasticity_enabled {
            continue;
        }

        let age = s.age.saturating_add(1);
        s.age = age;

        if age >= max_age {
            s.weight = Weight::ZERO;
            s.plasticity_enabled = false;
            s.pre = NeuronId::INVALID;
            s.post = NeuronId::INVALID;
            pruned += 1;
        } else {
            // Gradual decay: weight → 50%, plasticity → 20% at max_age
            let progress = FixedPoint::from_f32(age as f32 / max_age.max(1) as f32);
            let weight_keep = FixedPoint::ONE - progress * FixedPoint::from_f32(0.5);
            let plast_keep = FixedPoint::ONE - progress * FixedPoint::from_f32(0.8);
            s.weight = Weight::from_f32((s.weight.to_fixed() * weight_keep).to_f32());
            s.a_plus = (s.a_plus * plast_keep)
                .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
            s.a_minus = (s.a_minus * plast_keep)
                .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
        }
    }
    pruned
}

/// Homeostatic synaptic scaling (Section 20)
pub fn homeostatic_scaling(target_rate: FixedPoint, actual_rate: FixedPoint) {
    let scale = target_rate / actual_rate.max(FixedPoint::from_f32(0.001));
    let scale = scale.clamp(FixedPoint::from_f32(0.5), FixedPoint::from_f32(2.0));

    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse(id);
        if s.plasticity_enabled && s.weight.0 != 0 {
            let new_w = (s.weight.to_fixed() * scale).to_f32();
            s.weight = Weight::from_f32(new_w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdp_ltp() {
        let mut s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::ZERO, 1);
        let reward = FixedPoint::ONE;

        // Pre spikes
        s.on_pre_spike(10, reward);
        // Post spikes shortly after (causal)
        s.on_post_spike(12, reward);

        // Should have potentiated
        assert!(s.weight.0 > 0);
    }

    #[test]
    fn stdp_ltd() {
        let mut s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        let reward = FixedPoint::ONE;

        // Post spikes first
        s.on_post_spike(10, reward);
        // Pre spikes after (anti-causal)
        s.on_pre_spike(12, reward);

        // Should have depressed
        assert!(s.weight.to_f32() < 0.5);
    }

    #[test]
    fn reward_modulation() {
        let mut s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::ZERO, 1);

        // No reward - no plasticity
        s.on_pre_spike(10, FixedPoint::ZERO);
        s.on_post_spike(12, FixedPoint::ZERO);
        assert_eq!(s.weight.0, 0);

        // With reward - plasticity occurs
        s.on_pre_spike(20, FixedPoint::ONE);
        s.on_post_spike(22, FixedPoint::ONE);
        assert!(s.weight.0 > 0);
    }

    #[test]
    fn apply_senescence_increments_age() {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        let id = SynapseId::new(count as u16);
        let s = synapse(id);
        *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        SYNAPSE_COUNT.fetch_add(1, Ordering::SeqCst);

        assert_eq!(s.age, 0);
        apply_senescence(1_000_000);
        assert_eq!(s.age, 1);
        assert!(s.plasticity_enabled);

        SYNAPSE_COUNT.store(count, Ordering::SeqCst);
    }

    #[test]
    fn apply_senescence_reduces_weight() {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        let id = SynapseId::new(count as u16);
        let s = synapse(id);
        *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        SYNAPSE_COUNT.fetch_add(1, Ordering::SeqCst);

        s.age = 500_000; // Simulate middle age
        apply_senescence(1_000_000);
        // After 50% of lifespan: weight decays by ~25% (50% * 0.5)
        assert!(s.weight.to_f32() < 0.45, "Weight should decay with age");

        SYNAPSE_COUNT.store(count, Ordering::SeqCst);
    }

    #[test]
    fn apply_senescence_prunes_old_synapses() {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        let id = SynapseId::new(count as u16);
        let s = synapse(id);
        *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        SYNAPSE_COUNT.fetch_add(1, Ordering::SeqCst);

        s.age = 100;
        let pruned = apply_senescence(50);
        assert_eq!(pruned, 1);
        assert!(!s.plasticity_enabled);
        assert_eq!(s.pre, NeuronId::INVALID);

        SYNAPSE_COUNT.store(count, Ordering::SeqCst);
    }

    #[test]
    fn apply_senescence_skips_disabled() {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        let id = SynapseId::new(count as u16);
        let s = synapse(id);
        *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        s.plasticity_enabled = false;
        SYNAPSE_COUNT.fetch_add(1, Ordering::SeqCst);

        s.age = 100;
        let pruned = apply_senescence(50);
        assert_eq!(pruned, 0);

        SYNAPSE_COUNT.store(count, Ordering::SeqCst);
    }

    #[test]
    fn apply_senescence_reduces_plasticity() {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        let id = SynapseId::new(count as u16);
        let s = synapse(id);
        *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 1);
        SYNAPSE_COUNT.fetch_add(1, Ordering::SeqCst);

        let a0 = s.a_plus;
        s.age = 800_000;
        apply_senescence(1_000_000);
        // After 80% of lifespan: plasticity decays by ~64% (80% * 0.8)
        assert!(s.a_plus < a0, "Plasticity should decay with age");
        assert!(s.a_minus < a0, "Plasticity should decay with age");

        SYNAPSE_COUNT.store(count, Ordering::SeqCst);
    }
}
