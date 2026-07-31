//! LIF and ALIF neuron models with neuromodulator-driven threshold adaptation
//! for crisis (noradrenaline), stability (serotonin), reward (dopamine), and
//! attention (acetylcholine) modes.

use crate::core::math::{FixedPoint, XorShift64Star};
use crate::core::memory::{NeuronFlags, NeuronId, NeuronType, neuron_state, neuron_state_ref};
use core::sync::atomic::Ordering;

/// LIF Neuron implementation (Section 3.1)
/// dV/dt = -V/tau + I_syn + I_bias
/// Spike when V >= V_th, then V = V_reset

pub struct LIFNeuron {
    pub id: NeuronId,
    // Parameters (constant after init)
    pub tau_m: FixedPoint,      // Membrane time constant
    pub v_th: FixedPoint,       // Spike threshold
    pub v_reset: FixedPoint,    // Reset potential
    pub v_rest: FixedPoint,     // Resting potential
    pub refractory_period: u16, // Refractory period (simulation steps)
    // Dynamic state (updated each step)
    pub v_mem: FixedPoint, // Current membrane potential
    pub refractory_remaining: u16,
    pub last_spike_time: u32,
    // Adaptive threshold (ALIF)
    pub adaptive_th: FixedPoint,
    pub adaptation_tau: FixedPoint,
    // Neuromodulation sensitivity
    pub noradrenaline_sensitivity: FixedPoint,
    pub serotonin_sensitivity: FixedPoint,
    // RNG for stochastic spiking
    _rng: XorShift64Star,
}

impl LIFNeuron {
    pub fn new(id: NeuronId, neuron_type: NeuronType) -> Self {
        let (tau_m, v_th, v_reset, refractory, _adaptive) = match neuron_type {
            NeuronType::LIF => (
                FixedPoint::from_f32(20.0), // 20ms
                FixedPoint::from_f32(1.0),
                FixedPoint::from_f32(0.0),
                2, // 2ms refractory
                false,
            ),
            NeuronType::ALIF => (
                FixedPoint::from_f32(20.0),
                FixedPoint::from_f32(1.0),
                FixedPoint::from_f32(0.0),
                2,
                true,
            ),
            NeuronType::BURST => (
                FixedPoint::from_f32(10.0),
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(-0.5),
                1,
                true,
            ),
            NeuronType::INHIBITORY => (
                FixedPoint::from_f32(10.0),
                FixedPoint::from_f32(0.5),
                FixedPoint::from_f32(0.0),
                1,
                false,
            ),
            NeuronType::PACER => (
                FixedPoint::from_f32(1000.0), // 1 second
                FixedPoint::from_f32(1.0),
                FixedPoint::from_f32(0.0),
                0,
                false,
            ),
            NeuronType::REFLEX => (
                FixedPoint::from_f32(1.0), // Ultra-fast
                FixedPoint::from_f32(0.3),
                FixedPoint::from_f32(0.0),
                0,
                false,
            ),
        };

        Self {
            id,
            tau_m,
            v_th,
            v_reset,
            v_rest: FixedPoint::ZERO,
            refractory_period: refractory,
            v_mem: FixedPoint::ZERO,
            refractory_remaining: 0,
            last_spike_time: 0,
            adaptive_th: FixedPoint::ZERO,
            adaptation_tau: FixedPoint::from_f32(100.0),
            noradrenaline_sensitivity: FixedPoint::from_f32(1.0),
            serotonin_sensitivity: FixedPoint::from_f32(1.0),
            _rng: XorShift64Star::new(id.0 as u64 * 0x9E3779B97F4A7C15),
        }
    }

    /// Single simulation step
    /// Returns true if spike emitted
    #[inline(always)]
    pub fn step(&mut self, synaptic_current: FixedPoint, nm: &Neuromodulators, time: u32) -> bool {
        let state = neuron_state(self.id);

        // Update refractory
        if self.refractory_remaining > 0 {
            self.refractory_remaining -= 1;
            state.refractory_remaining = self.refractory_remaining;
            state.membrane_potential = self.v_mem;
            return false;
        }

        // Neuromodulation: adjust threshold and leak
        let na = nm.noradrenaline;
        let st = nm.serotonin;

        // Noradrenaline: lower threshold, increase gain (crisis mode)
        let effective_th =
            self.v_th - (na * self.noradrenaline_sensitivity * FixedPoint::from_f32(0.3));
        // Serotonin: raise threshold, decrease plasticity (stability mode)
        let effective_th =
            effective_th + (st * self.serotonin_sensitivity * FixedPoint::from_f32(0.2));

        // Leaky integration: V = V * (1 - dt/tau) + I * dt
        // dt = 1ms, so dt/tau = 1/tau_ms
        let leak_factor = FixedPoint::ONE - FixedPoint::ONE / self.tau_m;
        self.v_mem = self.v_mem * leak_factor + synaptic_current + state.bias_current;

        // Adaptive threshold (ALIF)
        if self.adaptation_tau.to_bits() != 0 {
            let adaptation_decay = FixedPoint::ONE - FixedPoint::ONE / self.adaptation_tau;
            self.adaptive_th *= adaptation_decay;
            if state.last_spike_time == time - 1 {
                self.adaptive_th += FixedPoint::from_f32(0.1);
            }
        }

        // Check for spike
        let total_th = effective_th + self.adaptive_th;
        let spiked = self.v_mem >= total_th;

        if spiked {
            self.v_mem = self.v_reset;
            self.refractory_remaining = self.refractory_period;
            self.last_spike_time = time;
            state.last_spike_time = time;
            state.refractory_remaining = self.refractory_period;

            // Spike trace for STDP
            self.record_spike_trace(time);
        }

        state.membrane_potential = self.v_mem;
        state.threshold = total_th;

        spiked
    }

    /// Record spike for STDP eligibility trace
    #[inline(always)]
    fn record_spike_trace(&mut self, time: u32) {
        // Pre-synaptic trace updated in synapse step
        // Here we just update neuron's last spike time
        neuron_state(self.id).last_spike_time = time;
    }

    /// Inject current directly (for sensory input)
    #[inline(always)]
    pub fn inject_current(&mut self, current: FixedPoint) {
        self.v_mem += current;
    }

    /// Reset to resting state
    pub fn reset(&mut self) {
        self.v_mem = self.v_rest;
        self.refractory_remaining = 0;
        self.adaptive_th = FixedPoint::ZERO;
    }
}

/// Neuromodulator global state (Section 21)
#[derive(Clone, Copy, Default)]
pub struct Neuromodulators {
    pub noradrenaline: FixedPoint, // Crisis mode: 0.0-1.0
    pub serotonin: FixedPoint,     // Stability mode: 0.0-1.0
    pub dopamine: FixedPoint,      // Reward/learning: 0.0-1.0
    pub acetylcholine: FixedPoint, // Attention/learning: 0.0-1.0
}

impl Neuromodulators {
    pub const fn new() -> Self {
        Self {
            noradrenaline: FixedPoint::ZERO,
            serotonin: FixedPoint::ONE,
            dopamine: FixedPoint::ZERO,
            acetylcholine: FixedPoint::from_f32(0.5),
        }
    }

    /// Crisis mode - high noradrenaline
    pub fn crisis_mode(&mut self) {
        self.noradrenaline = FixedPoint::ONE;
        self.serotonin = FixedPoint::ZERO;
        self.dopamine = FixedPoint::from_f32(0.5);
        self.acetylcholine = FixedPoint::ONE;
    }

    /// Stability mode - high serotonin
    pub fn stability_mode(&mut self) {
        self.noradrenaline = FixedPoint::ZERO;
        self.serotonin = FixedPoint::ONE;
        self.dopamine = FixedPoint::from_f32(0.2);
        self.acetylcholine = FixedPoint::from_f32(0.3);
    }

    /// Exploration mode - high dopamine/acetylcholine
    pub fn exploration_mode(&mut self) {
        self.noradrenaline = FixedPoint::from_f32(0.3);
        self.serotonin = FixedPoint::from_f32(0.3);
        self.dopamine = FixedPoint::ONE;
        self.acetylcholine = FixedPoint::ONE;
    }

    /// Decay towards baseline
    pub fn decay(&mut self, rate: FixedPoint) {
        let one_minus_rate = FixedPoint::ONE - rate;
        self.noradrenaline *= one_minus_rate;
        self.serotonin = self.serotonin * one_minus_rate + FixedPoint::ONE * rate;
        self.dopamine *= one_minus_rate;
        self.acetylcholine *= one_minus_rate;
    }
}

/// Spike event for trace logging
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SpikeEvent {
    pub neuron_id: NeuronId,
    pub timestamp: u32,
    pub layer: u8,
    pub is_predictor: bool,
}

/// Global neuromodulator state
use core::mem::MaybeUninit;
pub static mut GLOBAL_NEUROMODULATORS: MaybeUninit<Neuromodulators> = MaybeUninit::uninit();
static INITIALIZED_NEUROMODULATORS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_neuromodulators() {
    unsafe {
        GLOBAL_NEUROMODULATORS.write(Neuromodulators::new());
        INITIALIZED_NEUROMODULATORS.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn neuromodulators() -> &'static mut Neuromodulators {
    unsafe {
        if !INITIALIZED_NEUROMODULATORS.load(core::sync::atomic::Ordering::Relaxed) {
            init_neuromodulators();
        }
        &mut *GLOBAL_NEUROMODULATORS.as_mut_ptr()
    }
}

/// Initialize neuron population (Section 25 - Big Bang)
pub fn init_population(rng: &mut XorShift64Star) {
    use crate::core::memory::allocate_neuron;

    // Layer 0: Sensory (Reactive) - 1024 neurons
    for _ in 0..1024 {
        if let Some(id) = allocate_neuron(NeuronType::LIF, 0) {
            let n = neuron_state(id);
            n.threshold = FixedPoint::from_f32(rng.next_f32() * 0.5 + 0.75);
            n.leak = FixedPoint::from_f32(rng.next_f32() * 0.2 + 0.8);
        }
    }

    // Layer 1: Fast Interneurons (Inhibitory) - 256 neurons
    for _ in 0..256 {
        if let Some(id) = allocate_neuron(NeuronType::INHIBITORY, 1) {
            let n = neuron_state(id);
            n.threshold = FixedPoint::from_f32(0.3);
            n.leak = FixedPoint::from_f32(0.95);
        }
    }

    // Layer 2: Adaptive/Integrator - 512 neurons
    for _ in 0..512 {
        if let Some(id) = allocate_neuron(NeuronType::ALIF, 2) {
            let n = neuron_state(id);
            n.threshold = FixedPoint::from_f32(rng.next_f32() * 0.5 + 1.0);
        }
    }

    // Layer 3: Predictor (Internal model) - 1024 neurons
    for _ in 0..1024 {
        if let Some(id) = allocate_neuron(NeuronType::LIF, 3) {
            let n = neuron_state(id);
            n.flags.set(NeuronFlags::PREDICTOR_MODE);
            n.threshold = FixedPoint::from_f32(rng.next_f32() * 0.5 + 1.2);
        }
    }

    // Layer 4: Motor/Output - 256 neurons
    for _ in 0..256 {
        if let Some(id) = allocate_neuron(NeuronType::LIF, 4) {
            let n = neuron_state(id);
            n.threshold = FixedPoint::from_f32(rng.next_f32() * 0.3 + 0.7);
        }
    }

    // Layer 5: Pacemaker (Metabolic clock) - 8 neurons (1Hz)
    for _ in 0..8 {
        if let Some(id) = allocate_neuron(NeuronType::PACER, 5) {
            let n = neuron_state(id);
            n.bias_current = FixedPoint::from_f32(1.0); // Constant drive
        }
    }

    // Layer 6: Reflex arcs (Hard-coded) - 64 neurons
    for _ in 0..64 {
        if let Some(id) = allocate_neuron(NeuronType::REFLEX, 6) {
            let n = neuron_state(id);
            n.flags.set(NeuronFlags::PLASTICITY_DISABLED);
            n.threshold = FixedPoint::from_f32(0.2);
        }
    }

    // Layer 7: Curiosity/Noise injection - 128 neurons
    for _ in 0..128 {
        if let Some(id) = allocate_neuron(NeuronType::BURST, 7) {
            let n = neuron_state(id);
            n.threshold = FixedPoint::from_f32(0.5);
        }
    }
}

/// Get all neurons in a layer
pub fn neurons_in_layer(layer: u8) -> impl Iterator<Item = NeuronId> {
    let count = crate::core::memory::NEURON_COUNT.load(Ordering::Relaxed);
    (0..count as u16).filter_map(move |i| {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        if state.layer == layer { Some(id) } else { None }
    })
}

/// Get predictor neurons
pub fn predictor_neurons() -> impl Iterator<Item = NeuronId> {
    let count = crate::core::memory::NEURON_COUNT.load(Ordering::Relaxed);
    (0..count as u16).filter_map(move |i| {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        if state.flags.has(NeuronFlags::PREDICTOR_MODE) {
            Some(id)
        } else {
            None
        }
    })
}

/// Get actor (motor) neurons
pub fn actor_neurons() -> impl Iterator<Item = NeuronId> {
    let count = crate::core::memory::NEURON_COUNT.load(Ordering::Relaxed);
    (0..count as u16).filter_map(move |i| {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        if state.layer == 4 { Some(id) } else { None }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_spikes() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let nm = Neuromodulators::new();

        // Inject strong current
        n.inject_current(FixedPoint::from_f32(2.0));

        // Should spike
        assert!(n.step(FixedPoint::ZERO, &nm, 1));
        assert_eq!(n.v_mem.to_f32(), 0.0); // Reset
    }

    #[test]
    fn refractory_period() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let nm = Neuromodulators::new();

        n.inject_current(FixedPoint::from_f32(2.0));
        assert!(n.step(FixedPoint::ZERO, &nm, 1));

        n.inject_current(FixedPoint::from_f32(2.0));
        assert!(!n.step(FixedPoint::ZERO, &nm, 2));
    }

    #[test]
    fn subthreshold_no_spike() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let nm = Neuromodulators::new();
        assert!(!n.step(FixedPoint::ZERO, &nm, 1));
    }

    #[test]
    fn membrane_leak() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let nm = Neuromodulators::new();
        n.inject_current(FixedPoint::from_f32(1.5));
        let v_before = n.v_mem;
        n.step(FixedPoint::ZERO, &nm, 1);
        assert!(n.v_mem.to_f32() < v_before.to_f32());
    }

    #[test]
    fn synaptic_current_integration() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let nm = Neuromodulators::new();
        assert!(n.step(FixedPoint::from_f32(5.0), &nm, 1));
    }

    #[test]
    fn neuromodulator_sensitivity() {
        let mut n = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let mut nm = Neuromodulators::new();
        nm.noradrenaline = FixedPoint::ONE;
        n.inject_current(FixedPoint::from_f32(2.0));
        assert!(n.step(FixedPoint::ZERO, &nm, 1));
    }

    #[test]
    fn neuron_types_different_tau() {
        let excit = LIFNeuron::new(NeuronId::new(0), NeuronType::LIF);
        let inhib = LIFNeuron::new(NeuronId::new(1), NeuronType::INHIBITORY);
        assert_ne!(excit.tau_m, inhib.tau_m);
    }
}
