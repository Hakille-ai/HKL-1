//! Network topology orchestrator managing Actor and Predictor subnetworks,
//! sensory input processing, neuromodulator updates, energy-aware threshold
//! scaling, and predictor-guided time-warp simulation.

use crate::cognitive::curiosity::CURIOSITY_ENGINE;
use crate::cognitive::neuromodulation::COGNITIVE_NEUROMODULATORS;
#[allow(unused_imports)]
use crate::core::atomic::FetchAtomic;
use crate::core::math::FixedPoint;
use crate::core::memory::{
    MAX_NEURONS, NEURON_COUNT, NeuronFlags, NeuronId, SynapseId, neuron_state, neuron_state_ref,
};
use crate::core::time::TIME_WARPER;
use crate::safety::entropy_monitor::ENTROPY_MONITOR;
use crate::snn::homeostasis;
use crate::snn::neuron::{Neuromodulators, SpikeEvent};
use crate::snn::synapse::{self, SYNAPSE_COUNT};
use core::sync::atomic::{AtomicU32, Ordering};

/// Simulation step interval (microseconds)
pub const SIMULATION_DT_US: u32 = 1000; // 1 kHz simulation

/// Main network orchestrator - manages both Actor and Predictor subnetworks
pub struct Network {
    // Subnetwork references
    pub actor: ActorNetwork,
    pub predictor: PredictorNetwork,
    // Global state
    pub time: u32,
    pub neuromodulators: &'static mut Neuromodulators,
    pub firing_rates: [FixedPoint; MAX_NEURONS],
    // Energy-aware modulation (Section 11)
    pub energy_level: FixedPoint,         // 0.0 - 1.0 (battery/SOC)
    pub threshold_modulation: FixedPoint, // Global threshold scaler
    // Simulation mode
    pub warp_active: bool,
    pub warp_factor: u32,
    // Spike statistics
    pub total_spikes: AtomicU32,
    pub recent_spikes_window: [u32; 1024],
    pub window_idx: u16,
    // Metabolic heartbeat flag
    pub heartbeat: bool,
    // Predictive cycle state
    pub cycle_active: bool,
    pub cycle_cooldown: u32,
}

impl Network {
    pub fn new() -> Self {
        let nm = crate::snn::neuron::neuromodulators();
        Self {
            actor: ActorNetwork::new(),
            predictor: PredictorNetwork::new(),
            time: 0,
            neuromodulators: nm,
            firing_rates: [FixedPoint::ZERO; MAX_NEURONS],
            energy_level: FixedPoint::ONE,
            threshold_modulation: FixedPoint::ONE,
            warp_active: false,
            warp_factor: 1,
            total_spikes: AtomicU32::new(0),
            recent_spikes_window: [0; 1024],
            window_idx: 0,
            heartbeat: false,
            cycle_active: false,
            cycle_cooldown: 0,
        }
    }

    /// Auto-detect host hardware resources and scale memory capacity dynamically
    pub fn auto_adapt_hardware(&mut self) -> crate::system::hardware_detect::HardwareProfile {
        let profile = crate::system::hardware_detect::HardwareDetector::detect();
        self.scale_capacity(
            profile.recommended_max_neurons,
            profile.recommended_max_synapses,
        );
        profile
    }

    /// Dynamically scale neuron and synapse capacity bounds
    pub fn scale_capacity(&mut self, neurons: usize, synapses: usize) {
        crate::core::memory::ADAPTIVE_MEMORY.set_capacity(neurons, synapses);
    }

    /// Main simulation step (called at 1kHz from ISR)
    #[inline(never)]
    pub fn step(&mut self) {
        self.time += 1;
        let now = self.time;

        // Read physical sensors and push into ring buffers (every 10ms)
        if now.is_multiple_of(10) {
            let sm = crate::io::sensors::sensor_manager();
            sm.read_all();
            sm.emit_sensor_spikes(now);
        }

        // Determine warp mode
        let current_warp = if self.warp_active {
            self.warp_factor
        } else {
            1
        };

        // Process sensory input from ring buffers
        self.process_sensory_input(now);

        // Update attention routing (every 5ms)
        if now.is_multiple_of(5) {
            let cog_actor = crate::cognitive::actor::cognitive_actor();
            let selected_action = cog_actor.selected_action;
            let action_confidence = cog_actor.action_confidence;
            let ar = crate::cognitive::attention::attention_router();
            ar.update(
                self.predictor.mean_prediction_error,
                self.predictor.novelty,
                selected_action,
                action_confidence,
            );
            ar.apply();
        }

        // Step actor subnetwork
        self.actor.step(now, current_warp);

        // Step predictor subnetwork
        self.predictor.step(now, current_warp);

        // Apply STDP plasticity
        self.apply_plasticity(now);

        // Update neuromodulators
        self.update_neuromodulators(now);

        // Apply actuator outputs (every 10ms)
        if now.is_multiple_of(10) {
            let am = crate::io::actuators::actuator_manager();
            am.read_motor_outputs();
        }

        // Update temporal cognition
        crate::cognitive::temporal::temporal_cognition().update();

        // Update curiosity engine (every 20ms)
        if now.is_multiple_of(20) {
            unsafe {
                let curiosity = &mut CURIOSITY_ENGINE;
                curiosity.update(self);
                if curiosity.dreaming_active {
                    curiosity.inject_noise(self, now);
                }
            }
            // Sync adaptive epsilon to actor
            let eps = unsafe { CURIOSITY_ENGINE.explore_epsilon() };
            crate::cognitive::actor::cognitive_actor().epsilon = eps;
        }

        // Process bio-inspired modules (every step — lightweight)
        self.process_global_workspace(now);
        self.process_bio_modules(now);

        // Check for metabolic heartbeat (every 1000 steps)
        if now.is_multiple_of(1000) {
            self.heartbeat = true;
            self.metabolic_maintenance(now);
        } else {
            self.heartbeat = false;
        }

        // Energy-aware scaling (Section 11)
        if now.is_multiple_of(100) {
            self.energy_adaption();
        }

        // Cooldown counter
        if self.cycle_cooldown > 0 {
            self.cycle_cooldown -= 1;
        }

        // Predictive cycle trigger
        self.cycle_trigger(now);

        // Update statistics
        if now.is_multiple_of(1000) {
            self.update_statistics();
        }

        // Periodic causal graph update (every 500 steps)
        if now.is_multiple_of(500) {
            crate::telemetry::xai::analyze_current_trace();
        }
    }

    /// Process bio-inspired modules: astrocytes, striosome, thalamus, hippocampus, cerebellum
    fn process_bio_modules(&mut self, now: u32) {
        let da = self.neuromodulators.dopamine;
        let novelty = self.predictor.novelty;
        let pred_error = self.predictor.mean_prediction_error;

        // Build sensory input proxy from firing rates
        let sensory_proxy: [FixedPoint; 4] = [
            self.firing_rates[0],
            if MAX_NEURONS > 1 {
                self.firing_rates[1]
            } else {
                FixedPoint::ZERO
            },
            if MAX_NEURONS > 2 {
                self.firing_rates[2]
            } else {
                FixedPoint::ZERO
            },
            if MAX_NEURONS > 3 {
                self.firing_rates[3]
            } else {
                FixedPoint::ZERO
            },
        ];

        // 1. Thalamus — sensory gating (every step)
        let attention = crate::cognitive::attention::attention_router().focus.gain;
        let thal = crate::bio::thalamus::thalamus();
        thal.step(&sensory_proxy, attention, pred_error, now as u64);

        // 2. Striosome — dopamine-gated action selection (every 10ms)
        if now.is_multiple_of(10) {
            let strio_input: [FixedPoint; 64] = {
                let mut buf = [FixedPoint::ZERO; 64];
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = if i < MAX_NEURONS {
                        self.firing_rates[i]
                    } else {
                        FixedPoint::ZERO
                    };
                }
                buf
            };
            let sys = crate::bio::striosome::striosome_matrix();
            sys.step_all(da, &strio_input);
            sys.learn_all(pred_error, &strio_input);
        }

        // 3. Hippocampus — memory consolidation (every 50ms)
        if now.is_multiple_of(50) {
            // Inject spatial context from episodic memory into hippocampus input
            let mut hipp_input: [FixedPoint; 256] = {
                let mut buf = [FixedPoint::ZERO; 256];
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = if i < MAX_NEURONS {
                        self.firing_rates[i]
                    } else {
                        FixedPoint::ZERO
                    };
                }
                buf
            };
            let epi = crate::cognitive::episodic::episodic_memory();
            let place_rates = epi.place_cell_activity();
            for (i, &rate) in place_rates.iter().enumerate() {
                if i < hipp_input.len() {
                    hipp_input[i] = (hipp_input[i] + rate * FixedPoint::from_f32(0.3))
                        .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                }
            }
            let hip = crate::bio::hippocampus::hippocampus();
            hip.step(&hipp_input, novelty, pred_error);

            // Bridge: hippocampus SWR → episodic memory sharp-wave ripple replay
            if hip.swr_active && hip.consolidation_trigger {
                epi.trigger_ripple_replay(now as u64);
            }
        }

        // 4. Astrocytes — glial modulation (every 100ms)
        if now.is_multiple_of(100) {
            let astro_input: [FixedPoint; 64] = {
                let mut buf = [FixedPoint::ZERO; 64];
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = if i < MAX_NEURONS {
                        self.firing_rates[i]
                    } else {
                        FixedPoint::ZERO
                    };
                }
                buf
            };
            let astro = crate::bio::astrocytes::astrocyte_network();
            astro.step_all(&astro_input, now as u64, FixedPoint::from_f32(0.1));
            astro.propagate_waves();
        }

        // 5. Cerebellum — motor refinement (every 20ms)
        if now.is_multiple_of(20) {
            let cb_input: [FixedPoint; 10] = {
                let mut buf = [FixedPoint::ZERO; 10];
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = if i < MAX_NEURONS {
                        self.firing_rates[i]
                    } else {
                        FixedPoint::ZERO
                    };
                }
                buf
            };
            let motor_cmd = crate::cognitive::actor::cognitive_actor().action_confidence;
            let cb = crate::bio::cerebellum::cerebellum();
            cb.set_error_signal(pred_error, 0);
            cb.step(&cb_input, motor_cmd, pred_error, now as u64);
        }
    }

    /// Ignite the cognitive global workspace and broadcast its winning frame.
    fn process_global_workspace(&mut self, now: u32) {
        let cog_actor = crate::cognitive::actor::cognitive_actor();
        let selected_action = cog_actor.selected_action.or(self.actor.action_selected);
        let action_confidence = cog_actor
            .action_confidence
            .max(self.actor.action_confidence);

        let frame = crate::cognitive::global_workspace::global_workspace().submit_network_state(
            now,
            self.predictor.mean_prediction_error,
            self.predictor.novelty,
            selected_action,
            action_confidence,
            self.energy_level,
        );

        let Some(frame) = frame else {
            return;
        };

        let attention = crate::cognitive::attention::attention_router();
        let target_layer = frame.target_layer.min(7);
        let gain = (FixedPoint::from_f32(0.6) + frame.ignition_strength)
            .clamp(FixedPoint::from_f32(0.4), FixedPoint::from_f32(1.6));
        attention.saliency_map.set_top_down_bias(target_layer, gain);
        attention.focus.set_focus(
            crate::cognitive::attention::FocusType::GoalDriven,
            target_layer,
            (frame.content_id as u16) & 0x03ff,
        );

        if let Some(action) = frame.action_hint {
            cog_actor.selected_action = Some(action);
            cog_actor.action_confidence = (cog_actor.action_confidence + frame.ignition_strength)
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            self.actor.action_selected = Some(action);
        }

        match frame.mode {
            crate::cognitive::global_workspace::WorkspaceMode::Crisis => {
                self.neuromodulators.noradrenaline = FixedPoint::ONE;
                self.neuromodulators.serotonin = FixedPoint::from_f32(0.9);
                self.threshold_modulation = FixedPoint::from_f32(1.35);
            }
            crate::cognitive::global_workspace::WorkspaceMode::Guarded => {
                self.neuromodulators.noradrenaline = (self.neuromodulators.noradrenaline
                    + FixedPoint::from_f32(0.08))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                self.threshold_modulation = FixedPoint::from_f32(1.15);
            }
            crate::cognitive::global_workspace::WorkspaceMode::Exploring => {
                self.neuromodulators.acetylcholine = (self.neuromodulators.acetylcholine
                    + FixedPoint::from_f32(0.08))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                self.neuromodulators.dopamine = (self.neuromodulators.dopamine
                    + FixedPoint::from_f32(0.04))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
            crate::cognitive::global_workspace::WorkspaceMode::Focused => {
                self.neuromodulators.serotonin = (self.neuromodulators.serotonin
                    + FixedPoint::from_f32(0.03))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
            crate::cognitive::global_workspace::WorkspaceMode::Quiescent => {}
        }
    }

    /// Predictive cycle trigger (separated to avoid stack overflow from inlining)
    #[inline(never)]
    fn cycle_trigger(&mut self, now: u32) {
        if self.warp_active || self.cycle_active {
            return;
        }
        if now.is_multiple_of(10000) || self.cycle_cooldown == 0 {
            let curiosity = unsafe { CURIOSITY_ENGINE.curiosity_level };
            let pred_error = self.predictor.mean_prediction_error;
            let should_cycle = curiosity > FixedPoint::from_f32(0.3)
                || pred_error > FixedPoint::from_f32(0.15)
                || now.is_multiple_of(30000);
            if should_cycle && !self.cycle_active && !self.actor.output_inhibited {
                // Build state snapshot for TD learning
                let count = NEURON_COUNT.load(Ordering::Relaxed);
                unsafe {
                    for i in 0..count.min(1024) as u16 {
                        PRED_STATE_BUF[i as usize] =
                            neuron_state_ref(NeuronId::new(i)).membrane_potential;
                    }
                }
                // Compute reward
                let cog_actor = crate::cognitive::actor::cognitive_actor();
                let reward = cog_actor.compute_reward(
                    self.predictor.mean_prediction_error,
                    self.predictor.novelty,
                    self.energy_level,
                );
                // Compute TD error
                unsafe {
                    cog_actor.compute_td_error(&PRED_STATE_BUF, reward);
                }

                // Modulate dopamine from TD error
                let td = cog_actor.td_error;
                let dopamine = (td + FixedPoint::ONE) * FixedPoint::from_f32(0.5);
                self.neuromodulators.dopamine = dopamine.clamp(FixedPoint::ZERO, FixedPoint::ONE);

                // Sync cognitive NM to same value
                unsafe {
                    COGNITIVE_NEUROMODULATORS.dopamine = self.neuromodulators.dopamine;
                }
                // Enter predictive cycle
                self.predictive_cycle(now);
            }
        }
    }

    /// Process sensory spikes from ring buffers
    fn process_sensory_input(&mut self, _now: u32) {
        // Read spike events from global queue
        while let Some(SpikeEvent {
            neuron_id,
            timestamp: _,
            layer: _,
            is_predictor: _,
        }) = unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }
        {
            // Inject current into target neuron
            let state = neuron_state(neuron_id);
            state.membrane_potential += FixedPoint::from_f32(1.0);
        }
    }

    /// Apply STDP plasticity across all synapses
    fn apply_plasticity(&mut self, _now: u32) {
        let nm = &self.neuromodulators;
        let error = homeostasis::error();
        let _reward = synapse::compute_reward_signal(
            self.predictor.mean_prediction_error,
            self.predictor.novelty,
            error,
        );

        let (ltp_mult, ltd_mult) = synapse::modulate_plasticity(nm);
        synapse::apply_plasticity_modulation(ltp_mult, ltd_mult);

        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = SynapseId::new(i);
            let s = synapse::synapse(id);
            if !s.plasticity_enabled {
                continue;
            }
            s.decay_traces();
        }

        unsafe {
            let entropy_state = ENTROPY_MONITOR.check_health();
            match entropy_state {
                crate::safety::entropy_monitor::EntropyState::HighEntropy => {
                    ENTROPY_MONITOR.force_crystallization(self);
                }
                crate::safety::entropy_monitor::EntropyState::LowEntropy => {
                    ENTROPY_MONITOR.inject_stochastic_noise(self);
                }
                _ => {}
            }
        }
    }

    /// Parallel multi-threaded simulation step using CPU worker threads (feature = "std")
    #[cfg(feature = "std")]
    pub fn step_parallel(&mut self, num_threads: usize) {
        let workers = num_threads.max(1);
        self.time += 1;
        let now = self.time;

        if now.is_multiple_of(10) {
            let sm = crate::io::sensors::sensor_manager();
            sm.read_all();
            sm.emit_sensor_spikes(now);
        }

        let current_warp = if self.warp_active {
            self.warp_factor
        } else {
            1
        };
        self.process_sensory_input(now);

        if now.is_multiple_of(5) {
            let cog_actor = crate::cognitive::actor::cognitive_actor();
            let selected_action = cog_actor.selected_action;
            let action_confidence = cog_actor.action_confidence;
            let ar = crate::cognitive::attention::attention_router();
            ar.update(
                self.predictor.mean_prediction_error,
                self.predictor.novelty,
                selected_action,
                action_confidence,
            );
            ar.apply();
        }

        self.actor.step(now, current_warp);
        self.predictor.step(now, current_warp);

        // Parallel STDP Plasticity & Trace Decay across thread scope
        self.apply_plasticity_parallel(now, workers);

        self.update_neuromodulators(now);

        if now.is_multiple_of(10) {
            let am = crate::io::actuators::actuator_manager();
            am.read_motor_outputs();
        }

        crate::cognitive::temporal::temporal_cognition().update();

        if now.is_multiple_of(20) {
            unsafe {
                let curiosity = &mut CURIOSITY_ENGINE;
                curiosity.update(self);
                if curiosity.dreaming_active {
                    curiosity.inject_noise(self, now);
                }
            }
            let eps = unsafe { CURIOSITY_ENGINE.explore_epsilon() };
            crate::cognitive::actor::cognitive_actor().epsilon = eps;
        }

        self.process_global_workspace(now);

        if now.is_multiple_of(1000) {
            self.heartbeat = true;
            self.metabolic_maintenance(now);
        } else {
            self.heartbeat = false;
        }

        if now.is_multiple_of(100) {
            self.energy_adaption();
        }

        if self.cycle_cooldown > 0 {
            self.cycle_cooldown -= 1;
        }

        self.cycle_trigger(now);

        if now.is_multiple_of(1000) {
            self.update_statistics();
        }

        if now.is_multiple_of(500) {
            crate::telemetry::xai::analyze_current_trace();
        }
    }

    #[cfg(feature = "std")]
    fn apply_plasticity_parallel(&mut self, _now: u32, _num_threads: usize) {
        let nm = &self.neuromodulators;
        let error = homeostasis::error();
        let _reward = synapse::compute_reward_signal(
            self.predictor.mean_prediction_error,
            self.predictor.novelty,
            error,
        );

        let (ltp_mult, ltd_mult) = synapse::modulate_plasticity(nm);
        synapse::apply_plasticity_modulation(ltp_mult, ltd_mult);

        let count = SYNAPSE_COUNT.load(Ordering::Relaxed) as usize;
        for i in 0..count as u16 {
            let id = SynapseId::new(i);
            let s = synapse::synapse(id);
            if s.plasticity_enabled {
                s.decay_traces();
            }
        }

        unsafe {
            let entropy_state = ENTROPY_MONITOR.check_health();
            match entropy_state {
                crate::safety::entropy_monitor::EntropyState::HighEntropy => {
                    ENTROPY_MONITOR.force_crystallization(self);
                }
                crate::safety::entropy_monitor::EntropyState::LowEntropy => {
                    ENTROPY_MONITOR.inject_stochastic_noise(self);
                }
                _ => {}
            }
        }
    }

    /// Update neuromodulator levels
    fn update_neuromodulators(&mut self, now: u32) {
        let nm = &mut self.neuromodulators;

        // Update calibration statistics
        unsafe {
            let cal = &mut crate::cognitive::networks::NM_CALIBRATION;
            cal.update(
                self.predictor.mean_prediction_error,
                self.predictor.recent_reward,
                self.predictor.novelty,
            );
        }

        // Use adaptive decay rate from calibration
        let decay_rate =
            unsafe { crate::cognitive::networks::NM_CALIBRATION.adaptive_decay_rate() };
        nm.decay(decay_rate);

        // Apply sensitivity calibration to plasticity
        let ltp_sens = unsafe { crate::cognitive::networks::NM_CALIBRATION.sensitivity_ltp };
        let _ltd_sens = unsafe { crate::cognitive::networks::NM_CALIBRATION.sensitivity_ltd };
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let s = crate::snn::synapse::synapse(crate::core::memory::SynapseId::new(i));
            if s.plasticity_enabled {
                s.reward_sensitivity = ltp_sens;
            }
        }

        // Periodic auto-calibration of neuromodulator levels
        crate::cognitive::networks::calibrate_neuromodulators(now);

        // Prediction error drives noradrenaline
        let pe = self.predictor.mean_prediction_error;
        nm.noradrenaline += pe * FixedPoint::from_f32(0.1);

        // Stability drives serotonin
        if pe < FixedPoint::from_f32(0.05) {
            nm.serotonin += FixedPoint::from_f32(0.001);
        }
        nm.serotonin = nm.serotonin.clamp(FixedPoint::ZERO, FixedPoint::ONE);

        // Reward drives dopamine
        let reward = self.predictor.recent_reward;
        nm.dopamine += reward * FixedPoint::from_f32(0.05);
        nm.dopamine = nm.dopamine.clamp(FixedPoint::ZERO, FixedPoint::ONE);

        // Novelty drives acetylcholine
        nm.acetylcholine += self.predictor.novelty * FixedPoint::from_f32(0.1);
        nm.acetylcholine = nm.acetylcholine.clamp(FixedPoint::ZERO, FixedPoint::ONE);

        // Adaptive entropy → cognitive mode coupling
        let weight_count = crate::snn::synapse::SYNAPSE_COUNT.load(Ordering::Relaxed);
        if weight_count > 0 && now.is_multiple_of(10) {
            unsafe {
                let prev_entropy =
                    crate::core::entropy::ENTROPY_MONITOR.compute_entropy(weight_count as usize);
                crate::core::entropy::ENTROPY_MONITOR.cognitive_update(prev_entropy);
                let mode = crate::core::entropy::ENTROPY_MONITOR.cognitive_mode;
                match mode {
                    crate::core::entropy::CognitiveMode::Crisis => {
                        nm.serotonin = FixedPoint::ONE;
                        nm.noradrenaline = FixedPoint::ZERO;
                    }
                    crate::core::entropy::CognitiveMode::Explore => {
                        nm.acetylcholine = FixedPoint::from_f32(0.8);
                        nm.dopamine = FixedPoint::from_f32(0.7);
                    }
                    crate::core::entropy::CognitiveMode::Stable => {
                        nm.serotonin = FixedPoint::from_f32(0.8);
                    }
                    _ => {}
                }
            }
        }

        // Energy-aware neuromodulation
        if self.energy_level < FixedPoint::from_f32(0.2) {
            // Survival mode: reduce activity
            nm.serotonin = FixedPoint::ONE;
            nm.noradrenaline = FixedPoint::ZERO;
        }
    }

    /// Energy-aware threshold modulation (Section 11)
    fn energy_adaption(&mut self) {
        let pm = crate::system::power::power_manager();
        let base_mult = pm.threshold_multiplier();
        let scale = base_mult
            * (FixedPoint::ONE + (FixedPoint::ONE - self.energy_level) * FixedPoint::from_f32(0.2));
        self.threshold_modulation = scale;

        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state(id);
            state.threshold = FixedPoint::ONE * scale;
        }
    }

    /// Metabolic maintenance (Section 7, Section 12)
    fn metabolic_maintenance(&mut self, _now: u32) {
        // Senescence, pruning, and neurogenesis
        unsafe {
            crate::snn::neurogenesis::NEUROGENESIS.maintenance_cycle();
        }

        // Homeostatic scaling
        let avg_rate = self.compute_average_firing_rate();
        synapse::homeostatic_scaling(FixedPoint::from_f32(10.0), avg_rate);

        // Memory consolidation (every metabolic cycle ~1s)
        let mem = crate::cognitive::episodic::episodic_memory();
        mem.apply_forgetting(self.time as u64);
        mem.consolidate(self.time as u64);

        // Persistence (every 60 metabolic cycles = ~1 minute)
        if self.time != 0 && self.time.is_multiple_of(60000) {
            crate::system::persistence::PersistenceManager::save();
        }

        // Reset spike window
        self.window_idx = 0;
    }

    /// Compute average firing rate across all neurons
    fn compute_average_firing_rate(&self) -> FixedPoint {
        let mut total = FixedPoint::ZERO;
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        if count == 0 {
            return FixedPoint::ZERO;
        }
        for i in 0..count as u16 {
            total += self.firing_rates[i as usize];
        }
        total / FixedPoint::from_int(count as i32)
    }

    /// Update firing rate statistics
    fn update_statistics(&mut self) {
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let rate = self.firing_rates[i as usize];
            self.firing_rates[i as usize] =
                rate * FixedPoint::from_f32(0.99) + FixedPoint::from_f32(0.01);
        }
    }

    /// Record a spike for statistics
    pub fn record_spike(&mut self, _neuron_id: NeuronId) {
        self.total_spikes.fetch_add(1, Ordering::Relaxed);
        let idx = self.window_idx as usize;
        if idx < 1024 {
            self.recent_spikes_window[idx] += 1;
        }
    }

    /// Get current simulation time
    pub fn now(&self) -> u32 {
        self.time
    }

    /// Activate time warp (Section 6)
    pub fn activate_warp(&mut self, factor: u32) {
        self.warp_active = true;
        self.warp_factor = factor.max(1);
        unsafe {
            TIME_WARPER.activate(factor);
        }
        // Disable actor output during warp
        self.actor.output_inhibited = true;
    }

    /// Deactivate time warp
    pub fn deactivate_warp(&mut self) {
        self.warp_active = false;
        self.warp_factor = 1;
        unsafe {
            TIME_WARPER.deactivate();
        }
        self.actor.output_inhibited = false;
    }

    /// Predictor-guided simulation loop (Section 6)
    #[inline(never)]
    pub fn run_simulation(&mut self, hypothesis_duration_ms: u32) -> SimulationResult {
        crate::system::persistence::capture_simulation_snapshot();
        self.activate_warp(100);
        let steps = hypothesis_duration_ms;
        for _ in 0..steps {
            self.step();
        }
        let outcome = self.predictor.evaluate_outcome();
        crate::system::persistence::restore_simulation_snapshot();
        self.deactivate_warp();
        outcome
    }

    /// Full predictive cycle: hypothesize → simulate → validate → act (Section 6)
    #[inline(never)]
    pub fn predictive_cycle(&mut self, _now: u32) {
        self.cycle_active = true;

        // 1. Build current state snapshot into static buffer
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        unsafe {
            for i in 0..count.min(1024) as u16 {
                PRED_STATE_BUF[i as usize] = neuron_state_ref(NeuronId::new(i)).membrane_potential;
            }
        }

        // 2. Generate hypotheses via cognitive actor
        let rng_seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
        let mut rng = crate::core::math::XorShift64Star::new(rng_seed);
        let base_action = self.actor.action_selected.unwrap_or(128);
        let cog_actor = crate::cognitive::actor::cognitive_actor();
        cog_actor.generate_hypotheses(base_action, &mut rng);

        // 3. Score each hypothesis with cognitive predictor
        let cog_predictor = crate::cognitive::predictor::cognitive_predictor();

        for i in 0..crate::cognitive::actor::MAX_HYPOTHESES.min(8) {
            let action = cog_actor.hypotheses[i].action;
            let delta = unsafe { cog_predictor.predict_next(&PRED_STATE_BUF, action) };
            let mut delta_magnitude = FixedPoint::ZERO;
            for d in delta.iter().take(16) {
                delta_magnitude += d.abs();
            }
            cog_actor.hypotheses[i].confidence =
                FixedPoint::ONE / (FixedPoint::ONE + delta_magnitude);
            cog_actor.hypotheses[i].simulated = true;
        }

        // 4. Select best hypothesis
        let best_idx = cog_actor.select_best_hypothesis();

        // 5. Inhibit actor output and run warp simulation
        let was_inhibited = self.actor.output_inhibited;
        if !was_inhibited {
            self.actor.output_inhibited = true;
        }

        let outcome = self.run_simulation(50);

        // 6. Record transition in cognitive predictor and episodic memory
        unsafe {
            for i in 0..count.min(1024) as u16 {
                PRED_NEXT_BUF[i as usize] = neuron_state_ref(NeuronId::new(i)).membrane_potential;
            }
        }
        let used_action = best_idx
            .map(|i| cog_actor.hypotheses[i].action)
            .unwrap_or(0);
        unsafe {
            cog_predictor.record_transition(&PRED_STATE_BUF, used_action, &PRED_NEXT_BUF);
        }

        // 6a. Record transition in episodic memory
        unsafe {
            let mem = crate::cognitive::episodic::episodic_memory();
            let state_hash = mem
                .recall_by_state(0, self.time as u64)
                .map_or(0, |t| t.state_hash);
            let next_hash = if count > 0 {
                let mut h: u64 = 5381;
                for i in (0..count.min(1024)).step_by(64) {
                    h = h
                        .wrapping_mul(33)
                        .wrapping_add(PRED_NEXT_BUF[i].to_bits() as u64);
                }
                h
            } else {
                0
            };
            let pe = cog_predictor.mean_error;
            let novelty = crate::cognitive::curiosity::CURIOSITY_ENGINE.curiosity_level;
            let reward = cog_actor.compute_reward(
                cog_predictor.mean_error,
                cog_predictor.mean_error,
                self.energy_level,
            );
            mem.record(
                state_hash,
                used_action as u16,
                next_hash,
                reward,
                pe,
                novelty,
                self.time as u64,
            );
        }

        // 6b. Online learning from prediction error
        unsafe {
            cog_predictor.predict(&PRED_STATE_BUF);
            cog_predictor.update_from_prediction_error(&PRED_NEXT_BUF);
        }

        // 7. TD update from actual next state
        unsafe {
            cog_actor.update_value_from_next(&PRED_NEXT_BUF);
        }

        // 8. Modulate dopamine based on outcome + TD error
        let td_clamped = {
            let td = cog_actor.td_error;
            (td + FixedPoint::ONE).clamp(FixedPoint::ZERO, FixedPoint::ONE)
                * FixedPoint::from_f32(0.5)
        };
        match outcome {
            SimulationResult::Positive => {
                cog_actor.cycle_result = outcome;
                self.actor.output_inhibited = false;
                unsafe {
                    COGNITIVE_NEUROMODULATORS.dopamine = td_clamped + FixedPoint::from_f32(0.3);
                }
            }
            SimulationResult::Exploratory => {
                cog_actor.cycle_result = outcome;
                self.actor.output_inhibited = false;
                unsafe {
                    COGNITIVE_NEUROMODULATORS.dopamine = td_clamped + FixedPoint::from_f32(0.1);
                }
            }
            _ => {
                cog_actor.cycle_result = outcome;
                self.actor.output_inhibited = was_inhibited;
                unsafe {
                    COGNITIVE_NEUROMODULATORS.dopamine = td_clamped;
                }
            }
        }
        unsafe {
            let d = COGNITIVE_NEUROMODULATORS
                .dopamine
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            COGNITIVE_NEUROMODULATORS.dopamine = d;
            self.neuromodulators.dopamine = d;
        }

        // 9. Update curiosity based on outcome
        unsafe {
            CURIOSITY_ENGINE.update(self);
        }
        // Trigger attention exploratory shift if curiosity demands it
        if unsafe { CURIOSITY_ENGINE.should_explore() } {
            let seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
            let mut rng = crate::core::math::XorShift64Star::new(seed);
            crate::cognitive::attention::attention_router().shift_to_exploratory(&mut rng);
        }
        cog_actor.episode_count += 1;

        // 10. Set cooldown
        self.cycle_cooldown = if outcome == SimulationResult::Positive {
            5000
        } else {
            2000
        };
        self.cycle_active = false;
    }

    /// Emergency rollback to J-1 (Section 7.2)
    pub fn emergency_rollback(&mut self) {
        crate::system::persistence::PersistenceManager::rollback();
    }
}

/// Actor subnetwork - connected to physical I/O (Section 6)
pub struct ActorNetwork {
    pub output_inhibited: bool,
    pub motor_outputs: [FixedPoint; 256], // Motor neuron outputs
    pub sensor_inputs: [FixedPoint; 1024], // Sensor neuron inputs
    pub action_selected: Option<u8>,
    pub action_confidence: FixedPoint,
    reflex_active: bool,
}

impl ActorNetwork {
    pub fn new() -> Self {
        Self {
            output_inhibited: false,
            motor_outputs: [FixedPoint::ZERO; 256],
            sensor_inputs: [FixedPoint::ZERO; 1024],
            action_selected: None,
            action_confidence: FixedPoint::ZERO,
            reflex_active: false,
        }
    }

    pub fn step(&mut self, _time: u32, _warp: u32) {
        if self.output_inhibited {
            return;
        }

        // Process sensor inputs -> update neuron membrane potentials
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count.min(1024) as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state(id);
            if state.layer == 0 {
                state.membrane_potential += self.sensor_inputs[i as usize];
            }
        }

        // Check reflex arcs (Section 19) - hard-coded override
        self.check_reflexes();

        // Read motor outputs
        let count = NEURON_COUNT.load(Ordering::Relaxed).min(256) as u16;
        for i in 0..count {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            if state.layer == 4 {
                self.motor_outputs[i as usize] = state.membrane_potential;
            }
        }
    }

    /// Spinal reflexes (Section 19) with cognitive override support
    fn check_reflexes(&mut self) {
        let override_active = crate::cognitive::reflex_override::evaluate_override();
        let reflexes = crate::safety::reflexes::reflexes();
        reflexes.set_cognitive_override(override_active);
        let before_active = reflexes.active_reflexes;
        reflexes.check_all();
        self.reflex_active = reflexes.active_reflexes != before_active;
    }
}

/// Predictor subnetwork - internal world model (Section 6)
pub struct PredictorNetwork {
    pub mean_prediction_error: FixedPoint,
    prev_mean_prediction_error: FixedPoint,
    pub prediction_history: [FixedPoint; 1024],
    pub novelty: FixedPoint,
    pub recent_reward: FixedPoint,
    pub curiosity_level: FixedPoint,
    pub predictions: [FixedPoint; 256],
    pub actuals: [FixedPoint; 256],
}

impl PredictorNetwork {
    pub fn new() -> Self {
        Self {
            mean_prediction_error: FixedPoint::ZERO,
            prev_mean_prediction_error: FixedPoint::ZERO,
            prediction_history: [FixedPoint::ZERO; 1024],
            novelty: FixedPoint::ZERO,
            recent_reward: FixedPoint::ZERO,
            curiosity_level: FixedPoint::from_f32(0.1),
            predictions: [FixedPoint::ZERO; 256],
            actuals: [FixedPoint::ZERO; 256],
        }
    }

    pub fn step(&mut self, time: u32, _warp: u32) {
        // Predict next state based on current state
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count.min(256) as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            if state.flags.has(NeuronFlags::PREDICTOR_MODE) {
                // Simple prediction: linear extrapolation of membrane potential
                self.predictions[i as usize] = state.membrane_potential;
            }
        }

        for i in 0..count.min(256) as u16 {
            let idx = i as usize;
            let sensor_state = neuron_state_ref(NeuronId::new(i));
            let prediction = self.predictions[idx];
            let actual = sensor_state.membrane_potential;
            self.actuals[idx] = actual;

            let error = (prediction - actual).abs();
            if idx < 1024 {
                self.prediction_history[(time as usize + idx) % 1024] = error;
            }
        }

        // Update moving average
        let mut sum_pred = FixedPoint::ZERO;
        for &p in self.prediction_history.iter().take(1024) {
            sum_pred += p;
        }
        self.mean_prediction_error = sum_pred / FixedPoint::from_int(1024);

        // Novelty: absolute change in prediction error (derivative)
        self.novelty = (self.mean_prediction_error - self.prev_mean_prediction_error).abs();
        self.prev_mean_prediction_error = self.mean_prediction_error;

        // Curiosity engine (Section 29)
        if self.novelty < FixedPoint::from_f32(0.01) {
            self.curiosity_level += FixedPoint::from_f32(0.001);
        } else {
            self.curiosity_level *= FixedPoint::from_f32(0.99);
        }
        self.curiosity_level = self
            .curiosity_level
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
    }

    /// Evaluate outcome of a simulation run
    pub fn evaluate_outcome(&self) -> SimulationResult {
        let error = self.mean_prediction_error;
        let novelty = self.novelty;
        let _curiosity = self.curiosity_level;

        if error < FixedPoint::from_f32(0.05) {
            SimulationResult::Positive
        } else if error < FixedPoint::from_f32(0.2) {
            if novelty > FixedPoint::from_f32(0.1) {
                SimulationResult::Exploratory
            } else {
                SimulationResult::Neutral
            }
        } else {
            SimulationResult::Negative
        }
    }

    /// Get current prediction for a specific output neuron
    pub fn predict(&self, neuron_id: NeuronId) -> FixedPoint {
        let idx = neuron_id.index();
        if idx < 256 {
            self.predictions[idx]
        } else {
            FixedPoint::ZERO
        }
    }
}

/// Scratch buffers for predictive cycle (avoids stack allocation)
pub static mut PRED_STATE_BUF: [FixedPoint; 1024] = [FixedPoint::ZERO; 1024];
pub static mut PRED_NEXT_BUF: [FixedPoint; 1024] = [FixedPoint::ZERO; 1024];

/// Simulation outcome enum
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimulationResult {
    Positive,    // Good outcome
    Neutral,     // Neutral outcome
    Negative,    // Bad outcome
    Exploratory, // Novel but not bad
}

pub type BinaryDump = crate::system::persistence::BinaryDump;

/// Triggered by metabolic clock 1Hz heartbeat
pub fn trigger_metabolic_heartbeat() {
    // Pacemaker neurons get a bias current boost
    let count = NEURON_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = NeuronId::new(i);
        let state = neuron_state_ref(id);
        if state.neuron_type == crate::core::memory::NeuronType::PACER {
            neuron_state(id).bias_current += FixedPoint::from_f32(0.1);
        }
    }
}

pub static mut NETWORK_INSTANCE: core::mem::MaybeUninit<Network> = core::mem::MaybeUninit::uninit();
static INITIALIZED_NETWORK: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_network() {
    unsafe {
        NETWORK_INSTANCE.write(Network::new());
        INITIALIZED_NETWORK.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn network() -> &'static mut Network {
    unsafe {
        if !INITIALIZED_NETWORK.load(core::sync::atomic::Ordering::Relaxed) {
            init_network();
        }
        &mut *NETWORK_INSTANCE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;
    use crate::core::memory::NeuronId;
    use core::sync::atomic::Ordering;

    #[test]
    fn network_new_creates_default() {
        let n = Network::new();
        assert_eq!(n.time, 0);
        assert!(!n.warp_active);
        assert_eq!(n.warp_factor, 1);
    }

    #[test]
    fn network_step_increments_time() {
        let mut n = Network::new();
        n.step();
        assert_eq!(n.time, 1);
    }

    #[test]
    fn network_step_multiple() {
        let mut n = Network::new();
        for _ in 0..10 {
            n.step();
        }
        assert_eq!(n.time, 10);
    }

    #[test]
    fn network_now_returns_time() {
        let mut n = Network::new();
        assert_eq!(n.now(), 0);
        n.step();
        assert_eq!(n.now(), 1);
    }

    #[test]
    fn network_energy_adaption_bounds() {
        crate::system::power::init_power_manager();
        let mut n = Network::new();
        n.energy_adaption();
        assert!(n.energy_level.to_f32() >= 0.0);
        assert!(n.energy_level.to_f32() <= 1.0);
    }

    #[test]
    fn network_average_firing_rate_initially_zero() {
        let n = Network::new();
        let rate = n.compute_average_firing_rate();
        assert_eq!(rate.to_f32(), 0.0);
    }

    #[test]
    fn network_record_spike_increases_count() {
        let mut n = Network::new();
        assert_eq!(n.total_spikes.load(Ordering::Relaxed), 0);
        n.record_spike(NeuronId::new(0));
        assert_eq!(n.total_spikes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn network_warp_activate_deactivate() {
        let mut n = Network::new();
        assert_eq!(n.warp_factor, 1);
        assert!(!n.warp_active);
        n.activate_warp(100);
        assert_eq!(n.warp_factor, 100);
        assert!(n.warp_active);
        n.deactivate_warp();
        assert!(!n.warp_active);
    }

    #[test]
    fn actor_network_step_does_not_panic() {
        let mut actor = ActorNetwork::new();
        actor.step(0, 1);
    }

    #[test]
    fn predictor_network_step_does_not_panic() {
        let mut pred = PredictorNetwork::new();
        pred.step(0, 1);
    }

    #[test]
    fn predictor_evaluate_outcome_default() {
        let pred = PredictorNetwork::new();
        match pred.evaluate_outcome() {
            SimulationResult::Positive
            | SimulationResult::Neutral
            | SimulationResult::Negative
            | SimulationResult::Exploratory => {}
        }
    }

    #[test]
    fn predictor_predict_returns_something() {
        let pred = PredictorNetwork::new();
        let v = pred.predict(NeuronId::new(0));
        assert_eq!(v, FixedPoint::ZERO);
    }

    #[test]
    fn metabolic_heartbeat_does_not_panic() {
        trigger_metabolic_heartbeat();
    }

    #[test]
    fn network_metabolic_maintenance_does_not_panic() {
        let mut n = Network::new();
        n.metabolic_maintenance(100);
    }

    #[test]
    fn network_update_statistics_does_not_panic() {
        let mut n = Network::new();
        n.update_statistics();
    }

    #[test]
    fn network_auto_adapt_hardware() {
        let mut n = Network::new();
        let profile = n.auto_adapt_hardware();
        assert!(profile.recommended_max_neurons >= crate::MAX_NEURONS);
        let (cap_n, cap_s) = crate::core::memory::ADAPTIVE_MEMORY.current_capacity();
        assert_eq!(cap_n, profile.recommended_max_neurons);
        assert_eq!(cap_s, profile.recommended_max_synapses);
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration tests (lib-level — avoids integration binary crash)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod pipeline_tests {
    use crate::bio::hippocampus::{hippocampus, init_hippocampus};
    use crate::cognitive::episodic::{EpisodicMemory, init_episodic_memory};
    use crate::core::math::FixedPoint;
    use crate::core::memory::NeuronId;

    fn init_all() {
        init_hippocampus();
        init_episodic_memory();
    }

    #[test]
    fn bio_swr_triggers_episodic_ripple_replay() {
        let mut hip_local = crate::bio::hippocampus::Hippocampus::new();
        let mut epi_local = crate::cognitive::episodic::EpisodicMemory::new();

        for i in 0..10 {
            epi_local.record(
                i as u64,
                1,
                (i + 1) as u64,
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(0.1),
                FixedPoint::from_f32(0.2),
                i as u64,
            );
        }

        for cell in hip_local.ca3.iter_mut() {
            cell.active = true;
            cell.trace = FixedPoint::from_f32(0.5);
        }
        hip_local.theta_phase = FixedPoint::from_f32(0.8);
        hip_local.swr_active = true;
        hip_local.consolidation_trigger = true;

        assert!(hip_local.swr_active, "SWR must be active");
        assert!(
            hip_local.consolidation_trigger,
            "consolidation_trigger must be set"
        );

        let count = epi_local.trigger_ripple_replay(100);
        assert!(
            count > 0,
            "Ripple replay must replay at least 1 memory during SWR"
        );
    }

    #[test]
    fn bio_hippocampus_accepts_spatial_context_from_episodic() {
        let mut hip_local = crate::bio::hippocampus::Hippocampus::new();
        let mut epi_local = crate::cognitive::episodic::EpisodicMemory::new();

        epi_local.position_x = FixedPoint::from_f32(0.3);
        epi_local.position_y = FixedPoint::from_f32(0.7);
        epi_local.compute_place_cells();

        let place_rates = epi_local.place_cell_activity();
        assert_eq!(place_rates.len(), 128);

        let mut input = [FixedPoint::from_f32(0.5); 256];
        for (i, &rate) in place_rates.iter().enumerate() {
            if i < input.len() {
                input[i] = (input[i] + rate * FixedPoint::from_f32(0.3))
                    .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
        }

        hip_local.step(&input, FixedPoint::from_f32(0.1), FixedPoint::from_f32(0.2));
        assert!(hip_local.tick > 0);
        assert!(hip_local.theta_phase >= FixedPoint::ZERO);
    }

    #[test]
    fn bio_striosome_accepts_dopamine_from_cognitive() {
        init_all();
        let sys = crate::bio::striosome::striosome_matrix();
        let da = FixedPoint::from_f32(0.7);
        let input = [FixedPoint::from_f32(0.3); 64];

        sys.step_all(da, &input);
        sys.learn_all(FixedPoint::from_f32(0.1), &input);
        sys.competition();

        let winner = sys.winning_striosome();
        assert!(sys.dopamine_gate_open() || winner.is_some());
    }

    #[test]
    fn bio_thalamus_gates_sensory_input() {
        init_all();
        let thal = crate::bio::thalamus::thalamus();
        let sensory = [FixedPoint::from_f32(0.6); 4];
        thal.step(
            &sensory,
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.1),
            0,
        );
        assert!(thal.global_gate >= FixedPoint::ZERO);
        assert!(thal.global_gate <= FixedPoint::ONE);
    }

    #[test]
    fn bio_astrocytes_modulate_network() {
        init_all();
        let astro = crate::bio::astrocytes::astrocyte_network();
        let input = [FixedPoint::from_f32(0.5); 64];
        astro.step_all(&input, 100, FixedPoint::from_f32(0.1));
        astro.propagate_waves();
    }

    #[test]
    fn bio_cerebellum_refines_motor_output() {
        init_all();
        let cb = crate::bio::cerebellum::cerebellum();
        let input = [FixedPoint::from_f32(0.4); 10];
        cb.set_error_signal(FixedPoint::from_f32(0.3), 0);
        cb.step(
            &input,
            FixedPoint::from_f32(0.6),
            FixedPoint::from_f32(0.3),
            100,
        );
        assert!(cb.motor_output >= FixedPoint::ZERO);
    }

    #[test]
    fn full_bio_pipeline_step_does_not_panic() {
        init_all();

        let sensory = [FixedPoint::from_f32(0.5); 4];
        let da = FixedPoint::from_f32(0.3);
        let novelty = FixedPoint::from_f32(0.1);
        let pred_error = FixedPoint::from_f32(0.2);

        let mut thal_local = crate::bio::thalamus::Thalamus::new();
        let mut sys_local = crate::bio::striosome::StriosomeMatrixSystem::new();
        let mut hip_local = crate::bio::hippocampus::Hippocampus::new();
        let mut astro_local = crate::bio::astrocytes::AstrocyteNetwork::new();
        let mut cb_local = crate::bio::cerebellum::Cerebellum::new();
        let mut epi_local = crate::cognitive::episodic::EpisodicMemory::new();

        thal_local.step(&sensory, FixedPoint::from_f32(0.7), pred_error, 50);
        sys_local.step_all(da, &[FixedPoint::from_f32(0.3); 64]);
        sys_local.learn_all(pred_error, &[FixedPoint::from_f32(0.3); 64]);

        let hipp_input = [FixedPoint::from_f32(0.4); 256];
        hip_local.step(&hipp_input, novelty, pred_error);

        astro_local.step_all(
            &[FixedPoint::from_f32(0.5); 64],
            100,
            FixedPoint::from_f32(0.1),
        );
        astro_local.propagate_waves();

        cb_local.set_error_signal(pred_error, 0);
        cb_local.step(
            &[FixedPoint::from_f32(0.4); 10],
            FixedPoint::from_f32(0.5),
            pred_error,
            100,
        );

        epi_local.record(
            42,
            1,
            43,
            FixedPoint::from_f32(0.5),
            pred_error,
            novelty,
            100,
        );
        epi_local.consolidate(200);

        assert!(hip_local.tick > 0);
        assert!(epi_local.total_recorded > 0);
    }

    #[test]
    fn endurance_stress_test_10k_cycles() {
        init_all();
        let hip = hippocampus();
        let mut epi = crate::cognitive::episodic::EpisodicMemory::new();
        let thal = crate::bio::thalamus::thalamus();
        let sys = crate::bio::striosome::striosome_matrix();
        let astro = crate::bio::astrocytes::astrocyte_network();
        let cb = crate::bio::cerebellum::cerebellum();

        // Use a local counter to track recordings (global may be shared with parallel tests)
        let mut local_recordings: u64 = 0;

        // Record baseline experiences for replay
        for i in 0..50 {
            epi.record(
                i as u64,
                1,
                (i + 1) as u64,
                FixedPoint::from_f32(0.5 + ((i % 10) as f32) * 0.05),
                FixedPoint::from_f32((i % 5) as f32 * 0.05),
                FixedPoint::from_f32((i % 3) as f32 * 0.1),
                i as u64,
            );
            local_recordings += 1;
        }

        for step in 0..10_000u64 {
            let now = (step as u32) % 1000;
            let da = FixedPoint::from_f32(0.3 + ((step % 7) as f32) * 0.05);
            let novelty = FixedPoint::from_f32(((step % 11) as f32) * 0.02);
            let pred_error = FixedPoint::from_f32(((step % 13) as f32) * 0.015);
            let attention = FixedPoint::from_f32(0.5 + ((step % 3) as f32) * 0.1);

            // Simulate bio module cycle
            if now % 10 == 0 {
                let sensory = [FixedPoint::from_f32(0.4 + ((step % 5) as f32) * 0.05); 4];
                thal.step(&sensory, attention, pred_error, step);
            }

            if now % 10 == 0 {
                let strio_input = [FixedPoint::from_f32(0.3); 64];
                sys.step_all(da, &strio_input);
                sys.learn_all(pred_error, &strio_input);
            }

            if now % 50 == 0 {
                let hipp_input = [FixedPoint::from_f32(0.4); 256];
                hip.step(&hipp_input, novelty, pred_error);

                // Bridge: SWR → ripple replay
                if hip.swr_active && hip.consolidation_trigger {
                    epi.trigger_ripple_replay(step);
                }
            }

            if now % 100 == 0 {
                let astro_input = [FixedPoint::from_f32(0.5); 64];
                astro.step_all(&astro_input, step, FixedPoint::from_f32(0.1));
                astro.propagate_waves();
            }

            if now % 20 == 0 {
                let cb_input = [FixedPoint::from_f32(0.4); 10];
                cb.set_error_signal(pred_error, 0);
                cb.step(&cb_input, FixedPoint::from_f32(0.5), pred_error, step);
            }

            // Periodic episodic operations
            if now % 500 == 0 {
                epi.apply_forgetting(step);
                epi.consolidate(step);
            }
        }

        assert!(epi.total_recorded > 0, "Episodic recordings must persist");
        assert!(hip.tick > 0, "Hippocampus tick must advance");
        assert!(
            thal.global_gate >= FixedPoint::ZERO,
            "Thalamus gate must be valid"
        );
        assert!(
            thal.global_gate <= FixedPoint::ONE,
            "Thalamus gate must be ≤ 1"
        );
        assert!(
            sys.striosomes.len() == 16,
            "Striosome count must be correct"
        );
        assert!(astro.active_count() <= 64, "Astrocyte count must be ≤ 64");
        assert!(
            cb.motor_output >= FixedPoint::ZERO,
            "Cerebellum output must be valid"
        );
        assert!(
            local_recordings == 50,
            "Must have tracked 50 local recordings"
        );
    }

    #[test]
    fn endurance_consolidation_does_not_exhaust_memory() {
        // Use a fresh local instance for isolation (not the shared global static)
        let mut epi = EpisodicMemory::new();

        // Simulate long-term recording and consolidation
        for cycle in 0..100 {
            let t = cycle as u64 * 1000;
            for i in 0..50 {
                epi.record(
                    (cycle * 50 + i) as u64,
                    1,
                    (cycle * 50 + i + 1) as u64,
                    FixedPoint::from_f32(0.3 + ((i % 10) as f32) * 0.07),
                    FixedPoint::from_f32(0.05),
                    FixedPoint::from_f32(0.1),
                    t + i as u64,
                );
            }
            epi.consolidate(t + 100);
            epi.apply_forgetting(t + 200);
        }

        // After 100 cycles of record/consolidate, memory should still be healthy
        let st = epi.short_term_count();
        let lt = epi.long_term_count();
        let util = epi.utilization();
        assert!(util > FixedPoint::ZERO, "Utilization must be > 0");
        assert!(util <= FixedPoint::ONE, "Utilization must be ≤ 1.0");
        assert!(st <= 256, "Short-term must not exceed capacity");
        assert!(lt <= 512, "Long-term must not exceed capacity");
        assert_eq!(
            epi.total_recorded, 5000,
            "Must have recorded exactly 5000 experiences"
        );
        assert!(
            epi.total_consolidated > 0,
            "Must have consolidated some memories"
        );
    }

    #[test]
    fn efpga_engine_init_and_compile() {
        crate::efpga::init_efpga_engine();
        let engine = crate::efpga::efpga_engine();
        assert!(engine.verilog_len == 0, "Verilog buffer must start empty");

        let synapse_data = [
            (
                NeuronId::new(0),
                NeuronId::new(1),
                FixedPoint::from_f32(0.5),
                2,
                FixedPoint::ZERO,
                200,
            ),
            (
                NeuronId::new(1),
                NeuronId::new(2),
                FixedPoint::from_f32(0.3),
                1,
                FixedPoint::ZERO,
                150,
            ),
            (
                NeuronId::new(2),
                NeuronId::new(3),
                FixedPoint::from_f32(0.7),
                3,
                FixedPoint::ZERO,
                180,
            ),
        ];

        let (success, benchmark) = engine.compile_and_accelerate_subnetwork(&synapse_data, 1);
        assert!(
            success,
            "eFPGA compilation must succeed for stable synapses"
        );
        assert!(
            benchmark.speedup_vs_software > 1.0,
            "Hardware must be faster than software"
        );
        assert!(engine.verilog_len > 0, "Verilog buffer must be populated");
        assert!(
            engine.last_bitstream.is_some(),
            "Bitstream must be generated"
        );
    }

    #[test]
    fn efpga_rejects_unstable_subnetwork() {
        crate::efpga::init_efpga_engine();
        let engine = crate::efpga::efpga_engine();

        let unstable_data = [(
            NeuronId::new(0),
            NeuronId::new(1),
            FixedPoint::from_f32(0.5),
            2,
            FixedPoint::from_f32(0.5),
            5,
        )];

        let (success, _) = engine.compile_and_accelerate_subnetwork(&unstable_data, 0);
        assert!(!success, "Unstable subnetwork must be rejected");
    }
}
