//! Synaptic plasticity rules: STDP, Hebbian, and neuromodulated weight updates.
use crate::core::math::FixedPoint;
use crate::core::memory::{NeuronId, SynapseId};
use crate::snn::synapse::{self, SYNAPSE_COUNT};
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;

#[derive(Clone, Copy)]
pub struct CalciumModel {
    pub concentration: FixedPoint,
    pub decay_per_ms: FixedPoint,
    pub ltp_threshold: FixedPoint,
    pub ltd_threshold: FixedPoint,
    pub influx_per_spike: FixedPoint,
    pub plateau_influx_multiplier: FixedPoint,
}

impl CalciumModel {
    pub const fn new_const() -> Self {
        Self {
            concentration: FixedPoint::ZERO,
            decay_per_ms: FixedPoint::from_f32(0.98),
            ltp_threshold: FixedPoint::from_f32(0.15),
            ltd_threshold: FixedPoint::from_f32(0.05),
            influx_per_spike: FixedPoint::from_f32(0.03),
            plateau_influx_multiplier: FixedPoint::from_f32(3.0),
        }
    }

    pub fn on_spike(&mut self, time: u32, last_time: u32) {
        let dt = (time - last_time) as i32;
        if dt > 0 {
            let decay_pow = self.decay_per_ms.powi(dt);
            self.concentration *= decay_pow;
        }
        self.concentration += self.influx_per_spike;
    }

    pub fn on_plateau(&mut self, time: u32, last_time: u32) {
        let dt = (time - last_time) as i32;
        if dt > 0 {
            let decay_pow = self.decay_per_ms.powi(dt);
            self.concentration *= decay_pow;
        }
        self.concentration += self.influx_per_spike * self.plateau_influx_multiplier;
    }

    /// Returns LTP or LTD magnitude based on calcium concentration.
    pub fn plasticity_direction(&self) -> (FixedPoint, FixedPoint) {
        if self.concentration >= self.ltp_threshold {
            let excess = self.concentration - self.ltp_threshold;
            (excess, FixedPoint::ZERO)
        } else if self.concentration >= self.ltd_threshold {
            let mid = (self.concentration - self.ltd_threshold)
                / (self.ltp_threshold - self.ltd_threshold);
            (FixedPoint::ZERO, FixedPoint::ONE - mid)
        } else {
            (FixedPoint::ZERO, FixedPoint::ZERO)
        }
    }

    pub fn decay(&mut self, dt_ms: u32) {
        if dt_ms > 0 {
            self.concentration *= self.decay_per_ms.powi(dt_ms as i32);
        }
    }
}

#[derive(Clone, Copy)]
pub struct PlateauPotential {
    pub active: bool,
    pub duration_ms: u32,
    pub start_time: u32,
    pub amplitude: FixedPoint,
    pub calcium_multiplier: FixedPoint,
}

impl PlateauPotential {
    pub const fn new_const() -> Self {
        Self {
            active: false,
            duration_ms: 100,
            start_time: 0,
            amplitude: FixedPoint::from_f32(0.3),
            calcium_multiplier: FixedPoint::from_f32(3.0),
        }
    }

    pub fn trigger(&mut self, time: u32) {
        self.active = true;
        self.start_time = time;
    }

    pub fn update(&mut self, time: u32) -> bool {
        if self.active && time.saturating_sub(self.start_time) >= self.duration_ms {
            self.active = false;
        }
        self.active
    }
}

pub static mut CALCIUM_MODELS: [MaybeUninit<CalciumModel>; crate::MAX_NEURONS] =
    [const { MaybeUninit::new(CalciumModel::new_const()) }; crate::MAX_NEURONS];
pub static mut PLATEAU_POTENTIALS: [MaybeUninit<PlateauPotential>; crate::MAX_NEURONS] =
    [const { MaybeUninit::new(PlateauPotential::new_const()) }; crate::MAX_NEURONS];

pub fn calcium_model(neuron: NeuronId) -> &'static mut CalciumModel {
    unsafe { &mut *CALCIUM_MODELS[neuron.index() as usize].as_mut_ptr() }
}

pub fn plateau_potential(neuron: NeuronId) -> &'static mut PlateauPotential {
    unsafe { &mut *PLATEAU_POTENTIALS[neuron.index() as usize].as_mut_ptr() }
}

pub fn trigger_plateau(neuron: NeuronId, time: u32) {
    let pp = plateau_potential(neuron);
    pp.trigger(time);
    let cm = calcium_model(neuron);
    cm.on_plateau(time, 0);
}

pub fn update_calcium_on_spike(neuron: NeuronId, time: u32) {
    let cm = calcium_model(neuron);
    let last = if time > 1 { time - 1 } else { 0 };
    cm.on_spike(time, last);
}

#[derive(Clone, Copy)]
pub struct PlasticityController {
    pub learning_enabled: bool,
    pub global_reward: FixedPoint,
    pub ltp_rate: FixedPoint,
    pub ltd_rate: FixedPoint,
    pub tau_plus: FixedPoint,
    pub tau_minus: FixedPoint,
    pub eligibility_decay: FixedPoint,
    pub calcium_enabled: bool,
    pub plateau_enabled: bool,
}

impl PlasticityController {
    pub const fn new_const() -> Self {
        Self {
            learning_enabled: true,
            global_reward: FixedPoint::ZERO,
            ltp_rate: FixedPoint::from_f32(0.01),
            ltd_rate: FixedPoint::from_f32(0.012),
            tau_plus: FixedPoint::from_f32(20.0),
            tau_minus: FixedPoint::from_f32(20.0),
            eligibility_decay: FixedPoint::from_f32(0.95),
            calcium_enabled: true,
            plateau_enabled: true,
        }
    }
}

pub static mut PLASTICITY_CTRL: MaybeUninit<PlasticityController> =
    MaybeUninit::new(PlasticityController::new_const());

#[inline(always)]
pub fn plasticity() -> &'static mut PlasticityController {
    unsafe { &mut *PLASTICITY_CTRL.as_mut_ptr() }
}

#[inline(always)]
pub fn plasticity_ref() -> &'static PlasticityController {
    unsafe { &*PLASTICITY_CTRL.as_ptr() }
}

const UNINIT_TRACE: MaybeUninit<EligibilityTrace> = MaybeUninit::uninit();
pub static mut ELIGIBILITY_TRACES: [MaybeUninit<EligibilityTrace>; crate::MAX_SYNAPSES] =
    [UNINIT_TRACE; crate::MAX_SYNAPSES];
pub static mut SPIKE_TIMES_PRE: [u32; crate::MAX_NEURONS] = [0; crate::MAX_NEURONS];
pub static mut SPIKE_TIMES_POST: [u32; crate::MAX_NEURONS] = [0; crate::MAX_NEURONS];
pub static mut LAST_PRE_TIME: [u32; crate::MAX_NEURONS] = [0; crate::MAX_NEURONS];
pub static mut LAST_POST_TIME: [u32; crate::MAX_NEURONS] = [0; crate::MAX_NEURONS];

#[derive(Clone, Copy)]
pub struct EligibilityTrace {
    pub trace: FixedPoint,
    pub decay: FixedPoint,
    pub last_update: u32,
}

impl EligibilityTrace {
    pub const fn new_const() -> Self {
        Self {
            trace: FixedPoint::ZERO,
            decay: FixedPoint::from_f32(0.95),
            last_update: 0,
        }
    }
}

pub fn on_pre_spike(pre: NeuronId, time: u32) {
    let idx = pre.index();
    unsafe {
        LAST_PRE_TIME[idx] = time;
        SPIKE_TIMES_PRE[idx] = time;
    }

    if plasticity_ref().calcium_enabled {
        update_calcium_on_spike(pre, time);
    }

    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse::synapse_ref(id);
        if s.pre == pre && s.plasticity_enabled {
            unsafe {
                let trace = &mut *ELIGIBILITY_TRACES[i as usize].as_mut_ptr();
                let dt = (time - trace.last_update) as i32;
                if dt > 0 {
                    let tau = plasticity_ref().tau_plus;
                    let decay = (FixedPoint::ONE - FixedPoint::ONE / tau).pow(dt as u32);
                    trace.trace *= decay;
                }
                if LAST_POST_TIME[idx] > 0 {
                    let delta = time - LAST_POST_TIME[idx];
                    if delta < 50 {
                        let delta_fp = FixedPoint::from_int(delta as i32);
                        let ratio = delta_fp / plasticity_ref().tau_plus;
                        let stdp = (-ratio).exp();

                        let calcium_mod = if plasticity_ref().calcium_enabled {
                            let cm = calcium_model(pre);
                            let (ltp_contrib, _) = cm.plasticity_direction();
                            if ltp_contrib > FixedPoint::ZERO {
                                FixedPoint::ONE + ltp_contrib
                            } else {
                                FixedPoint::from_f32(0.5)
                            }
                        } else {
                            FixedPoint::ONE
                        };

                        trace.trace += plasticity_ref().ltp_rate * stdp * calcium_mod;
                    }
                }
                trace.last_update = time;
            }
        }
    }
}

pub fn on_post_spike(post: NeuronId, time: u32) {
    let idx = post.index();
    unsafe {
        LAST_POST_TIME[idx] = time;
        SPIKE_TIMES_POST[idx] = time;
    }

    if plasticity_ref().calcium_enabled {
        update_calcium_on_spike(post, time);
    }

    let pp = plateau_potential(post);
    pp.update(time);

    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse::synapse_ref(id);
        if s.post == post && s.plasticity_enabled {
            unsafe {
                let trace = &mut *ELIGIBILITY_TRACES[i as usize].as_mut_ptr();
                let dt = (time - trace.last_update) as i32;
                if dt > 0 {
                    let tau = plasticity_ref().tau_minus;
                    let decay = (FixedPoint::ONE - FixedPoint::ONE / tau).pow(dt as u32);
                    trace.trace *= decay;
                }
                if LAST_PRE_TIME[post.index()] > 0 {
                    let delta = time - LAST_PRE_TIME[post.index()];
                    if delta < 50 {
                        let delta_fp = FixedPoint::from_int(delta as i32);
                        let ratio = delta_fp / plasticity_ref().tau_minus;
                        let stdp = (-ratio).exp();

                        let plateau_mod = if plasticity_ref().plateau_enabled && pp.active {
                            FixedPoint::ONE + pp.amplitude
                        } else {
                            FixedPoint::ONE
                        };

                        let calcium_mod = if plasticity_ref().calcium_enabled {
                            let cm = calcium_model(post);
                            let (_, ltd_contrib) = cm.plasticity_direction();
                            if ltd_contrib > FixedPoint::ZERO {
                                FixedPoint::ONE + ltd_contrib
                            } else {
                                FixedPoint::from_f32(0.5)
                            }
                        } else {
                            FixedPoint::ONE
                        };

                        trace.trace -= plasticity_ref().ltd_rate * stdp * plateau_mod * calcium_mod;
                    }
                }
                trace.last_update = time;
            }
        }
    }
}

pub fn apply_reward_modulation(reward: FixedPoint, time: u32) {
    let enable = plasticity_ref().learning_enabled;
    if !enable {
        return;
    }

    let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
    for i in 0..count as u16 {
        let id = SynapseId::new(i);
        let s = synapse::synapse(id);
        if !s.plasticity_enabled {
            continue;
        }

        unsafe {
            let trace = &*ELIGIBILITY_TRACES[i as usize].as_ptr();
            let dt = (time - trace.last_update) as i32;
            let decay = if dt > 0 {
                plasticity_ref().eligibility_decay.powi(dt)
            } else {
                FixedPoint::ONE
            };
            let effective_trace = trace.trace * decay;

            if effective_trace != FixedPoint::ZERO {
                let dw = reward * effective_trace;
                let w = s.weight.to_fixed() + dw;
                s.weight = crate::core::math::Weight::from_f32(w.to_f32());
                s.weight.0 = s.weight.0.clamp(-30000, 30000);
            }
        }
    }
}

pub fn decay_all_traces() {
    decay_traces_chunk(0, crate::MAX_SYNAPSES);
}

pub fn decay_traces_chunk(start_idx: usize, end_idx: usize) {
    let end = end_idx.min(crate::MAX_SYNAPSES);
    for i in start_idx..end {
        unsafe {
            let trace = &mut *ELIGIBILITY_TRACES[i].as_mut_ptr();
            trace.trace *= trace.decay;
        }
    }
}

pub fn reset_traces() {
    for i in 0..crate::MAX_SYNAPSES {
        unsafe {
            let trace = &mut *ELIGIBILITY_TRACES[i].as_mut_ptr();
            trace.trace = FixedPoint::ZERO;
            trace.last_update = 0;
        }
    }
}

pub fn modulate_rates(
    noradrenaline: FixedPoint,
    serotonin: FixedPoint,
    dopamine: FixedPoint,
    acetylcholine: FixedPoint,
) {
    let crisis_factor = FixedPoint::ONE + noradrenaline * FixedPoint::from_f32(2.0);
    let stability_factor = FixedPoint::ONE - serotonin * FixedPoint::from_f32(0.5);
    let reward_factor = FixedPoint::ONE + dopamine * FixedPoint::from_f32(1.0);
    let attention_factor = FixedPoint::ONE + acetylcholine * FixedPoint::from_f32(0.5);

    let ltp_mult = crisis_factor * stability_factor * reward_factor * attention_factor;
    let ltd_mult = crisis_factor * stability_factor;

    let ctrl = plasticity();
    ctrl.ltp_rate = FixedPoint::from_f32(0.01) * ltp_mult;
    ctrl.ltd_rate = FixedPoint::from_f32(0.012) * ltd_mult;
    ctrl.ltp_rate = ctrl
        .ltp_rate
        .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
    ctrl.ltd_rate = ctrl
        .ltd_rate
        .clamp(FixedPoint::from_f32(0.0001), FixedPoint::from_f32(0.1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::NeuronId;

    #[test]
    fn plasticity_controller_default() {
        let c = PlasticityController::new_const();
        assert!((c.ltp_rate.to_f32() - 0.01).abs() < 0.001);
        assert!((c.ltd_rate.to_f32() - 0.012).abs() < 0.001);
    }

    #[test]
    fn eligibility_trace_default() {
        let t = EligibilityTrace::new_const();
        assert_eq!(t.trace.to_f32(), 0.0);
        assert!((t.decay.to_f32() - 0.95).abs() < 0.001);
        assert_eq!(t.last_update, 0);
    }

    #[test]
    fn on_pre_spike_updates_trace() {
        let id = NeuronId::new(0);
        on_pre_spike(id, 100);
        unsafe {
            let pre = SPIKE_TIMES_PRE[id.index() as usize];
            assert_eq!(pre, 100);
        }
    }

    #[test]
    fn on_post_spike_updates_trace() {
        let id = NeuronId::new(0);
        on_post_spike(id, 200);
        unsafe {
            let post = SPIKE_TIMES_POST[id.index() as usize];
            assert_eq!(post, 200);
        }
    }

    #[test]
    fn decay_all_traces_decays_trace() {
        decay_traces_chunk(0, 10);
        decay_all_traces();
    }

    #[test]
    fn modulate_rates_changes_plasticity() {
        modulate_rates(
            FixedPoint::ONE,
            FixedPoint::ZERO,
            FixedPoint::from_f32(0.5),
            FixedPoint::from_f32(0.5),
        );
        let ctrl = plasticity();
        assert!(ctrl.ltp_rate.to_f32() > 0.005);
    }

    #[test]
    fn reset_traces_clears_trace_fields() {
        unsafe {
            let trace = &mut *ELIGIBILITY_TRACES[0].as_mut_ptr();
            trace.trace = FixedPoint::from_f32(0.5);
            trace.last_update = 100;
        }
        reset_traces();
        unsafe {
            let trace = &*ELIGIBILITY_TRACES[0].as_ptr();
            assert_eq!(trace.trace.to_f32(), 0.0);
            assert_eq!(trace.last_update, 0);
        }
    }

    #[test]
    fn calcium_model_default() {
        let cm = CalciumModel::new_const();
        assert_eq!(cm.concentration.to_f32(), 0.0);
        assert!((cm.ltp_threshold.to_f32() - 0.15).abs() < 0.001);
        assert!((cm.ltd_threshold.to_f32() - 0.05).abs() < 0.001);
    }

    #[test]
    fn calcium_model_on_spike_raises_concentration() {
        let mut cm = CalciumModel::new_const();
        cm.on_spike(10, 0);
        assert!(cm.concentration > FixedPoint::ZERO);
        let c1 = cm.concentration.to_f32();
        cm.on_spike(20, 10);
        assert!(cm.concentration.to_f32() > c1);
    }

    #[test]
    fn calcium_model_ltp_threshold() {
        let mut cm = CalciumModel::new_const();
        cm.concentration = FixedPoint::from_f32(0.2);
        let (ltp, ltd) = cm.plasticity_direction();
        assert!(ltp > FixedPoint::ZERO);
        assert_eq!(ltd, FixedPoint::ZERO);
    }

    #[test]
    fn calcium_model_ltd_threshold() {
        let mut cm = CalciumModel::new_const();
        cm.concentration = FixedPoint::from_f32(0.08);
        let (ltp, ltd) = cm.plasticity_direction();
        assert_eq!(ltp, FixedPoint::ZERO);
        assert!(ltd > FixedPoint::ZERO);
    }

    #[test]
    fn calcium_model_below_ltd_no_plasticity() {
        let mut cm = CalciumModel::new_const();
        cm.concentration = FixedPoint::from_f32(0.01);
        let (ltp, ltd) = cm.plasticity_direction();
        assert_eq!(ltp, FixedPoint::ZERO);
        assert_eq!(ltd, FixedPoint::ZERO);
    }

    #[test]
    fn plateau_potential_default() {
        let pp = PlateauPotential::new_const();
        assert!(!pp.active);
        assert_eq!(pp.duration_ms, 100);
    }

    #[test]
    fn plateau_potential_trigger_activates() {
        let mut pp = PlateauPotential::new_const();
        assert!(!pp.active);
        pp.trigger(50);
        assert!(pp.active);
        assert_eq!(pp.start_time, 50);
    }

    #[test]
    fn plateau_potential_expires_after_duration() {
        let mut pp = PlateauPotential::new_const();
        pp.trigger(10);
        assert!(pp.update(10));
        assert!(pp.update(50));
        assert!(pp.update(109));
        let still_active = pp.update(110);
        assert!(!still_active);
    }

    #[test]
    fn calcium_decay_over_time() {
        let mut cm = CalciumModel::new_const();
        cm.concentration = FixedPoint::from_f32(0.5);
        cm.decay(10);
        assert!(cm.concentration.to_f32() < 0.5);
        assert!(cm.concentration.to_f32() > 0.4);
    }

    #[test]
    fn plateau_increases_calcium_more() {
        let mut cm = CalciumModel::new_const();
        cm.on_spike(10, 0);
        let spike_conc = cm.concentration.to_f32();
        cm.concentration = FixedPoint::ZERO;
        cm.on_plateau(10, 0);
        let plateau_conc = cm.concentration.to_f32();
        assert!(plateau_conc > spike_conc);
    }

    #[test]
    fn trigger_plateau_integrates_globals() {
        let nid = NeuronId::new(0);
        trigger_plateau(nid, 100);
        unsafe {
            let pp = &*PLATEAU_POTENTIALS[0].as_ptr();
            let cm = &*CALCIUM_MODELS[0].as_ptr();
            assert!(pp.active);
            assert!(cm.concentration > FixedPoint::ZERO);
        }
    }
}
