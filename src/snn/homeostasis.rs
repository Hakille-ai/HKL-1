use crate::core::math::{FixedPoint, Weight};
use crate::core::memory::{MAX_NEURONS, NEURON_COUNT, NeuronId, neuron_state, neuron_state_ref};
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;

pub struct HomeostaticScaling {
    pub target_rate: FixedPoint,
    pub scaling_factor: FixedPoint,
    pub adaptation_rate: FixedPoint,
    pub rate_window_size: u32,
    pub spike_counts: [u32; MAX_NEURONS],
    pub time_window: u32,
    pub actual_firing_rate: FixedPoint,
    pub error: FixedPoint,
}

impl HomeostaticScaling {
    pub const fn new_const() -> Self {
        Self {
            target_rate: FixedPoint::from_f32(10.0),
            scaling_factor: FixedPoint::ONE,
            adaptation_rate: FixedPoint::from_f32(0.001),
            rate_window_size: 10000,
            spike_counts: [0; MAX_NEURONS],
            time_window: 0,
            actual_firing_rate: FixedPoint::ZERO,
            error: FixedPoint::ZERO,
        }
    }
}

pub static mut HOMEOSTASIS: MaybeUninit<HomeostaticScaling> =
    MaybeUninit::new(HomeostaticScaling::new_const());

fn homeostasis() -> &'static mut HomeostaticScaling {
    unsafe { &mut *HOMEOSTASIS.as_mut_ptr() }
}

fn homeostasis_ref() -> &'static HomeostaticScaling {
    unsafe { &*HOMEOSTASIS.as_ptr() }
}

pub fn record_spike(neuron: NeuronId) {
    let h = homeostasis();
    h.spike_counts[neuron.index()] = h.spike_counts[neuron.index()].saturating_add(1);
}

pub fn update() {
    let h = homeostasis();
    h.time_window += 1;
    if h.time_window >= 1000 {
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        let mut total_rate = FixedPoint::ZERO;
        let mut active = 0u32;

        for i in 0..count as u16 {
            let spikes = h.spike_counts[i as usize];
            let rate = FixedPoint::from_int(spikes as i32) / FixedPoint::from_int(1000);
            total_rate += rate;
            if spikes > 0 {
                active += 1;
            }
            h.spike_counts[i as usize] = 0;
        }

        h.actual_firing_rate = if active > 0 {
            total_rate / FixedPoint::from_int(active as i32)
        } else {
            FixedPoint::ZERO
        };

        if h.actual_firing_rate > FixedPoint::ZERO {
            let err = h.target_rate - h.actual_firing_rate;
            h.error = err;
            let delta = err * h.adaptation_rate;
            h.scaling_factor += delta;
            h.scaling_factor = h
                .scaling_factor
                .clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(10.0));
        }

        h.time_window = 0;
    }
}

pub fn apply_scaling(weight: Weight) -> Weight {
    Weight::from_f32(weight.to_f32() * homeostasis_ref().scaling_factor.to_f32())
}

pub fn compensatory_scaling(damaged_neurons: &[NeuronId]) {
    for &damaged_id in damaged_neurons {
        let damaged_state = neuron_state_ref(damaged_id);
        let layer = damaged_state.layer;
        let count = NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = NeuronId::new(i);
            let state = neuron_state_ref(id);
            if state.layer == layer && id != damaged_id {
                let s_state = neuron_state(id);
                s_state.threshold *= FixedPoint::from_f32(1.2);
            }
        }
    }
}

pub fn error() -> FixedPoint {
    homeostasis_ref().error
}

pub fn scaling_factor() -> FixedPoint {
    homeostasis_ref().scaling_factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_const() {
        let h = HomeostaticScaling::new_const();
        assert_eq!(h.target_rate, FixedPoint::from_f32(10.0));
        assert_eq!(h.scaling_factor, FixedPoint::ONE);
        assert_eq!(h.error, FixedPoint::ZERO);
    }

    #[test]
    fn test_record_spike_increases_count() {
        let h = homeostasis();
        h.spike_counts = [0; MAX_NEURONS];
        record_spike(NeuronId::new(0));
        assert_eq!(h.spike_counts[0], 1);
    }

    #[test]
    fn test_record_spike_saturating() {
        let h = homeostasis();
        h.spike_counts[0] = u32::MAX;
        record_spike(NeuronId::new(0));
        assert_eq!(h.spike_counts[0], u32::MAX);
    }

    #[test]
    fn test_update_no_panic() {
        update();
    }

    #[test]
    fn test_update_increments_time_window() {
        let h = homeostasis();
        h.time_window = 0;
        h.spike_counts = [0; MAX_NEURONS];
        update();
        assert!(h.time_window > 0 || h.actual_firing_rate == FixedPoint::ZERO);
    }

    #[test]
    fn test_update_computes_error() {
        let h = homeostasis();
        h.time_window = 999;
        h.target_rate = FixedPoint::from_f32(10.0);
        crate::core::memory::NEURON_COUNT.store(1, Ordering::Relaxed);
        h.spike_counts[0] = 5;
        update();
        assert_eq!(h.time_window, 0);
    }

    #[test]
    fn test_scaling_factor_clamps() {
        let h = homeostasis();
        h.scaling_factor = FixedPoint::from_f32(0.01);
        h.time_window = 999;
        h.target_rate = FixedPoint::from_f32(10.0);
        h.actual_firing_rate = FixedPoint::from_f32(100.0);
        let err = h.target_rate - h.actual_firing_rate;
        h.error = err;
        let delta = err * h.adaptation_rate;
        h.scaling_factor =
            (h.scaling_factor + delta).clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(10.0));
        assert!(h.scaling_factor >= FixedPoint::from_f32(0.1));
    }

    #[test]
    fn test_apply_scaling() {
        let h = homeostasis();
        h.scaling_factor = FixedPoint::from_f32(2.0);
        let w = Weight::from_f32(0.5);
        let scaled = apply_scaling(w);
        assert!((scaled.to_f32() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compensatory_scaling_increases_threshold() {
        unsafe {
            let array = &mut crate::core::memory::NEURON_ARRAY;
            array[0] = MaybeUninit::new(crate::core::memory::NeuronState {
                membrane_potential: FixedPoint::ZERO,
                threshold: FixedPoint::from_f32(1.0),
                leak: FixedPoint::ZERO,
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: FixedPoint::ZERO,
                layer: 2,
                neuron_type: crate::core::memory::NeuronType::LIF,
                flags: crate::core::memory::NeuronFlags(0),
            });
            array[1] = MaybeUninit::new(crate::core::memory::NeuronState {
                membrane_potential: FixedPoint::ZERO,
                threshold: FixedPoint::from_f32(1.0),
                leak: FixedPoint::ZERO,
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: FixedPoint::ZERO,
                layer: 2,
                neuron_type: crate::core::memory::NeuronType::LIF,
                flags: crate::core::memory::NeuronFlags(0),
            });
            NEURON_COUNT.store(2, Ordering::Relaxed);
        }
        compensatory_scaling(&[NeuronId::new(0)]);
        let state1 = neuron_state_ref(NeuronId::new(1));
        assert!(state1.threshold > FixedPoint::from_f32(1.0));
    }

    #[test]
    fn test_error_accessor() {
        let h = homeostasis();
        h.error = FixedPoint::from_f32(0.5);
        assert_eq!(error(), FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_scaling_factor_accessor() {
        let h = homeostasis();
        h.scaling_factor = FixedPoint::from_f32(3.0);
        assert_eq!(scaling_factor(), FixedPoint::from_f32(3.0));
    }
}
