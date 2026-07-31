use crate::core::math::FixedPoint;

pub struct NeuromodulationCalibration {
    pub pred_error_mean: FixedPoint,
    pub pred_error_var: FixedPoint,
    pub reward_mean: FixedPoint,
    pub novelty_mean: FixedPoint,
    pub sample_count: u32,
    pub decay_rate_mean: FixedPoint,
    pub sensitivity_ltp: FixedPoint,
    pub sensitivity_ltd: FixedPoint,
    pub volatility_estimate: FixedPoint,
}

impl NeuromodulationCalibration {
    pub const fn new() -> Self {
        Self {
            pred_error_mean: FixedPoint::ZERO,
            pred_error_var: FixedPoint::ZERO,
            reward_mean: FixedPoint::from_f32(0.1),
            novelty_mean: FixedPoint::from_f32(0.05),
            sample_count: 0,
            decay_rate_mean: FixedPoint::from_f32(0.001),
            sensitivity_ltp: FixedPoint::ONE,
            sensitivity_ltd: FixedPoint::ONE,
            volatility_estimate: FixedPoint::ZERO,
        }
    }

    pub fn update(&mut self, pred_error: FixedPoint, reward: FixedPoint, novelty: FixedPoint) {
        let n = self.sample_count;
        let alpha = FixedPoint::from_f32(0.01);

        let error_diff = pred_error - self.pred_error_mean;
        self.pred_error_mean += alpha * error_diff;
        let var_diff = error_diff.abs() - self.pred_error_var;
        self.pred_error_var += alpha * var_diff;
        self.pred_error_var = self.pred_error_var.max(FixedPoint::ZERO);

        let reward_diff = reward - self.reward_mean;
        self.reward_mean += alpha * reward_diff;

        let novelty_diff = novelty - self.novelty_mean;
        self.novelty_mean += alpha * novelty_diff;

        self.volatility_estimate = self.pred_error_var * FixedPoint::from_f32(2.0)
            + (self.pred_error_mean * FixedPoint::from_f32(0.5));
        self.volatility_estimate = self
            .volatility_estimate
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);

        self.sample_count = n.saturating_add(1);

        let vol = self.volatility_estimate;
        let decay_min = FixedPoint::from_f32(0.0005);
        let decay_max = FixedPoint::from_f32(0.01);
        self.decay_rate_mean = decay_min + (decay_max - decay_min) * (FixedPoint::ONE - vol);
        self.decay_rate_mean = self.decay_rate_mean.clamp(decay_min, decay_max);

        let sens_min = FixedPoint::from_f32(0.5);
        let sens_max = FixedPoint::from_f32(2.0);
        self.sensitivity_ltp = sens_min + (sens_max - sens_min) * self.reward_mean;
        self.sensitivity_ltp = self.sensitivity_ltp.clamp(sens_min, sens_max);
        self.sensitivity_ltd =
            sens_min + (sens_max - sens_min) * (FixedPoint::ONE - self.reward_mean);
        self.sensitivity_ltd = self.sensitivity_ltd.clamp(sens_min, sens_max);
    }

    pub fn adaptive_decay_rate(&self) -> FixedPoint {
        self.decay_rate_mean
    }
}

impl Default for NeuromodulationCalibration {
    fn default() -> Self {
        Self::new()
    }
}

pub static mut NM_CALIBRATION: NeuromodulationCalibration = NeuromodulationCalibration::new();

pub fn nm_calibration() -> &'static mut NeuromodulationCalibration {
    unsafe { &mut NM_CALIBRATION }
}

pub fn calibrate_neuromodulators(now: u32) {
    if now % 50 != 0 {
        return;
    }
    let cal = nm_calibration();
    let pe = cal.pred_error_mean;
    let reward = cal.reward_mean;
    let novelty = cal.novelty_mean;
    let vol = cal.volatility_estimate;

    // DA: reward-driven, boosted by surprise
    let target_da = if pe > FixedPoint::from_f32(0.1) {
        FixedPoint::from_f32(0.5) + pe * FixedPoint::from_f32(2.0)
    } else {
        reward * FixedPoint::from_f32(0.8)
    };

    // NA: volatility-driven (high vol → high alertness)
    let target_na = vol * FixedPoint::from_f32(0.9);

    // 5-HT: stability-driven (low vol + low pe → high well-being)
    let stability =
        FixedPoint::ONE - (vol * FixedPoint::from_f32(0.5) + pe * FixedPoint::from_f32(0.5));
    let target_ht = stability.clamp(FixedPoint::ZERO, FixedPoint::from_f32(0.8));

    // ACh: novelty-driven, attention
    let target_ach =
        (novelty + pe * FixedPoint::from_f32(0.3)).clamp(FixedPoint::ZERO, FixedPoint::ONE);

    let nm = crate::snn::neuron::neuromodulators();
    nm.dopamine = (nm.dopamine * FixedPoint::from_f32(0.9) + target_da * FixedPoint::from_f32(0.1))
        .clamp(FixedPoint::ZERO, FixedPoint::ONE);
    nm.noradrenaline = (nm.noradrenaline * FixedPoint::from_f32(0.9)
        + target_na * FixedPoint::from_f32(0.1))
    .clamp(FixedPoint::ZERO, FixedPoint::ONE);
    nm.serotonin = (nm.serotonin * FixedPoint::from_f32(0.9)
        + target_ht * FixedPoint::from_f32(0.1))
    .clamp(FixedPoint::ZERO, FixedPoint::ONE);
    nm.acetylcholine = (nm.acetylcholine * FixedPoint::from_f32(0.9)
        + target_ach * FixedPoint::from_f32(0.1))
    .clamp(FixedPoint::ZERO, FixedPoint::ONE);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_new() {
        let c = NeuromodulationCalibration::new();
        assert_eq!(c.pred_error_mean, FixedPoint::ZERO);
        assert_eq!(c.sample_count, 0);
    }

    #[test]
    fn test_calibration_update() {
        let mut c = NeuromodulationCalibration::new();
        c.update(
            FixedPoint::from_f32(0.1),
            FixedPoint::from_f32(0.3),
            FixedPoint::from_f32(0.05),
        );
        assert!(c.pred_error_mean > FixedPoint::ZERO);
        assert!(c.reward_mean > FixedPoint::ZERO);
        assert_eq!(c.sample_count, 1);
    }

    #[test]
    fn test_adaptive_decay_rate_bounded() {
        let c = NeuromodulationCalibration::new();
        let rate = c.adaptive_decay_rate();
        assert!(rate >= FixedPoint::from_f32(0.0005));
        assert!(rate <= FixedPoint::from_f32(0.01));
    }

    #[test]
    fn test_sensitivity_ltp_increases_with_reward() {
        let mut c = NeuromodulationCalibration::new();
        c.update(
            FixedPoint::from_f32(0.05),
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.1),
        );
        assert!(c.sensitivity_ltp > FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_volatility_estimate_bounded() {
        let mut c = NeuromodulationCalibration::new();
        for _ in 0..100 {
            c.update(
                FixedPoint::from_f32(0.2),
                FixedPoint::from_f32(0.1),
                FixedPoint::from_f32(0.05),
            );
        }
        assert!(c.volatility_estimate >= FixedPoint::ZERO);
        assert!(c.volatility_estimate <= FixedPoint::ONE);
    }

    #[test]
    fn test_calibrate_all_nm() {
        // Initialize GLOBAL_NEUROMODULATORS
        unsafe {
            let nm = &mut *crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            nm.dopamine = FixedPoint::from_f32(0.5);
            nm.noradrenaline = FixedPoint::from_f32(0.5);
            nm.serotonin = FixedPoint::from_f32(0.5);
            nm.acetylcholine = FixedPoint::from_f32(0.5);
        }
        // Run calibration with high prediction error + high novelty
        let cal = nm_calibration();
        cal.pred_error_mean = FixedPoint::from_f32(0.3);
        cal.reward_mean = FixedPoint::from_f32(0.1);
        cal.novelty_mean = FixedPoint::from_f32(0.4);
        cal.volatility_estimate = FixedPoint::from_f32(0.6);

        calibrate_neuromodulators(50);

        unsafe {
            let nm = &*crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            assert!(nm.dopamine >= FixedPoint::ZERO && nm.dopamine <= FixedPoint::ONE);
            assert!(nm.noradrenaline >= FixedPoint::ZERO && nm.noradrenaline <= FixedPoint::ONE);
            assert!(nm.serotonin >= FixedPoint::ZERO && nm.serotonin <= FixedPoint::ONE);
            assert!(nm.acetylcholine >= FixedPoint::ZERO && nm.acetylcholine <= FixedPoint::ONE);
        }
    }

    #[test]
    fn test_calibrate_na_high_volatility() {
        unsafe {
            let nm = &mut *crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            nm.noradrenaline = FixedPoint::ZERO;
        }
        let cal = nm_calibration();
        cal.pred_error_mean = FixedPoint::from_f32(0.1);
        cal.reward_mean = FixedPoint::from_f32(0.1);
        cal.novelty_mean = FixedPoint::ZERO;
        cal.volatility_estimate = FixedPoint::from_f32(0.9);

        calibrate_neuromodulators(50);

        unsafe {
            let nm = &*crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            assert!(nm.noradrenaline > FixedPoint::ZERO);
        }
    }

    #[test]
    fn test_calibrate_ach_high_novelty() {
        unsafe {
            let nm = &mut *crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            nm.acetylcholine = FixedPoint::ZERO;
        }
        let cal = nm_calibration();
        cal.pred_error_mean = FixedPoint::ZERO;
        cal.reward_mean = FixedPoint::ZERO;
        cal.novelty_mean = FixedPoint::from_f32(0.8);
        cal.volatility_estimate = FixedPoint::ZERO;

        calibrate_neuromodulators(50);

        unsafe {
            let nm = &*crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            assert!(nm.acetylcholine > FixedPoint::ZERO);
        }
    }

    #[test]
    fn test_calibrate_ht_high_stability() {
        unsafe {
            let nm = &mut *crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            nm.serotonin = FixedPoint::ZERO;
        }
        let cal = nm_calibration();
        cal.pred_error_mean = FixedPoint::from_f32(0.01);
        cal.reward_mean = FixedPoint::from_f32(0.5);
        cal.novelty_mean = FixedPoint::ZERO;
        cal.volatility_estimate = FixedPoint::from_f32(0.05);

        calibrate_neuromodulators(50);

        unsafe {
            let nm = &*crate::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
            assert!(nm.serotonin > FixedPoint::ZERO);
        }
    }
}
