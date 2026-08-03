use crate::core::math::FixedPoint;
use crate::core::memory::{NeuronId, neuron_state};

const BODY_MODEL_SLOTS: usize = 64;

#[derive(Clone, Copy)]
pub struct BodyModelEntry {
    pub motor_id: u8,
    pub predicted_feedback: FixedPoint,
    pub actual_feedback_sum: FixedPoint,
    pub sample_count: u16,
    pub learned_weight: FixedPoint,
    pub last_error: FixedPoint,
}

impl BodyModelEntry {
    pub const fn empty() -> Self {
        Self {
            motor_id: 0,
            predicted_feedback: FixedPoint::ZERO,
            actual_feedback_sum: FixedPoint::ZERO,
            sample_count: 0,
            learned_weight: FixedPoint::from_f32(0.5),
            last_error: FixedPoint::ZERO,
        }
    }
}

pub struct Proprioception {
    pub efference_copy: [FixedPoint; 256],
    pub predicted_feedback: [FixedPoint; 256],
    pub actual_feedback: [FixedPoint; 256],
    pub prediction_error: FixedPoint,
    pub correction_active: bool,
    pub body_model: [BodyModelEntry; BODY_MODEL_SLOTS],
    pub body_model_count: u16,
    pub last_correction_time: u32,
    pub correction_cooldown: u32,
    pub smooth_correction_factor: FixedPoint,
}

impl Proprioception {
    pub const fn new() -> Self {
        Self {
            efference_copy: [FixedPoint::ZERO; 256],
            predicted_feedback: [FixedPoint::ZERO; 256],
            actual_feedback: [FixedPoint::ZERO; 256],
            prediction_error: FixedPoint::ZERO,
            correction_active: false,
            body_model: [BodyModelEntry::empty(); BODY_MODEL_SLOTS],
            body_model_count: 0,
            last_correction_time: 0,
            correction_cooldown: 0,
            smooth_correction_factor: FixedPoint::from_f32(0.3),
        }
    }

    fn find_body_model_entry(&self, motor_id: u8) -> Option<usize> {
        (0..self.body_model_count as usize).find(|&i| self.body_model[i].motor_id == motor_id)
    }

    pub fn record_efference(&mut self, motor_id: u8, predicted: FixedPoint) {
        let idx = motor_id as usize;
        self.efference_copy[idx] = predicted;
        let body_pred = self
            .find_body_model_entry(motor_id)
            .map(|entry| predicted * self.body_model[entry].learned_weight);
        if let Some(bp) = body_pred {
            self.predicted_feedback[idx] = bp;
        } else {
            self.predicted_feedback[idx] = predicted;
        }
    }

    pub fn set_predicted_feedback(&mut self, motor_id: u8, pred: FixedPoint) {
        self.predicted_feedback[motor_id as usize] = pred;
    }

    pub fn record_actual_feedback(&mut self, motor_id: u8, actual: FixedPoint) {
        let idx = motor_id as usize;
        self.actual_feedback[idx] = actual;

        let predicted = self.predicted_feedback[idx];
        let error = (predicted - actual).abs();
        self.prediction_error = error;

        let efference = self.efference_copy[idx];
        let entry_idx = if let Some(e) = self.find_body_model_entry(motor_id) {
            e
        } else {
            let n = self.body_model_count as usize;
            if n < BODY_MODEL_SLOTS {
                self.body_model_count = (n + 1) as u16;
                n
            } else {
                let mut worst_idx = 0;
                let mut max_error = self.body_model[0].last_error;
                for i in 1..BODY_MODEL_SLOTS {
                    if self.body_model[i].last_error > max_error {
                        max_error = self.body_model[i].last_error;
                        worst_idx = i;
                    }
                }
                self.body_model[worst_idx] = BodyModelEntry::empty();
                worst_idx
            }
        };

        let entry = &mut self.body_model[entry_idx];
        entry.motor_id = motor_id;
        entry.actual_feedback_sum += actual;
        entry.sample_count += 1;

        let alpha = FixedPoint::from_f32(0.1);
        let expected = entry.actual_feedback_sum / FixedPoint::from_int(entry.sample_count as i32);
        let weight_error = expected - entry.learned_weight * efference;
        entry.learned_weight += alpha * weight_error;
        entry.learned_weight = entry
            .learned_weight
            .clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(2.0));
        entry.last_error = error;

        let thresh = FixedPoint::from_f32(0.2);
        if error > thresh && self.correction_cooldown == 0 {
            self.correction_active = true;
            self.correction_cooldown = 10;
            self.apply_correction(motor_id, error);
        }
    }

    fn apply_correction(&mut self, motor_id: u8, error: FixedPoint) {
        let idx = motor_id as usize;
        let correction = error * self.smooth_correction_factor;

        for i in
            0..crate::core::memory::NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed) as u16
        {
            let id = NeuronId::new(i);
            let state = neuron_state(id);
            if state.layer == 4 && id.index() == idx {
                state.membrane_potential += correction * FixedPoint::from_f32(0.5);
                state.bias_current += correction * FixedPoint::from_f32(0.1);
            }
        }

        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline += error * FixedPoint::from_f32(0.2);
        nm.acetylcholine += error * FixedPoint::from_f32(0.1);
        nm.noradrenaline = nm.noradrenaline.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        nm.acetylcholine = nm.acetylcholine.clamp(FixedPoint::ZERO, FixedPoint::ONE);

        self.last_correction_time = unsafe { crate::core::time::METABOLIC_CLOCK.ticks_1khz() };
        self.correction_cooldown = 10;
    }

    pub fn update(&mut self) {
        if self.correction_cooldown > 0 {
            self.correction_cooldown -= 1;
        }
        if self.correction_cooldown == 0 {
            self.correction_active = false;
        }

        let decay = FixedPoint::from_f32(0.001);
        for i in 0..self.body_model_count as usize {
            let entry = &mut self.body_model[i];
            if entry.sample_count > 100 {
                entry.sample_count = 50;
                entry.actual_feedback_sum *= FixedPoint::from_f32(0.5);
            }
            entry.last_error *= FixedPoint::ONE - decay;
        }
    }

    pub fn reset(&mut self) {
        self.efference_copy = [FixedPoint::ZERO; 256];
        self.predicted_feedback = [FixedPoint::ZERO; 256];
        self.actual_feedback = [FixedPoint::ZERO; 256];
        self.prediction_error = FixedPoint::ZERO;
        self.correction_active = false;
        self.body_model_count = 0;
        self.correction_cooldown = 0;
        for i in 0..BODY_MODEL_SLOTS {
            self.body_model[i] = BodyModelEntry::empty();
        }
    }

    pub fn body_model_accuracy(&self) -> FixedPoint {
        let mut total_error = FixedPoint::ZERO;
        let count = self.body_model_count.max(1);
        for i in 0..self.body_model_count as usize {
            total_error += self.body_model[i].last_error;
        }
        FixedPoint::ONE - (total_error / FixedPoint::from_int(count as i32))
    }
}

pub static mut PROPRIOCEPTION: Proprioception = Proprioception::new();

pub fn proprioception() -> &'static mut Proprioception {
    unsafe { &mut PROPRIOCEPTION }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proprioception_new() {
        let p = Proprioception::new();
        assert_eq!(p.prediction_error, FixedPoint::ZERO);
        assert!(!p.correction_active);
        assert_eq!(p.body_model_count, 0);
    }

    #[test]
    fn test_record_efference() {
        let mut p = Proprioception::new();
        p.record_efference(5, FixedPoint::from_f32(0.8));
        assert_eq!(p.efference_copy[5], FixedPoint::from_f32(0.8));
    }

    #[test]
    fn test_record_actual_feedback_creates_body_model() {
        let mut p = Proprioception::new();
        p.record_efference(3, FixedPoint::from_f32(0.5));
        p.record_actual_feedback(3, FixedPoint::from_f32(0.45));
        assert_eq!(p.body_model_count, 1);
        assert_eq!(p.body_model[0].motor_id, 3);
    }

    #[test]
    fn test_body_model_learns() {
        let mut p = Proprioception::new();
        for _ in 0..20 {
            p.record_efference(1, FixedPoint::from_f32(0.8));
            p.record_actual_feedback(1, FixedPoint::from_f32(0.75));
        }
        assert!(p.body_model[0].learned_weight >= FixedPoint::from_f32(0.1));
        assert!(p.body_model[0].sample_count >= 20);
    }

    #[test]
    fn test_prediction_error_computed() {
        let mut p = Proprioception::new();
        p.record_efference(0, FixedPoint::from_f32(1.0));
        p.set_predicted_feedback(0, FixedPoint::from_f32(1.0));
        p.record_actual_feedback(0, FixedPoint::from_f32(0.5));
        assert!(p.prediction_error > FixedPoint::ZERO);
    }

    #[test]
    fn test_correction_triggers_on_large_error() {
        let mut p = Proprioception::new();
        p.record_efference(10, FixedPoint::from_f32(1.0));
        p.set_predicted_feedback(10, FixedPoint::from_f32(1.0));
        assert_eq!(p.predicted_feedback[10], FixedPoint::from_f32(1.0));
        assert_eq!(p.correction_cooldown, 0);
        assert!(!p.correction_active);
        p.record_actual_feedback(10, FixedPoint::ZERO);
        assert!(
            p.prediction_error > FixedPoint::from_f32(0.2),
            "error should be large"
        );
        assert!(
            p.correction_active || p.last_correction_time > 0,
            "correction should trigger"
        );
    }

    #[test]
    fn test_update_cooldown() {
        let mut p = Proprioception::new();
        p.correction_cooldown = 5;
        p.update();
        assert_eq!(p.correction_cooldown, 4);
    }

    #[test]
    fn test_body_model_accuracy() {
        let p = Proprioception::new();
        let acc = p.body_model_accuracy();
        assert!(acc >= FixedPoint::ZERO);
        assert!(acc <= FixedPoint::ONE);
    }

    #[test]
    fn test_reset_clears() {
        let mut p = Proprioception::new();
        p.record_efference(0, FixedPoint::from_f32(1.0));
        p.record_actual_feedback(0, FixedPoint::from_f32(0.9));
        p.reset();
        assert_eq!(p.body_model_count, 0);
        assert!(!p.correction_active);
    }

    #[test]
    fn test_body_model_slot_eviction() {
        let mut p = Proprioception::new();
        for i in 0..BODY_MODEL_SLOTS as u8 {
            p.record_efference(i, FixedPoint::from_f32(1.0));
            p.record_actual_feedback(i, FixedPoint::from_f32(1.0));
        }
        assert_eq!(p.body_model_count, BODY_MODEL_SLOTS as u16);

        p.body_model[10].last_error = FixedPoint::from_f32(5.0);

        p.record_efference(100, FixedPoint::from_f32(1.0));
        p.record_actual_feedback(100, FixedPoint::from_f32(1.0));

        assert_eq!(p.body_model_count, BODY_MODEL_SLOTS as u16);
        assert_eq!(p.body_model[10].motor_id, 100);
    }

    #[test]
    fn test_reset_clears_body_model_entries() {
        let mut p = Proprioception::new();
        p.record_efference(1, FixedPoint::from_f32(1.0));
        p.record_actual_feedback(1, FixedPoint::from_f32(0.5));

        p.body_model[0].learned_weight = FixedPoint::from_f32(1.5);

        p.reset();

        assert_eq!(p.body_model_count, 0);
        assert_eq!(p.body_model[0].learned_weight, FixedPoint::from_f32(0.5));
    }
}
