//! Metacognitive Auto-Tuner & Cognitive Self-Monitoring Module (`src/cognition/metacognition.rs`).
//! Evaluates real-time learning stability, curiosity, and entropy to dynamically
//! auto-tune learning rate scales, surrogate gradient slopes, and LIF neuron thresholds.
#![cfg(feature = "hkl2")]

use crate::core::math::FixedPoint;

/// Metacognitive Tuning Recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningAction {
    Maintain,
    IncreaseLearningScale,
    DecreaseLearningScale,
    SharpenSurrogateGradient,
    SmoothSurrogateGradient,
    AdjustNeuronThreshold,
}

/// Metacognitive Status Report
#[derive(Debug, Clone, PartialEq)]
pub struct MetacognitiveReport {
    pub learning_scale: FixedPoint,
    pub surrogate_slope: FixedPoint,
    pub threshold_adj: FixedPoint,
    pub action: TuningAction,
    pub stability_score: f32,
    pub entropy_status: &'static str,
}

/// Metacognitive Self-Monitoring Auto-Tuner
pub struct MetacognitiveAutoTuner {
    pub learning_scale: FixedPoint,
    pub surrogate_slope: FixedPoint,
    pub threshold_adj: FixedPoint,
    pub loss_history: [FixedPoint; 16],
    pub loss_count: usize,
    pub consecutive_flat_steps: u32,
}

impl MetacognitiveAutoTuner {
    pub fn new() -> Self {
        Self {
            learning_scale: FixedPoint::ONE,
            surrogate_slope: FixedPoint::from_f32(1.0),
            threshold_adj: FixedPoint::ZERO,
            loss_history: [FixedPoint::ZERO; 16],
            loss_count: 0,
            consecutive_flat_steps: 0,
        }
    }

    /// Record new loss step and evaluate metacognitive adjustments
    pub fn record_and_evaluate(
        &mut self,
        current_loss: FixedPoint,
        curiosity: FixedPoint,
        boredom: f32,
    ) -> MetacognitiveReport {
        // Record loss history
        if self.loss_count < 16 {
            self.loss_history[self.loss_count] = current_loss;
            self.loss_count += 1;
        } else {
            for i in 0..15 {
                self.loss_history[i] = self.loss_history[i + 1];
            }
            self.loss_history[15] = current_loss;
        }

        // Calculate loss variance across history
        let mut sum = 0.0f32;
        for i in 0..self.loss_count {
            sum += self.loss_history[i].to_f32();
        }
        let mean = if self.loss_count > 0 {
            sum / self.loss_count as f32
        } else {
            0.0
        };

        let mut var_sum = 0.0f32;
        for i in 0..self.loss_count {
            let diff = self.loss_history[i].to_f32() - mean;
            var_sum += diff * diff;
        }
        let variance = if self.loss_count > 0 {
            var_sum / self.loss_count as f32
        } else {
            0.0
        };

        // Evaluate action based on variance and boredom
        let mut action = TuningAction::Maintain;

        if boredom > 0.8 {
            action = TuningAction::IncreaseLearningScale;
            self.learning_scale =
                (self.learning_scale * FixedPoint::from_f32(1.1)).min(FixedPoint::from_f32(5.0));
        } else if variance < 0.0001 && self.loss_count >= 8 {
            self.consecutive_flat_steps += 1;
            if self.consecutive_flat_steps > 5 {
                action = TuningAction::SharpenSurrogateGradient;
                self.surrogate_slope = (self.surrogate_slope * FixedPoint::from_f32(1.2))
                    .min(FixedPoint::from_f32(10.0));
            }
        } else if variance > 10.0 {
            action = TuningAction::DecreaseLearningScale;
            self.learning_scale =
                (self.learning_scale * FixedPoint::from_f32(0.8)).max(FixedPoint::from_f32(0.1));
            self.consecutive_flat_steps = 0;
        } else {
            self.consecutive_flat_steps = 0;
        }

        let entropy_str = if curiosity.to_f32() > 0.7 {
            "High Exploration"
        } else if variance < 0.001 {
            "Stable Convergence"
        } else {
            "Active Learning"
        };

        MetacognitiveReport {
            learning_scale: self.learning_scale,
            surrogate_slope: self.surrogate_slope,
            threshold_adj: self.threshold_adj,
            action,
            stability_score: 1.0 / (1.0 + variance),
            entropy_status: entropy_str,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognitive_tuner_initial() {
        let tuner = MetacognitiveAutoTuner::new();
        assert_eq!(tuner.learning_scale, FixedPoint::ONE);
        assert_eq!(tuner.surrogate_slope, FixedPoint::from_f32(1.0));
    }

    #[test]
    fn test_metacognitive_tuner_boredom_increases_learning_scale() {
        let mut tuner = MetacognitiveAutoTuner::new();
        let report =
            tuner.record_and_evaluate(FixedPoint::from_f32(5.0), FixedPoint::from_f32(0.5), 0.9);

        assert_eq!(report.action, TuningAction::IncreaseLearningScale);
        assert!(tuner.learning_scale > FixedPoint::ONE);
    }

    #[test]
    fn test_metacognitive_tuner_high_variance_decreases_scale() {
        let mut tuner = MetacognitiveAutoTuner::new();
        tuner.record_and_evaluate(FixedPoint::from_f32(1.0), FixedPoint::from_f32(0.5), 0.1);
        let report =
            tuner.record_and_evaluate(FixedPoint::from_f32(50.0), FixedPoint::from_f32(0.5), 0.1);

        assert_eq!(report.action, TuningAction::DecreaseLearningScale);
        assert!(tuner.learning_scale < FixedPoint::ONE);
    }
}
