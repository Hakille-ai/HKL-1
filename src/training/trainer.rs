//! Trainer module for HKL-2 model training using e-prop and Spiking Cross-Entropy Loss.

use crate::core::math::FixedPoint;
use crate::embedding::spike_embedding::VOCAB_SIZE;
use crate::learning::eprop::EpropEngine;
use crate::learning::loss::SpikingCrossEntropyLoss;
use crate::transformer::attention::MAX_SEQ_LEN;
use crate::transformer::backbone::SpikingTransformer;
use alloc::vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainStepStatus {
    Empty,
    Truncated,
    Complete,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainStepReport {
    pub loss: FixedPoint,
    pub tokens_seen: usize,
    pub saturated_losses: usize,
    pub invalid_inputs: usize,
    pub invalid_targets: usize,
    pub status: TrainStepStatus,
}

pub struct Trainer {
    pub model: SpikingTransformer,
    pub eprop: EpropEngine,
    pub step_count: u64,
}

impl Trainer {
    pub fn new(num_layers: usize) -> Self {
        let mut model = SpikingTransformer::new(num_layers);
        model.init_random(42);
        Self {
            model,
            eprop: EpropEngine::new(),
            step_count: 0,
        }
    }

    /// Perform a single training step on a sequence pair (inputs, targets)
    /// Returns average loss over sequence
    pub fn train_step(&mut self, inputs: &[u16], targets: &[u16]) -> FixedPoint {
        self.train_step_report(inputs, targets).loss
    }

    pub fn reset_model_state(&mut self) {
        self.model.reset_state();
    }

    /// Perform a bounded training step and return telemetry for safety monitors.
    pub fn train_step_report(&mut self, inputs: &[u16], targets: &[u16]) -> TrainStepReport {
        self.run_step(inputs, targets, FixedPoint::ONE, true, true)
    }

    /// Evaluate a bounded training step without applying adaptive weight updates.
    pub fn preview_step_report(&mut self, inputs: &[u16], targets: &[u16]) -> TrainStepReport {
        self.reset_model_state();
        let report = self.run_step(inputs, targets, FixedPoint::ZERO, false, false);
        self.reset_model_state();
        report
    }

    /// Perform a bounded training step with a guard-provided learning-rate scale.
    pub fn train_step_report_scaled(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        lr_scale: FixedPoint,
    ) -> TrainStepReport {
        self.run_step(inputs, targets, lr_scale.max(FixedPoint::ZERO), true, true)
    }

    fn run_step(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        lr_scale: FixedPoint,
        apply_updates: bool,
        count_step: bool,
    ) -> TrainStepReport {
        let requested_len = inputs.len().min(targets.len());
        let bounded_len = requested_len.min(MAX_SEQ_LEN);
        if requested_len == 0 {
            return TrainStepReport {
                loss: FixedPoint::ZERO,
                tokens_seen: 0,
                saturated_losses: 0,
                invalid_inputs: 0,
                invalid_targets: 0,
                status: TrainStepStatus::Empty,
            };
        }

        // 1. Forward pass: logits [seq_len][VOCAB_SIZE]
        let logits = self.model.forward(&inputs[..bounded_len]);
        let active_len = logits.len().min(bounded_len);
        if active_len == 0 {
            return TrainStepReport {
                loss: FixedPoint::ZERO,
                tokens_seen: 0,
                saturated_losses: 0,
                invalid_inputs: 0,
                invalid_targets: 0,
                status: TrainStepStatus::Empty,
            };
        }

        let mut total_loss = FixedPoint::ZERO;
        let mut learning_signals = vec![FixedPoint::ZERO; VOCAB_SIZE];
        let mut saturated_losses = 0;
        let mut invalid_inputs = 0;
        let mut invalid_targets = 0;
        let lr = self.eprop.lr * lr_scale;

        // 2. Compute loss and learning signals for each token position
        for pos in 0..active_len {
            let input_invalid = inputs[pos] as usize >= VOCAB_SIZE;
            let target_idx = targets[pos] as usize;
            if input_invalid {
                invalid_inputs += 1;
            }
            let loss = SpikingCrossEntropyLoss::compute_loss(&logits[pos], target_idx, VOCAB_SIZE);
            if target_idx >= VOCAB_SIZE {
                invalid_targets += 1;
            }
            if loss == FixedPoint::MAX {
                saturated_losses += 1;
            }
            total_loss = total_loss + loss;

            if input_invalid || target_idx >= VOCAB_SIZE {
                continue;
            }

            if !apply_updates || lr == FixedPoint::ZERO {
                continue;
            }

            SpikingCrossEntropyLoss::compute_learning_signals(
                &logits[pos],
                target_idx,
                &mut learning_signals,
            );

            // Apply e-prop updates to output head weights based on learning signal
            for v in 0..VOCAB_SIZE {
                let signal = learning_signals[v];
                if signal != FixedPoint::ZERO {
                    self.model.head.bias[v] = self.model.head.bias[v] - lr * signal;
                }
            }
        }

        if count_step {
            self.step_count += 1;
        }
        let status = if active_len < requested_len {
            TrainStepStatus::Truncated
        } else {
            TrainStepStatus::Complete
        };

        TrainStepReport {
            loss: total_loss / FixedPoint::from_int(active_len as i32),
            tokens_seen: active_len,
            saturated_losses,
            invalid_inputs,
            invalid_targets,
            status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::attention::MAX_SEQ_LEN;

    #[test]
    fn test_trainer_step() {
        let mut trainer = Trainer::new(1);
        let inputs = [10u16, 20u16, 30u16];
        let targets = [20u16, 30u16, 40u16];

        let loss = trainer.train_step(&inputs, &targets);
        assert!(loss > FixedPoint::ZERO);
        assert_eq!(trainer.step_count, 1);
    }

    #[test]
    fn test_trainer_step_clamps_oversized_sequence() {
        let mut trainer = Trainer::new(1);
        let inputs = alloc::vec![10u16; MAX_SEQ_LEN + 5];
        let targets = alloc::vec![20u16; MAX_SEQ_LEN + 5];

        let report = trainer.train_step_report(&inputs, &targets);

        assert!(report.loss > FixedPoint::ZERO);
        assert_eq!(report.tokens_seen, MAX_SEQ_LEN);
        assert_eq!(report.status, TrainStepStatus::Truncated);
        assert_eq!(trainer.step_count, 1);
    }

    #[test]
    fn test_trainer_step_report_empty_sequence() {
        let mut trainer = Trainer::new(1);
        let report = trainer.train_step_report(&[], &[]);

        assert_eq!(report.tokens_seen, 0);
        assert_eq!(report.status, TrainStepStatus::Empty);
        assert_eq!(trainer.step_count, 0);
    }

    #[test]
    fn test_trainer_step_report_counts_invalid_targets_without_update() {
        let mut trainer = Trainer::new(1);
        let inputs = [10u16, 20u16];
        let targets = [VOCAB_SIZE as u16, 30u16];
        let before = trainer.model.head.bias.clone();

        let report = trainer.train_step_report(&inputs, &targets);

        assert_eq!(report.tokens_seen, 2);
        assert_eq!(report.invalid_inputs, 0);
        assert_eq!(report.invalid_targets, 1);
        assert!(report.saturated_losses >= report.invalid_targets);
        assert_eq!(trainer.step_count, 1);
        assert_ne!(trainer.model.head.bias, before);
    }

    #[test]
    fn test_trainer_step_report_invalid_only_keeps_biases_stable() {
        let mut trainer = Trainer::new(1);
        let inputs = [10u16];
        let targets = [VOCAB_SIZE as u16];
        let before = trainer.model.head.bias.clone();

        let report = trainer.train_step_report(&inputs, &targets);

        assert_eq!(report.invalid_targets, 1);
        assert_eq!(trainer.model.head.bias, before);
    }

    #[test]
    fn test_trainer_step_report_invalid_input_only_keeps_biases_stable() {
        let mut trainer = Trainer::new(1);
        let inputs = [VOCAB_SIZE as u16];
        let targets = [30u16];
        let before = trainer.model.head.bias.clone();

        let report = trainer.train_step_report(&inputs, &targets);

        assert_eq!(report.invalid_inputs, 1);
        assert_eq!(report.invalid_targets, 0);
        assert_eq!(trainer.model.head.bias, before);
    }

    #[test]
    fn test_preview_step_report_keeps_biases_and_step_count_stable() {
        let mut trainer = Trainer::new(1);
        let inputs = [10u16, 20u16];
        let targets = [20u16, 30u16];
        let before = trainer.model.head.bias.clone();
        trainer.model.blocks[0].attention.q_membranes[0] = FixedPoint::ONE;

        let report = trainer.preview_step_report(&inputs, &targets);

        assert_eq!(report.tokens_seen, 2);
        assert_eq!(trainer.step_count, 0);
        assert_eq!(trainer.model.head.bias, before);
        assert_eq!(
            trainer.model.blocks[0].attention.q_membranes[0],
            FixedPoint::ZERO
        );
    }

    #[test]
    fn test_scaled_train_step_zero_lr_keeps_biases_but_counts_step() {
        let mut trainer = Trainer::new(1);
        let inputs = [10u16, 20u16];
        let targets = [20u16, 30u16];
        let before = trainer.model.head.bias.clone();

        let report = trainer.train_step_report_scaled(&inputs, &targets, FixedPoint::ZERO);

        assert_eq!(report.tokens_seen, 2);
        assert_eq!(trainer.step_count, 1);
        assert_eq!(trainer.model.head.bias, before);
    }

    #[test]
    fn test_trainer_reset_model_state_propagates() {
        let mut trainer = Trainer::new(1);
        trainer.model.blocks[0].attention.q_membranes[0] = FixedPoint::ONE;

        trainer.reset_model_state();

        assert_eq!(
            trainer.model.blocks[0].attention.q_membranes[0],
            FixedPoint::ZERO
        );
    }
}
