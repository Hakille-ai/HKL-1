//! Metacognitive guard for HKL-2 training reports.
//!
//! The guard converts bounded trainer telemetry into a simple action that a
//! higher-level cognition loop can obey before continuing adaptive updates.

use crate::core::math::FixedPoint;
use crate::training::trainer::{TrainStepReport, TrainStepStatus};

const MAX_RATIO_COUNT: usize = 32_767;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingGuardAction {
    Continue,
    Throttle,
    Halt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingGuardReason {
    Healthy,
    EmptyBatch,
    TruncatedSequence,
    InvalidTokenRatio,
    SaturatedLossRatio,
    LossTooHigh,
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingGuardPolicy {
    pub max_invalid_ratio: FixedPoint,
    pub max_saturated_ratio: FixedPoint,
    pub max_loss: FixedPoint,
}

impl TrainingGuardPolicy {
    pub fn conservative() -> Self {
        Self {
            max_invalid_ratio: FixedPoint::from_f32(0.05),
            max_saturated_ratio: FixedPoint::from_f32(0.25),
            max_loss: FixedPoint::from_f32(64.0),
        }
    }

    pub fn normalized(self) -> Self {
        Self {
            max_invalid_ratio: clamp01(self.max_invalid_ratio),
            max_saturated_ratio: clamp01(self.max_saturated_ratio),
            max_loss: self.max_loss.max(FixedPoint::ZERO),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingGuardDecision {
    pub action: TrainingGuardAction,
    pub reason: TrainingGuardReason,
    pub invalid_ratio: FixedPoint,
    pub saturated_ratio: FixedPoint,
    pub recommended_lr_scale: FixedPoint,
}

pub struct TrainingGuard {
    pub policy: TrainingGuardPolicy,
    pub last_decision: TrainingGuardDecision,
}

impl TrainingGuard {
    pub fn new(policy: TrainingGuardPolicy) -> Self {
        Self {
            policy: policy.normalized(),
            last_decision: TrainingGuardDecision {
                action: TrainingGuardAction::Continue,
                reason: TrainingGuardReason::Healthy,
                invalid_ratio: FixedPoint::ZERO,
                saturated_ratio: FixedPoint::ZERO,
                recommended_lr_scale: FixedPoint::ONE,
            },
        }
    }

    pub fn evaluate(&mut self, report: &TrainStepReport) -> TrainingGuardDecision {
        let decision = self.decide(report);
        self.last_decision = decision;
        decision
    }

    fn decide(&self, report: &TrainStepReport) -> TrainingGuardDecision {
        if report.tokens_seen == 0 || report.status == TrainStepStatus::Empty {
            return TrainingGuardDecision {
                action: TrainingGuardAction::Halt,
                reason: TrainingGuardReason::EmptyBatch,
                invalid_ratio: FixedPoint::ZERO,
                saturated_ratio: FixedPoint::ZERO,
                recommended_lr_scale: FixedPoint::ZERO,
            };
        }

        let denom_count = report.tokens_seen.clamp(1, MAX_RATIO_COUNT);
        let invalid_positions = report
            .invalid_inputs
            .saturating_add(report.invalid_targets)
            .min(denom_count);
        let saturated_positions = report.saturated_losses.min(denom_count);
        let denom = FixedPoint::from_int(denom_count as i32);
        let invalid_ratio = FixedPoint::from_int(invalid_positions as i32) / denom;
        let saturated_ratio = FixedPoint::from_int(saturated_positions as i32) / denom;

        if invalid_ratio > self.policy.max_invalid_ratio {
            return TrainingGuardDecision {
                action: TrainingGuardAction::Halt,
                reason: TrainingGuardReason::InvalidTokenRatio,
                invalid_ratio,
                saturated_ratio,
                recommended_lr_scale: FixedPoint::ZERO,
            };
        }

        if report.loss > self.policy.max_loss {
            return TrainingGuardDecision {
                action: TrainingGuardAction::Halt,
                reason: TrainingGuardReason::LossTooHigh,
                invalid_ratio,
                saturated_ratio,
                recommended_lr_scale: FixedPoint::ZERO,
            };
        }

        if saturated_ratio > self.policy.max_saturated_ratio {
            return TrainingGuardDecision {
                action: TrainingGuardAction::Throttle,
                reason: TrainingGuardReason::SaturatedLossRatio,
                invalid_ratio,
                saturated_ratio,
                recommended_lr_scale: FixedPoint::from_f32(0.25),
            };
        }

        if report.status == TrainStepStatus::Truncated {
            return TrainingGuardDecision {
                action: TrainingGuardAction::Throttle,
                reason: TrainingGuardReason::TruncatedSequence,
                invalid_ratio,
                saturated_ratio,
                recommended_lr_scale: FixedPoint::from_f32(0.5),
            };
        }

        TrainingGuardDecision {
            action: TrainingGuardAction::Continue,
            reason: TrainingGuardReason::Healthy,
            invalid_ratio,
            saturated_ratio,
            recommended_lr_scale: FixedPoint::ONE,
        }
    }
}

fn clamp01(value: FixedPoint) -> FixedPoint {
    value.clamp(FixedPoint::ZERO, FixedPoint::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(
        tokens_seen: usize,
        invalid_inputs: usize,
        invalid_targets: usize,
        saturated_losses: usize,
        status: TrainStepStatus,
        loss: FixedPoint,
    ) -> TrainStepReport {
        TrainStepReport {
            loss,
            tokens_seen,
            saturated_losses,
            invalid_inputs,
            invalid_targets,
            status,
        }
    }

    #[test]
    fn guard_continues_on_healthy_report() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            16,
            0,
            0,
            0,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.action, TrainingGuardAction::Continue);
        assert_eq!(decision.reason, TrainingGuardReason::Healthy);
        assert_eq!(decision.recommended_lr_scale, FixedPoint::ONE);
    }

    #[test]
    fn guard_throttles_on_truncation() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            128,
            0,
            0,
            0,
            TrainStepStatus::Truncated,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.action, TrainingGuardAction::Throttle);
        assert_eq!(decision.reason, TrainingGuardReason::TruncatedSequence);
        assert_eq!(decision.recommended_lr_scale, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn guard_halts_on_invalid_token_ratio() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            10,
            1,
            0,
            1,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.action, TrainingGuardAction::Halt);
        assert_eq!(decision.reason, TrainingGuardReason::InvalidTokenRatio);
        assert_eq!(decision.recommended_lr_scale, FixedPoint::ZERO);
    }

    #[test]
    fn guard_throttles_on_saturated_loss_ratio() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            8,
            0,
            0,
            3,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.action, TrainingGuardAction::Throttle);
        assert_eq!(decision.reason, TrainingGuardReason::SaturatedLossRatio);
        assert_eq!(decision.recommended_lr_scale, FixedPoint::from_f32(0.25));
    }

    #[test]
    fn guard_halts_on_empty_batch() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            0,
            0,
            0,
            0,
            TrainStepStatus::Empty,
            FixedPoint::ZERO,
        ));

        assert_eq!(decision.action, TrainingGuardAction::Halt);
        assert_eq!(decision.reason, TrainingGuardReason::EmptyBatch);
        assert_eq!(decision.recommended_lr_scale, FixedPoint::ZERO);
    }

    #[test]
    fn guard_invalid_ratio_is_bounded_by_positions_seen() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            4,
            4,
            4,
            0,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.invalid_ratio, FixedPoint::ONE);
    }

    #[test]
    fn guard_saturated_ratio_is_bounded_by_positions_seen() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            4,
            0,
            0,
            40,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.saturated_ratio, FixedPoint::ONE);
    }

    #[test]
    fn guard_ratio_accounting_handles_large_external_counts() {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let decision = guard.evaluate(&report(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(4.0),
        ));

        assert_eq!(decision.invalid_ratio, FixedPoint::ONE);
        assert_eq!(decision.saturated_ratio, FixedPoint::ONE);
    }

    #[test]
    fn guard_normalizes_out_of_range_policy_ratios() {
        let guard = TrainingGuard::new(TrainingGuardPolicy {
            max_invalid_ratio: FixedPoint::from_f32(3.0),
            max_saturated_ratio: FixedPoint::from_f32(-1.0),
            max_loss: FixedPoint::from_f32(-4.0),
        });

        assert_eq!(guard.policy.max_invalid_ratio, FixedPoint::ONE);
        assert_eq!(guard.policy.max_saturated_ratio, FixedPoint::ZERO);
        assert_eq!(guard.policy.max_loss, FixedPoint::ZERO);
    }
}
