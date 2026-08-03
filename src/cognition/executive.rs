//! Executive cognition loop for HKL-2.
//!
//! The loop combines trainer telemetry, metacognitive guard output, curiosity,
//! novelty, prediction error, and safety pressure into one bounded decision.

use crate::core::math::FixedPoint;
use crate::training::monitor::{TrainingGuardAction, TrainingGuardDecision};
use crate::training::trainer::{TrainStepReport, TrainStepStatus};

pub const MAX_CAUSAL_SIGNALS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutiveAction {
    Learn,
    Explore,
    Consolidate,
    Recover,
    Idle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutiveReason {
    HealthyLearning,
    CuriosityNovelty,
    TrainingThrottled,
    TrainingHalted,
    SafetyPressure,
    NoWork,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalSignalKind {
    Guard,
    Safety,
    Curiosity,
    PredictionError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalSignal {
    pub kind: CausalSignalKind,
    pub intensity: FixedPoint,
}

impl CausalSignal {
    pub const fn neutral() -> Self {
        Self {
            kind: CausalSignalKind::Guard,
            intensity: FixedPoint::ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CognitiveObservation {
    pub train_report: TrainStepReport,
    pub guard_decision: TrainingGuardDecision,
    pub curiosity: FixedPoint,
    pub novelty: FixedPoint,
    pub prediction_error: FixedPoint,
    pub safety_pressure: FixedPoint,
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutivePolicy {
    pub max_safety_pressure: FixedPoint,
    pub explore_curiosity: FixedPoint,
    pub explore_novelty: FixedPoint,
}

impl ExecutivePolicy {
    pub const fn conservative() -> Self {
        Self {
            max_safety_pressure: FixedPoint::from_f32(0.7),
            explore_curiosity: FixedPoint::from_f32(0.55),
            explore_novelty: FixedPoint::from_f32(0.15),
        }
    }

    pub fn normalized(self) -> Self {
        Self {
            max_safety_pressure: clamp01(self.max_safety_pressure),
            explore_curiosity: clamp01(self.explore_curiosity),
            explore_novelty: clamp01(self.explore_novelty),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutiveDecision {
    pub action: ExecutiveAction,
    pub reason: ExecutiveReason,
    pub priority: FixedPoint,
    pub learning_scale: FixedPoint,
    pub exploration_scale: FixedPoint,
    pub trace: [CausalSignal; MAX_CAUSAL_SIGNALS],
    pub trace_len: usize,
}

pub struct ExecutiveLoop {
    pub policy: ExecutivePolicy,
    pub last_decision: ExecutiveDecision,
}

impl ExecutiveLoop {
    pub fn new(policy: ExecutivePolicy) -> Self {
        Self {
            policy: policy.normalized(),
            last_decision: ExecutiveDecision::idle(),
        }
    }

    pub fn evaluate(&mut self, observation: &CognitiveObservation) -> ExecutiveDecision {
        let decision = self.decide(observation);
        self.last_decision = decision;
        decision
    }

    fn decide(&self, observation: &CognitiveObservation) -> ExecutiveDecision {
        let trace = build_trace(observation);

        if observation.train_report.tokens_seen == 0
            || observation.train_report.status == TrainStepStatus::Empty
        {
            return ExecutiveDecision {
                action: ExecutiveAction::Idle,
                reason: ExecutiveReason::NoWork,
                priority: FixedPoint::from_f32(0.1),
                learning_scale: FixedPoint::ZERO,
                exploration_scale: FixedPoint::ZERO,
                trace,
                trace_len: MAX_CAUSAL_SIGNALS,
            };
        }

        if observation.safety_pressure > self.policy.max_safety_pressure {
            return ExecutiveDecision {
                action: ExecutiveAction::Recover,
                reason: ExecutiveReason::SafetyPressure,
                priority: FixedPoint::ONE,
                learning_scale: FixedPoint::ZERO,
                exploration_scale: FixedPoint::ZERO,
                trace,
                trace_len: MAX_CAUSAL_SIGNALS,
            };
        }

        if observation.guard_decision.action == TrainingGuardAction::Halt {
            return ExecutiveDecision {
                action: ExecutiveAction::Recover,
                reason: ExecutiveReason::TrainingHalted,
                priority: FixedPoint::from_f32(0.95),
                learning_scale: FixedPoint::ZERO,
                exploration_scale: FixedPoint::ZERO,
                trace,
                trace_len: MAX_CAUSAL_SIGNALS,
            };
        }

        if observation.guard_decision.action == TrainingGuardAction::Throttle {
            return ExecutiveDecision {
                action: ExecutiveAction::Consolidate,
                reason: ExecutiveReason::TrainingThrottled,
                priority: FixedPoint::from_f32(0.7),
                learning_scale: clamp01(observation.guard_decision.recommended_lr_scale),
                exploration_scale: FixedPoint::from_f32(0.1),
                trace,
                trace_len: MAX_CAUSAL_SIGNALS,
            };
        }

        if observation.curiosity >= self.policy.explore_curiosity
            && observation.novelty >= self.policy.explore_novelty
        {
            return ExecutiveDecision {
                action: ExecutiveAction::Explore,
                reason: ExecutiveReason::CuriosityNovelty,
                priority: (observation.curiosity * FixedPoint::from_f32(0.6)
                    + observation.novelty * FixedPoint::from_f32(0.4))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE),
                learning_scale: FixedPoint::from_f32(0.5),
                exploration_scale: clamp01(observation.curiosity.max(observation.novelty)),
                trace,
                trace_len: MAX_CAUSAL_SIGNALS,
            };
        }

        ExecutiveDecision {
            action: ExecutiveAction::Learn,
            reason: ExecutiveReason::HealthyLearning,
            priority: FixedPoint::from_f32(0.5),
            learning_scale: clamp01(observation.guard_decision.recommended_lr_scale),
            exploration_scale: FixedPoint::from_f32(0.05),
            trace,
            trace_len: MAX_CAUSAL_SIGNALS,
        }
    }
}

impl ExecutiveDecision {
    pub const fn idle() -> Self {
        Self {
            action: ExecutiveAction::Idle,
            reason: ExecutiveReason::NoWork,
            priority: FixedPoint::ZERO,
            learning_scale: FixedPoint::ZERO,
            exploration_scale: FixedPoint::ZERO,
            trace: [CausalSignal::neutral(); MAX_CAUSAL_SIGNALS],
            trace_len: 0,
        }
    }
}

fn build_trace(observation: &CognitiveObservation) -> [CausalSignal; MAX_CAUSAL_SIGNALS] {
    [
        CausalSignal {
            kind: CausalSignalKind::Guard,
            intensity: clamp01(
                observation
                    .guard_decision
                    .invalid_ratio
                    .max(observation.guard_decision.saturated_ratio),
            ),
        },
        CausalSignal {
            kind: CausalSignalKind::Safety,
            intensity: clamp01(observation.safety_pressure),
        },
        CausalSignal {
            kind: CausalSignalKind::Curiosity,
            intensity: clamp01(observation.curiosity.max(observation.novelty)),
        },
        CausalSignal {
            kind: CausalSignalKind::PredictionError,
            intensity: clamp01(observation.prediction_error),
        },
    ]
}

fn clamp01(value: FixedPoint) -> FixedPoint {
    value.clamp(FixedPoint::ZERO, FixedPoint::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::monitor::{
        TrainingGuard, TrainingGuardAction, TrainingGuardDecision, TrainingGuardPolicy,
        TrainingGuardReason,
    };

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

    fn observe(
        train_report: TrainStepReport,
        curiosity: FixedPoint,
        novelty: FixedPoint,
        prediction_error: FixedPoint,
        safety_pressure: FixedPoint,
    ) -> CognitiveObservation {
        let mut guard = TrainingGuard::new(TrainingGuardPolicy::conservative());
        let guard_decision = guard.evaluate(&train_report);
        CognitiveObservation {
            train_report,
            guard_decision,
            curiosity,
            novelty,
            prediction_error,
            safety_pressure,
        }
    }

    #[test]
    fn executive_learns_when_training_is_healthy() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            FixedPoint::from_f32(0.2),
            FixedPoint::from_f32(0.05),
            FixedPoint::from_f32(0.1),
            FixedPoint::ZERO,
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Learn);
        assert_eq!(decision.reason, ExecutiveReason::HealthyLearning);
        assert_eq!(decision.learning_scale, FixedPoint::ONE);
    }

    #[test]
    fn executive_explores_when_curiosity_and_novelty_are_high() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.3),
            FixedPoint::from_f32(0.2),
            FixedPoint::ZERO,
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Explore);
        assert_eq!(decision.reason, ExecutiveReason::CuriosityNovelty);
        assert!(decision.exploration_scale >= FixedPoint::from_f32(0.8));
    }

    #[test]
    fn executive_consolidates_when_guard_throttles() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(
                128,
                0,
                0,
                0,
                TrainStepStatus::Truncated,
                FixedPoint::from_f32(4.0),
            ),
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.3),
            FixedPoint::from_f32(0.2),
            FixedPoint::ZERO,
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(
            observation.guard_decision.reason,
            TrainingGuardReason::TruncatedSequence
        );
        assert_eq!(decision.action, ExecutiveAction::Consolidate);
        assert_eq!(decision.reason, ExecutiveReason::TrainingThrottled);
        assert_eq!(decision.learning_scale, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn executive_recovers_when_guard_halts() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(
                8,
                1,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.2),
            FixedPoint::ZERO,
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Recover);
        assert_eq!(decision.reason, ExecutiveReason::TrainingHalted);
        assert_eq!(decision.learning_scale, FixedPoint::ZERO);
    }

    #[test]
    fn safety_pressure_overrides_exploration() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.2),
            FixedPoint::from_f32(0.9),
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Recover);
        assert_eq!(decision.reason, ExecutiveReason::SafetyPressure);
        assert_eq!(decision.priority, FixedPoint::ONE);
    }

    #[test]
    fn executive_idles_on_empty_training_report() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = observe(
            report(0, 0, 0, 0, TrainStepStatus::Empty, FixedPoint::ZERO),
            FixedPoint::from_f32(0.9),
            FixedPoint::from_f32(0.9),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
        );

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Idle);
        assert_eq!(decision.reason, ExecutiveReason::NoWork);
        assert_eq!(decision.trace_len, MAX_CAUSAL_SIGNALS);
    }

    #[test]
    fn executive_clamps_external_learning_scale() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = CognitiveObservation {
            train_report: report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            guard_decision: TrainingGuardDecision {
                action: TrainingGuardAction::Continue,
                reason: TrainingGuardReason::Healthy,
                invalid_ratio: FixedPoint::ZERO,
                saturated_ratio: FixedPoint::ZERO,
                recommended_lr_scale: FixedPoint::from_f32(4.0),
            },
            curiosity: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            prediction_error: FixedPoint::ZERO,
            safety_pressure: FixedPoint::ZERO,
        };

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Learn);
        assert_eq!(decision.learning_scale, FixedPoint::ONE);
    }

    #[test]
    fn executive_clamps_trace_and_exploration_outputs() {
        let mut loop_ = ExecutiveLoop::new(ExecutivePolicy::conservative());
        let observation = CognitiveObservation {
            train_report: report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            guard_decision: TrainingGuardDecision {
                action: TrainingGuardAction::Continue,
                reason: TrainingGuardReason::Healthy,
                invalid_ratio: FixedPoint::from_f32(3.0),
                saturated_ratio: FixedPoint::from_f32(2.0),
                recommended_lr_scale: FixedPoint::ONE,
            },
            curiosity: FixedPoint::from_f32(1.5),
            novelty: FixedPoint::from_f32(2.0),
            prediction_error: FixedPoint::from_f32(5.0),
            safety_pressure: FixedPoint::from_f32(-0.5),
        };

        let decision = loop_.evaluate(&observation);

        assert_eq!(decision.action, ExecutiveAction::Explore);
        assert_eq!(decision.exploration_scale, FixedPoint::ONE);
        for signal in decision.trace.iter().take(decision.trace_len) {
            assert!(signal.intensity >= FixedPoint::ZERO);
            assert!(signal.intensity <= FixedPoint::ONE);
        }
    }

    #[test]
    fn executive_policy_normalization_bounds_thresholds() {
        let policy = ExecutivePolicy {
            max_safety_pressure: FixedPoint::from_f32(2.0),
            explore_curiosity: FixedPoint::from_f32(-1.0),
            explore_novelty: FixedPoint::from_f32(0.25),
        }
        .normalized();

        assert_eq!(policy.max_safety_pressure, FixedPoint::ONE);
        assert_eq!(policy.explore_curiosity, FixedPoint::ZERO);
        assert_eq!(policy.explore_novelty, FixedPoint::from_f32(0.25));
    }

    #[test]
    fn executive_loop_new_stores_normalized_policy() {
        let loop_ = ExecutiveLoop::new(ExecutivePolicy {
            max_safety_pressure: FixedPoint::from_f32(-0.5),
            explore_curiosity: FixedPoint::from_f32(4.0),
            explore_novelty: FixedPoint::from_f32(-2.0),
        });

        assert_eq!(loop_.policy.max_safety_pressure, FixedPoint::ZERO);
        assert_eq!(loop_.policy.explore_curiosity, FixedPoint::ONE);
        assert_eq!(loop_.policy.explore_novelty, FixedPoint::ZERO);
    }
}
