//! Cycle controller for HKL-2 cognition.
//!
//! This module composes the metacognitive guard, executive loop, and bounded
//! planner into one auditable dry-run cycle. The controller does not mutate the
//! model; it decides whether a caller may apply a later training update.

use crate::cognition::executive::{
    CognitiveObservation, ExecutiveAction, ExecutiveDecision, ExecutiveLoop, ExecutivePolicy,
};
use crate::cognition::planner::{CognitivePlan, CognitivePlanner, PlanStepKind};
use crate::core::math::FixedPoint;
use crate::training::monitor::{
    TrainingGuard, TrainingGuardAction, TrainingGuardDecision, TrainingGuardPolicy,
};
use crate::training::trainer::TrainStepReport;

#[derive(Clone, Copy, Debug)]
pub struct CycleSignals {
    pub curiosity: FixedPoint,
    pub novelty: FixedPoint,
    pub prediction_error: FixedPoint,
    pub safety_pressure: FixedPoint,
}

impl CycleSignals {
    pub const fn neutral() -> Self {
        Self {
            curiosity: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            prediction_error: FixedPoint::ZERO,
            safety_pressure: FixedPoint::ZERO,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CognitiveCycleReport {
    pub preview_report: TrainStepReport,
    pub guard_decision: TrainingGuardDecision,
    pub executive_decision: ExecutiveDecision,
    pub plan: CognitivePlan,
    pub may_apply_learning: bool,
    pub must_recover: bool,
    pub checkpoint_required: bool,
    pub learning_budget_ticks: u16,
    pub recommended_learning_scale: FixedPoint,
}

pub struct CognitiveController {
    pub guard: TrainingGuard,
    pub executive: ExecutiveLoop,
    pub planner: CognitivePlanner,
    pub last_report: Option<CognitiveCycleReport>,
}

impl CognitiveController {
    pub fn conservative(max_plan_budget_ticks: u16) -> Self {
        Self::new(
            TrainingGuardPolicy::conservative(),
            ExecutivePolicy::conservative(),
            max_plan_budget_ticks,
        )
    }

    pub fn new(
        guard_policy: TrainingGuardPolicy,
        executive_policy: ExecutivePolicy,
        max_plan_budget_ticks: u16,
    ) -> Self {
        Self {
            guard: TrainingGuard::new(guard_policy),
            executive: ExecutiveLoop::new(executive_policy),
            planner: CognitivePlanner::new(max_plan_budget_ticks),
            last_report: None,
        }
    }

    pub fn preview_cycle(
        &mut self,
        preview_report: TrainStepReport,
        signals: CycleSignals,
    ) -> CognitiveCycleReport {
        let guard_decision = self.guard.evaluate(&preview_report);
        let executive_decision = self.executive.evaluate(&CognitiveObservation {
            train_report: preview_report,
            guard_decision,
            curiosity: signals.curiosity,
            novelty: signals.novelty,
            prediction_error: signals.prediction_error,
            safety_pressure: signals.safety_pressure,
        });
        let plan = self.planner.plan(&executive_decision);
        let learning_step = find_step(&plan, PlanStepKind::ApplyLearning);
        let learning_budget_ticks = learning_step.map(|step| step.budget_ticks).unwrap_or(0);
        let may_apply_learning = guard_decision.action != TrainingGuardAction::Halt
            && executive_decision.action != ExecutiveAction::Recover
            && executive_decision.learning_scale > FixedPoint::ZERO
            && learning_budget_ticks > 0;
        let must_recover = guard_decision.action == TrainingGuardAction::Halt
            || executive_decision.action == ExecutiveAction::Recover
            || plan.rollback_required;
        let checkpoint_required = must_recover
            || plan
                .steps
                .iter()
                .take(plan.len)
                .any(|step| step.requires_checkpoint && step.budget_ticks > 0);
        let recommended_learning_scale = if may_apply_learning {
            executive_decision.learning_scale
        } else {
            FixedPoint::ZERO
        };

        let report = CognitiveCycleReport {
            preview_report,
            guard_decision,
            executive_decision,
            plan,
            may_apply_learning,
            must_recover,
            checkpoint_required,
            learning_budget_ticks,
            recommended_learning_scale,
        };
        self.last_report = Some(report);
        report
    }
}

fn find_step(
    plan: &CognitivePlan,
    kind: PlanStepKind,
) -> Option<crate::cognition::planner::PlanStep> {
    plan.steps
        .iter()
        .take(plan.len)
        .copied()
        .find(|step| step.kind == kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::executive::ExecutiveReason;
    use crate::training::trainer::{TrainStepReport, TrainStepStatus};

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
    fn controller_allows_learning_for_healthy_cycle() {
        let mut controller = CognitiveController::conservative(16);
        let cycle = controller.preview_cycle(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals::neutral(),
        );

        assert!(cycle.may_apply_learning);
        assert!(!cycle.must_recover);
        assert_eq!(cycle.executive_decision.action, ExecutiveAction::Learn);
        assert!(cycle.learning_budget_ticks > 0);
        assert_eq!(cycle.recommended_learning_scale, FixedPoint::ONE);
    }

    #[test]
    fn controller_blocks_learning_when_guard_halts() {
        let mut controller = CognitiveController::conservative(16);
        let cycle = controller.preview_cycle(
            report(
                4,
                0,
                0,
                1,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(8192.0),
            ),
            CycleSignals::neutral(),
        );

        assert!(!cycle.may_apply_learning);
        assert!(cycle.must_recover);
        assert_eq!(cycle.learning_budget_ticks, 0);
        assert_eq!(cycle.executive_decision.action, ExecutiveAction::Recover);
        assert_eq!(
            cycle.executive_decision.reason,
            ExecutiveReason::TrainingHalted
        );
        assert!(cycle.plan.rollback_required);
    }

    #[test]
    fn controller_routes_high_curiosity_to_explore_plan() {
        let mut controller = CognitiveController::conservative(32);
        let cycle = controller.preview_cycle(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals {
                curiosity: FixedPoint::from_f32(0.8),
                novelty: FixedPoint::from_f32(0.3),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::ZERO,
            },
        );

        assert!(cycle.may_apply_learning);
        assert_eq!(cycle.executive_decision.action, ExecutiveAction::Explore);
        assert_eq!(cycle.plan.steps[1].kind, PlanStepKind::ExploreProbe);
        assert!(cycle.learning_budget_ticks > 0);
        assert_eq!(cycle.recommended_learning_scale, FixedPoint::from_f32(0.5));
    }

    #[test]
    fn controller_blocks_learning_when_budget_starves_apply_step() {
        let mut controller = CognitiveController::conservative(2);
        let cycle = controller.preview_cycle(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals::neutral(),
        );

        assert!(!cycle.may_apply_learning);
        assert!(!cycle.must_recover);
        assert_eq!(cycle.learning_budget_ticks, 0);
        assert_eq!(cycle.recommended_learning_scale, FixedPoint::ZERO);
    }

    #[test]
    fn controller_safety_pressure_forces_recovery() {
        let mut controller = CognitiveController::conservative(16);
        let cycle = controller.preview_cycle(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals {
                curiosity: FixedPoint::from_f32(0.8),
                novelty: FixedPoint::from_f32(0.8),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::from_f32(0.9),
            },
        );

        assert!(!cycle.may_apply_learning);
        assert!(cycle.must_recover);
        assert_eq!(cycle.learning_budget_ticks, 0);
        assert_eq!(
            cycle.executive_decision.reason,
            ExecutiveReason::SafetyPressure
        );
    }

    #[test]
    fn controller_keeps_recover_requirement_when_budget_trims_recover_step() {
        let mut controller = CognitiveController::conservative(1);
        let cycle = controller.preview_cycle(
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals {
                curiosity: FixedPoint::ZERO,
                novelty: FixedPoint::ZERO,
                prediction_error: FixedPoint::ZERO,
                safety_pressure: FixedPoint::from_f32(0.9),
            },
        );

        assert_eq!(cycle.executive_decision.action, ExecutiveAction::Recover);
        assert!(cycle.must_recover);
        assert!(cycle.checkpoint_required);
        assert!(!cycle.may_apply_learning);
        assert_eq!(cycle.learning_budget_ticks, 0);
        assert_eq!(cycle.recommended_learning_scale, FixedPoint::ZERO);
    }
}
