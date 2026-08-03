//! Compact audit records for HKL-2 cognition cycles.
//!
//! Audit records summarize a dry-run cycle without owning model state or
//! executing effects. They are intended for telemetry, scenario comparison, and
//! future persistence checkpoints.

use crate::cognition::controller::CognitiveCycleReport;
use crate::cognition::executive::{ExecutiveAction, ExecutiveReason};
use crate::cognition::planner::{MAX_PLAN_STEPS, PlanStepKind};
use crate::core::math::FixedPoint;
use crate::training::monitor::{TrainingGuardAction, TrainingGuardReason};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleRisk {
    Nominal,
    Elevated,
    Critical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CycleAuditFlags(pub u16);

impl CycleAuditFlags {
    pub const APPLY_LEARNING: u16 = 1 << 0;
    pub const RECOVERY_REQUIRED: u16 = 1 << 1;
    pub const CHECKPOINT_REQUIRED: u16 = 1 << 2;
    pub const LOSS_SATURATED: u16 = 1 << 3;
    pub const INVALID_TOKENS: u16 = 1 << 4;
    pub const PLAN_EMPTY: u16 = 1 << 5;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn has(self, bit: u16) -> bool {
        self.0 & bit != 0
    }

    fn set(&mut self, bit: u16, enabled: bool) {
        if enabled {
            self.0 |= bit;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CycleAuditRecord {
    pub trace_id: u32,
    pub risk: CycleRisk,
    pub action: ExecutiveAction,
    pub reason: ExecutiveReason,
    pub guard_action: TrainingGuardAction,
    pub guard_reason: TrainingGuardReason,
    pub first_step: PlanStepKind,
    pub plan_len: usize,
    pub tokens_seen: usize,
    pub budget_ticks: u16,
    pub learning_budget_ticks: u16,
    pub flags: CycleAuditFlags,
    pub learning_scale: FixedPoint,
    pub loss: FixedPoint,
}

impl CycleAuditRecord {
    pub const fn empty() -> Self {
        Self {
            trace_id: 0,
            risk: CycleRisk::Elevated,
            action: ExecutiveAction::Idle,
            reason: ExecutiveReason::NoWork,
            guard_action: TrainingGuardAction::Continue,
            guard_reason: TrainingGuardReason::Healthy,
            first_step: PlanStepKind::Idle,
            plan_len: 0,
            tokens_seen: 0,
            budget_ticks: 0,
            learning_budget_ticks: 0,
            flags: CycleAuditFlags(CycleAuditFlags::PLAN_EMPTY),
            learning_scale: FixedPoint::ZERO,
            loss: FixedPoint::ZERO,
        }
    }

    pub fn from_cycle(cycle: &CognitiveCycleReport) -> Self {
        let plan_len = cycle.plan.len.min(MAX_PLAN_STEPS);
        let learning_budget_ticks = cycle
            .learning_budget_ticks
            .min(cycle.plan.total_budget_ticks);
        let learning_allowed = cycle.may_apply_learning
            && !cycle.must_recover
            && learning_budget_ticks > 0
            && cycle.recommended_learning_scale > FixedPoint::ZERO;

        let mut flags = CycleAuditFlags::empty();
        flags.set(CycleAuditFlags::APPLY_LEARNING, learning_allowed);
        flags.set(CycleAuditFlags::RECOVERY_REQUIRED, cycle.must_recover);
        flags.set(
            CycleAuditFlags::CHECKPOINT_REQUIRED,
            cycle.checkpoint_required,
        );
        flags.set(
            CycleAuditFlags::LOSS_SATURATED,
            cycle.preview_report.saturated_losses > 0,
        );
        flags.set(
            CycleAuditFlags::INVALID_TOKENS,
            cycle.preview_report.invalid_inputs > 0 || cycle.preview_report.invalid_targets > 0,
        );
        flags.set(CycleAuditFlags::PLAN_EMPTY, plan_len == 0);

        Self {
            trace_id: cycle.plan.trace_id,
            risk: risk_for(cycle, flags),
            action: cycle.executive_decision.action,
            reason: cycle.executive_decision.reason,
            guard_action: cycle.guard_decision.action,
            guard_reason: cycle.guard_decision.reason,
            first_step: cycle.plan.first_step().kind,
            plan_len,
            tokens_seen: cycle.preview_report.tokens_seen,
            budget_ticks: cycle.plan.total_budget_ticks,
            learning_budget_ticks,
            flags,
            learning_scale: cycle.recommended_learning_scale,
            loss: cycle.preview_report.loss,
        }
    }

    pub fn allows_external_effects(&self) -> bool {
        self.flags.has(CycleAuditFlags::APPLY_LEARNING)
            && !self.flags.has(CycleAuditFlags::RECOVERY_REQUIRED)
            && matches!(self.risk, CycleRisk::Nominal | CycleRisk::Elevated)
    }
}

fn risk_for(cycle: &CognitiveCycleReport, flags: CycleAuditFlags) -> CycleRisk {
    if cycle.must_recover || cycle.guard_decision.action == TrainingGuardAction::Halt {
        return CycleRisk::Critical;
    }

    if cycle.checkpoint_required
        || flags.has(CycleAuditFlags::LOSS_SATURATED)
        || flags.has(CycleAuditFlags::INVALID_TOKENS)
        || flags.has(CycleAuditFlags::PLAN_EMPTY)
    {
        return CycleRisk::Elevated;
    }

    CycleRisk::Nominal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::controller::{CognitiveController, CycleSignals};
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
    fn audit_record_marks_learning_cycle_as_effect_allowed() {
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

        let audit = CycleAuditRecord::from_cycle(&cycle);

        assert_eq!(audit.risk, CycleRisk::Elevated);
        assert!(audit.flags.has(CycleAuditFlags::APPLY_LEARNING));
        assert!(audit.flags.has(CycleAuditFlags::CHECKPOINT_REQUIRED));
        assert!(audit.learning_budget_ticks > 0);
        assert!(audit.allows_external_effects());
    }

    #[test]
    fn audit_record_marks_recovery_as_critical() {
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

        let audit = CycleAuditRecord::from_cycle(&cycle);

        assert_eq!(audit.risk, CycleRisk::Critical);
        assert!(audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED));
        assert!(audit.flags.has(CycleAuditFlags::LOSS_SATURATED));
        assert_eq!(audit.learning_budget_ticks, 0);
        assert!(!audit.allows_external_effects());
    }

    #[test]
    fn audit_record_carries_invalid_token_flag() {
        let mut controller = CognitiveController::conservative(16);
        let cycle = controller.preview_cycle(
            report(
                8,
                1,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals::neutral(),
        );

        let audit = CycleAuditRecord::from_cycle(&cycle);

        assert!(audit.flags.has(CycleAuditFlags::INVALID_TOKENS));
        assert_eq!(audit.risk, CycleRisk::Critical);
    }

    #[test]
    fn audit_record_is_stable_for_same_cycle() {
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
                novelty: FixedPoint::from_f32(0.3),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::ZERO,
            },
        );

        let a = CycleAuditRecord::from_cycle(&cycle);
        let b = CycleAuditRecord::from_cycle(&cycle);

        assert_eq!(a.trace_id, b.trace_id);
        assert_eq!(a.flags, b.flags);
        assert_eq!(a.first_step, b.first_step);
        assert_eq!(a.learning_budget_ticks, b.learning_budget_ticks);
    }

    #[test]
    fn audit_record_rejects_forged_learning_without_budget() {
        let mut controller = CognitiveController::conservative(2);
        let mut cycle = controller.preview_cycle(
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
        cycle.may_apply_learning = true;
        cycle.recommended_learning_scale = FixedPoint::ONE;

        let audit = CycleAuditRecord::from_cycle(&cycle);

        assert_eq!(audit.learning_budget_ticks, 0);
        assert!(!audit.flags.has(CycleAuditFlags::APPLY_LEARNING));
        assert!(!audit.allows_external_effects());
    }

    #[test]
    fn audit_record_bounds_externally_supplied_plan_len() {
        let mut controller = CognitiveController::conservative(16);
        let mut cycle = controller.preview_cycle(
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
        cycle.plan.len = usize::MAX;

        let audit = CycleAuditRecord::from_cycle(&cycle);

        assert_eq!(audit.plan_len, MAX_PLAN_STEPS);
        assert!(!audit.flags.has(CycleAuditFlags::PLAN_EMPTY));
    }
}
