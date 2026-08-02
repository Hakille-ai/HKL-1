//! Operational runtime gates for HKL-2 cognition.
//!
//! Readiness reports describe global scenario maturity. Runtime gates combine
//! that maturity with the live cycle audit so a good suite score cannot bypass
//! recovery, budget, or checkpoint facts from the current step.

use crate::cognition::audit::{CycleAuditFlags, CycleAuditRecord, CycleRisk};
use crate::cognition::executive::ExecutiveAction;
use crate::cognition::readiness::{ReadinessLevel, ReadinessReport};
use crate::core::math::FixedPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeGateMode {
    Blocked,
    ObserveOnly,
    LearningAllowed,
    ExplorationAllowed,
    RecoveryOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeGateFlags(pub u16);

impl RuntimeGateFlags {
    pub const READINESS_BLOCKED: u16 = 1 << 0;
    pub const LIVE_RECOVERY: u16 = 1 << 1;
    pub const LIVE_CRITICAL: u16 = 1 << 2;
    pub const NO_EFFECT_AUTH: u16 = 1 << 3;
    pub const CHECKPOINT_REQUIRED: u16 = 1 << 4;
    pub const LEARNING_BUDGET_EMPTY: u16 = 1 << 5;
    pub const PLAN_EMPTY: u16 = 1 << 6;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn has(self, bit: u16) -> bool {
        self.0 & bit != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn set(&mut self, bit: u16, enabled: bool) {
        if enabled {
            self.0 |= bit;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeGateDecision {
    pub mode: RuntimeGateMode,
    pub flags: RuntimeGateFlags,
    pub readiness_level: ReadinessLevel,
    pub cycle_risk: CycleRisk,
    pub trace_id: u32,
    pub checkpoint_required: bool,
    pub learning_budget_ticks: u16,
    pub learning_scale: FixedPoint,
}

impl RuntimeGateDecision {
    pub fn permits_learning(&self) -> bool {
        matches!(self.mode, RuntimeGateMode::LearningAllowed)
            && !self.flags.has(RuntimeGateFlags::READINESS_BLOCKED)
            && !self.flags.has(RuntimeGateFlags::LIVE_RECOVERY)
            && !self.flags.has(RuntimeGateFlags::LIVE_CRITICAL)
            && !self.flags.has(RuntimeGateFlags::NO_EFFECT_AUTH)
            && !self.flags.has(RuntimeGateFlags::LEARNING_BUDGET_EMPTY)
            && !self.flags.has(RuntimeGateFlags::PLAN_EMPTY)
            && !matches!(self.cycle_risk, CycleRisk::Critical)
            && self.learning_budget_ticks > 0
            && self.learning_scale > FixedPoint::ZERO
    }

    pub fn permits_exploration(&self) -> bool {
        match self.mode {
            RuntimeGateMode::LearningAllowed => self.permits_learning(),
            RuntimeGateMode::ExplorationAllowed => {
                !self.flags.has(RuntimeGateFlags::READINESS_BLOCKED)
                    && !self.flags.has(RuntimeGateFlags::LIVE_RECOVERY)
                    && !self.flags.has(RuntimeGateFlags::LIVE_CRITICAL)
                    && !self.flags.has(RuntimeGateFlags::PLAN_EMPTY)
                    && !matches!(self.cycle_risk, CycleRisk::Critical)
            }
            _ => false,
        }
    }

    pub fn requires_recovery(&self) -> bool {
        matches!(self.mode, RuntimeGateMode::RecoveryOnly)
            && (self.flags.has(RuntimeGateFlags::LIVE_RECOVERY)
                || self.flags.has(RuntimeGateFlags::LIVE_CRITICAL)
                || matches!(self.cycle_risk, CycleRisk::Critical))
    }
}

pub fn evaluate_runtime_gate(
    readiness: &ReadinessReport,
    audit: &CycleAuditRecord,
) -> RuntimeGateDecision {
    let mut flags = RuntimeGateFlags::empty();
    let readiness_allowed = readiness.permits_agentic_loop();
    let live_recovery = audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED)
        || matches!(audit.action, ExecutiveAction::Recover);
    let live_critical = matches!(audit.risk, CycleRisk::Critical);
    let checkpoint_required = audit.flags.has(CycleAuditFlags::CHECKPOINT_REQUIRED);
    let learning_budget_empty = audit.learning_budget_ticks == 0;
    let plan_empty = audit.flags.has(CycleAuditFlags::PLAN_EMPTY);
    let effect_authorized = audit.allows_external_effects();
    let unauthorized_learning_attempt =
        matches!(audit.action, ExecutiveAction::Learn) && !effect_authorized;

    flags.set(RuntimeGateFlags::READINESS_BLOCKED, !readiness_allowed);
    flags.set(RuntimeGateFlags::LIVE_RECOVERY, live_recovery);
    flags.set(RuntimeGateFlags::LIVE_CRITICAL, live_critical);
    flags.set(
        RuntimeGateFlags::NO_EFFECT_AUTH,
        unauthorized_learning_attempt,
    );
    flags.set(RuntimeGateFlags::CHECKPOINT_REQUIRED, checkpoint_required);
    flags.set(
        RuntimeGateFlags::LEARNING_BUDGET_EMPTY,
        learning_budget_empty,
    );
    flags.set(RuntimeGateFlags::PLAN_EMPTY, plan_empty);

    let mode = if !readiness_allowed {
        RuntimeGateMode::Blocked
    } else if live_recovery || live_critical {
        RuntimeGateMode::RecoveryOnly
    } else if plan_empty {
        RuntimeGateMode::ObserveOnly
    } else if matches!(audit.action, ExecutiveAction::Explore) {
        RuntimeGateMode::ExplorationAllowed
    } else if effect_authorized && matches!(audit.action, ExecutiveAction::Learn) {
        RuntimeGateMode::LearningAllowed
    } else {
        RuntimeGateMode::ObserveOnly
    };

    let learning_scale = if matches!(mode, RuntimeGateMode::LearningAllowed) {
        audit.learning_scale
    } else {
        FixedPoint::ZERO
    };

    RuntimeGateDecision {
        mode,
        flags,
        readiness_level: readiness.level,
        cycle_risk: audit.risk,
        trace_id: audit.trace_id,
        checkpoint_required,
        learning_budget_ticks: audit.learning_budget_ticks,
        learning_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::controller::{CognitiveController, CycleSignals};
    use crate::cognition::readiness::{ReadinessReport, evaluate_readiness};
    use crate::cognition::scenario::run_default_scenarios;
    use crate::training::trainer::{TrainStepReport, TrainStepStatus};

    fn mature_readiness() -> ReadinessReport {
        evaluate_readiness(&run_default_scenarios())
    }

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

    fn audited_cycle(
        max_plan_budget_ticks: u16,
        preview: TrainStepReport,
        signals: CycleSignals,
    ) -> CycleAuditRecord {
        let mut controller = CognitiveController::conservative(max_plan_budget_ticks);
        let cycle = controller.preview_cycle(preview, signals);
        CycleAuditRecord::from_cycle(&cycle)
    }

    #[test]
    fn mature_readiness_and_healthy_cycle_allow_learning() {
        let readiness = mature_readiness();
        let audit = audited_cycle(
            16,
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

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::LearningAllowed);
        assert!(gate.permits_learning());
        assert!(gate.permits_exploration());
        assert_eq!(gate.learning_scale, audit.learning_scale);
        assert!(gate.flags.has(RuntimeGateFlags::CHECKPOINT_REQUIRED));
        assert!(!gate.flags.has(RuntimeGateFlags::READINESS_BLOCKED));
    }

    #[test]
    fn mature_readiness_cannot_override_live_recovery() {
        let readiness = mature_readiness();
        let audit = audited_cycle(
            16,
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

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::RecoveryOnly);
        assert!(gate.requires_recovery());
        assert!(!gate.permits_learning());
        assert_eq!(gate.learning_scale, FixedPoint::ZERO);
        assert!(gate.flags.has(RuntimeGateFlags::LIVE_CRITICAL));
        assert!(gate.flags.has(RuntimeGateFlags::LIVE_RECOVERY));
    }

    #[test]
    fn blocked_readiness_blocks_healthy_cycle() {
        let readiness = ReadinessReport::blocked();
        let audit = audited_cycle(
            16,
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

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::Blocked);
        assert!(!gate.permits_learning());
        assert!(gate.flags.has(RuntimeGateFlags::READINESS_BLOCKED));
    }

    #[test]
    fn exploration_cycle_can_probe_without_learning() {
        let readiness = mature_readiness();
        let audit = audited_cycle(
            16,
            report(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            CycleSignals {
                curiosity: FixedPoint::from_f32(0.85),
                novelty: FixedPoint::from_f32(0.35),
                prediction_error: FixedPoint::from_f32(0.15),
                safety_pressure: FixedPoint::ZERO,
            },
        );

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::ExplorationAllowed);
        assert!(!gate.permits_learning());
        assert!(gate.permits_exploration());
        assert_eq!(gate.learning_scale, FixedPoint::ZERO);
    }

    #[test]
    fn budget_starved_cycle_observes_only() {
        let readiness = mature_readiness();
        let audit = audited_cycle(
            2,
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

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::ObserveOnly);
        assert!(!gate.permits_learning());
        assert!(gate.flags.has(RuntimeGateFlags::LEARNING_BUDGET_EMPTY));
    }

    #[test]
    fn learning_without_effect_authorization_is_reported() {
        let readiness = mature_readiness();
        let audit = audited_cycle(
            2,
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

        assert_eq!(audit.action, ExecutiveAction::Learn);
        assert!(!audit.allows_external_effects());

        let gate = evaluate_runtime_gate(&readiness, &audit);

        assert_eq!(gate.mode, RuntimeGateMode::ObserveOnly);
        assert!(!gate.permits_learning());
        assert!(gate.flags.has(RuntimeGateFlags::NO_EFFECT_AUTH));
        assert!(gate.flags.has(RuntimeGateFlags::LEARNING_BUDGET_EMPTY));
        assert_eq!(gate.learning_scale, FixedPoint::ZERO);
    }

    #[test]
    fn forged_learning_decision_does_not_permit_learning() {
        let mut gate = RuntimeGateDecision {
            mode: RuntimeGateMode::LearningAllowed,
            flags: RuntimeGateFlags::empty(),
            readiness_level: ReadinessLevel::AgenticCandidate,
            cycle_risk: CycleRisk::Nominal,
            trace_id: 1,
            checkpoint_required: false,
            learning_budget_ticks: 1,
            learning_scale: FixedPoint::ONE,
        };

        assert!(gate.permits_learning());

        gate.flags = RuntimeGateFlags(RuntimeGateFlags::NO_EFFECT_AUTH);
        assert!(!gate.permits_learning());
        assert!(!gate.permits_exploration());

        gate.flags = RuntimeGateFlags::empty();
        gate.learning_budget_ticks = 0;
        assert!(!gate.permits_learning());
        assert!(!gate.permits_exploration());

        gate.learning_budget_ticks = 1;
        gate.learning_scale = FixedPoint::ZERO;
        assert!(!gate.permits_learning());
        assert!(!gate.permits_exploration());
    }

    #[test]
    fn forged_recovery_mode_requires_live_recovery_evidence() {
        let mut gate = RuntimeGateDecision {
            mode: RuntimeGateMode::RecoveryOnly,
            flags: RuntimeGateFlags::empty(),
            readiness_level: ReadinessLevel::AgenticCandidate,
            cycle_risk: CycleRisk::Nominal,
            trace_id: 1,
            checkpoint_required: false,
            learning_budget_ticks: 0,
            learning_scale: FixedPoint::ZERO,
        };

        assert!(!gate.requires_recovery());

        gate.flags = RuntimeGateFlags(RuntimeGateFlags::LIVE_RECOVERY);
        assert!(gate.requires_recovery());
    }
}
