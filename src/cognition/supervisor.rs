//! Multi-cycle supervision for HKL-2 cognition runtime gates.
//!
//! The supervisor ledger turns per-cycle runtime gate decisions into a compact
//! episode summary. It keeps future long-running loops auditable without
//! allocating memory or executing model effects.

use crate::cognition::audit::CycleRisk;
use crate::cognition::runtime_gate::{RuntimeGateDecision, RuntimeGateFlags, RuntimeGateMode};
use crate::core::math::FixedPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupervisionStatus {
    Stable,
    Watching,
    Recovering,
    Quarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupervisionFlags(pub u16);

impl SupervisionFlags {
    pub const HAS_BLOCKED: u16 = 1 << 0;
    pub const HAS_OBSERVED: u16 = 1 << 1;
    pub const HAS_LEARNING: u16 = 1 << 2;
    pub const HAS_EXPLORATION: u16 = 1 << 3;
    pub const HAS_RECOVERY: u16 = 1 << 4;
    pub const CRITICAL_SEEN: u16 = 1 << 5;
    pub const CHECKPOINT_SEEN: u16 = 1 << 6;
    pub const RECOVERY_STREAK_LIMIT: u16 = 1 << 7;

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
pub struct SupervisionPolicy {
    pub max_recovery_streak: u16,
}

impl SupervisionPolicy {
    pub const fn conservative() -> Self {
        Self {
            max_recovery_streak: 2,
        }
    }

    pub const fn normalized(self) -> Self {
        Self {
            max_recovery_streak: if self.max_recovery_streak == 0 {
                1
            } else {
                self.max_recovery_streak
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SupervisionLedger {
    pub policy: SupervisionPolicy,
    pub cycles: u32,
    pub learning_allowed: u32,
    pub exploration_allowed: u32,
    pub observed: u32,
    pub blocked: u32,
    pub recovery: u32,
    pub recovery_streak: u16,
    pub max_recovery_streak_seen: u16,
    pub max_risk: CycleRisk,
    pub last_mode: RuntimeGateMode,
    pub last_trace_id: u32,
    pub cumulative_learning_scale: FixedPoint,
    pub flags: SupervisionFlags,
    pub summary_hash: u32,
}

impl SupervisionLedger {
    pub const fn new(policy: SupervisionPolicy) -> Self {
        Self {
            policy: policy.normalized(),
            cycles: 0,
            learning_allowed: 0,
            exploration_allowed: 0,
            observed: 0,
            blocked: 0,
            recovery: 0,
            recovery_streak: 0,
            max_recovery_streak_seen: 0,
            max_risk: CycleRisk::Nominal,
            last_mode: RuntimeGateMode::ObserveOnly,
            last_trace_id: 0,
            cumulative_learning_scale: FixedPoint::ZERO,
            flags: SupervisionFlags::empty(),
            summary_hash: 0x9e37_79b9,
        }
    }

    pub const fn conservative() -> Self {
        Self::new(SupervisionPolicy::conservative())
    }

    pub fn record(&mut self, decision: &RuntimeGateDecision) -> SupervisionSnapshot {
        let effective_mode = effective_mode(decision);
        self.cycles = self.cycles.saturating_add(1);
        self.last_mode = effective_mode;
        self.last_trace_id = decision.trace_id;
        self.max_risk = max_risk(self.max_risk, decision.cycle_risk);

        match effective_mode {
            RuntimeGateMode::Blocked => {
                self.blocked = self.blocked.saturating_add(1);
                self.recovery_streak = 0;
                self.flags.set(SupervisionFlags::HAS_BLOCKED, true);
            }
            RuntimeGateMode::ObserveOnly => {
                self.observed = self.observed.saturating_add(1);
                self.recovery_streak = 0;
                self.flags.set(SupervisionFlags::HAS_OBSERVED, true);
            }
            RuntimeGateMode::LearningAllowed => {
                self.learning_allowed = self.learning_allowed.saturating_add(1);
                self.recovery_streak = 0;
                self.cumulative_learning_scale =
                    self.cumulative_learning_scale + decision.learning_scale;
                self.flags.set(SupervisionFlags::HAS_LEARNING, true);
            }
            RuntimeGateMode::ExplorationAllowed => {
                self.exploration_allowed = self.exploration_allowed.saturating_add(1);
                self.recovery_streak = 0;
                self.flags.set(SupervisionFlags::HAS_EXPLORATION, true);
            }
            RuntimeGateMode::RecoveryOnly => {
                self.recovery = self.recovery.saturating_add(1);
                self.recovery_streak = self.recovery_streak.saturating_add(1);
                self.max_recovery_streak_seen =
                    self.max_recovery_streak_seen.max(self.recovery_streak);
                self.flags.set(SupervisionFlags::HAS_RECOVERY, true);
            }
        }

        self.flags.set(
            SupervisionFlags::CRITICAL_SEEN,
            matches!(decision.cycle_risk, CycleRisk::Critical),
        );
        self.flags.set(
            SupervisionFlags::CHECKPOINT_SEEN,
            decision.checkpoint_required,
        );
        self.flags.set(
            SupervisionFlags::RECOVERY_STREAK_LIMIT,
            self.recovery_streak >= self.policy.max_recovery_streak,
        );
        self.summary_hash = mix_decision(self.summary_hash, decision, effective_mode, self.cycles);

        self.snapshot()
    }

    pub const fn snapshot(&self) -> SupervisionSnapshot {
        SupervisionSnapshot {
            status: self.status(),
            cycles: self.cycles,
            learning_allowed: self.learning_allowed,
            exploration_allowed: self.exploration_allowed,
            observed: self.observed,
            blocked: self.blocked,
            recovery: self.recovery,
            recovery_streak: self.recovery_streak,
            max_recovery_streak_seen: self.max_recovery_streak_seen,
            max_risk: self.max_risk,
            last_mode: self.last_mode,
            last_trace_id: self.last_trace_id,
            cumulative_learning_scale: self.cumulative_learning_scale,
            flags: self.flags,
            summary_hash: self.summary_hash,
        }
    }

    pub const fn status(&self) -> SupervisionStatus {
        if self.flags.has(SupervisionFlags::RECOVERY_STREAK_LIMIT) {
            SupervisionStatus::Quarantined
        } else if self.recovery_streak > 0 {
            SupervisionStatus::Recovering
        } else if self.flags.has(SupervisionFlags::CRITICAL_SEEN)
            || self.flags.has(SupervisionFlags::HAS_BLOCKED)
        {
            SupervisionStatus::Watching
        } else {
            SupervisionStatus::Stable
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SupervisionSnapshot {
    pub status: SupervisionStatus,
    pub cycles: u32,
    pub learning_allowed: u32,
    pub exploration_allowed: u32,
    pub observed: u32,
    pub blocked: u32,
    pub recovery: u32,
    pub recovery_streak: u16,
    pub max_recovery_streak_seen: u16,
    pub max_risk: CycleRisk,
    pub last_mode: RuntimeGateMode,
    pub last_trace_id: u32,
    pub cumulative_learning_scale: FixedPoint,
    pub flags: SupervisionFlags,
    pub summary_hash: u32,
}

fn effective_mode(decision: &RuntimeGateDecision) -> RuntimeGateMode {
    if decision.requires_recovery() {
        RuntimeGateMode::RecoveryOnly
    } else if matches!(decision.mode, RuntimeGateMode::Blocked)
        && decision.flags.has(RuntimeGateFlags::READINESS_BLOCKED)
    {
        RuntimeGateMode::Blocked
    } else if decision.permits_learning() {
        RuntimeGateMode::LearningAllowed
    } else if decision.permits_exploration() {
        RuntimeGateMode::ExplorationAllowed
    } else {
        RuntimeGateMode::ObserveOnly
    }
}

const fn max_risk(left: CycleRisk, right: CycleRisk) -> CycleRisk {
    match (left, right) {
        (CycleRisk::Critical, _) | (_, CycleRisk::Critical) => CycleRisk::Critical,
        (CycleRisk::Elevated, _) | (_, CycleRisk::Elevated) => CycleRisk::Elevated,
        _ => CycleRisk::Nominal,
    }
}

fn mix_decision(
    hash: u32,
    decision: &RuntimeGateDecision,
    effective_mode: RuntimeGateMode,
    cycle: u32,
) -> u32 {
    let mut next = hash ^ cycle.rotate_left(7);
    next = mix(next, mode_id(effective_mode));
    next = mix(next, risk_id(decision.cycle_risk));
    next = mix(next, decision.flags.0 as u32);
    next = mix(next, decision.trace_id);
    next = mix(next, decision.learning_budget_ticks as u32);
    next = mix(next, decision.learning_scale.to_bits() as u32);
    next
}

fn mix(hash: u32, value: u32) -> u32 {
    hash.rotate_left(5) ^ value.wrapping_mul(0x85eb_ca6b)
}

const fn mode_id(mode: RuntimeGateMode) -> u32 {
    match mode {
        RuntimeGateMode::Blocked => 1,
        RuntimeGateMode::ObserveOnly => 2,
        RuntimeGateMode::LearningAllowed => 3,
        RuntimeGateMode::ExplorationAllowed => 4,
        RuntimeGateMode::RecoveryOnly => 5,
    }
}

const fn risk_id(risk: CycleRisk) -> u32 {
    match risk {
        CycleRisk::Nominal => 1,
        CycleRisk::Elevated => 2,
        CycleRisk::Critical => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::controller::{CognitiveController, CycleSignals};
    use crate::cognition::readiness::evaluate_readiness;
    use crate::cognition::runtime_gate::evaluate_runtime_gate;
    use crate::cognition::scenario::run_default_scenarios;
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

    fn gate(
        max_plan_budget_ticks: u16,
        preview: TrainStepReport,
        signals: CycleSignals,
    ) -> RuntimeGateDecision {
        let readiness = evaluate_readiness(&run_default_scenarios());
        let mut controller = CognitiveController::conservative(max_plan_budget_ticks);
        let cycle = controller.preview_cycle(preview, signals);
        let audit = crate::cognition::audit::CycleAuditRecord::from_cycle(&cycle);
        evaluate_runtime_gate(&readiness, &audit)
    }

    #[test]
    fn ledger_records_learning_and_stays_stable() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = gate(
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

        let snapshot = ledger.record(&decision);

        assert_eq!(snapshot.status, SupervisionStatus::Stable);
        assert_eq!(snapshot.cycles, 1);
        assert_eq!(snapshot.learning_allowed, 1);
        assert_eq!(snapshot.recovery, 0);
        assert!(snapshot.flags.has(SupervisionFlags::HAS_LEARNING));
        assert_eq!(snapshot.cumulative_learning_scale, decision.learning_scale);
    }

    #[test]
    fn ledger_tracks_exploration_without_learning_scale() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = gate(
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
                curiosity: FixedPoint::from_f32(0.9),
                novelty: FixedPoint::from_f32(0.4),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::ZERO,
            },
        );

        let snapshot = ledger.record(&decision);

        assert_eq!(snapshot.status, SupervisionStatus::Stable);
        assert_eq!(snapshot.exploration_allowed, 1);
        assert_eq!(snapshot.learning_allowed, 0);
        assert_eq!(snapshot.cumulative_learning_scale, FixedPoint::ZERO);
        assert!(snapshot.flags.has(SupervisionFlags::HAS_EXPLORATION));
    }

    #[test]
    fn repeated_recovery_quarantines_episode() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = gate(
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

        let first = ledger.record(&decision);
        let second = ledger.record(&decision);

        assert_eq!(first.status, SupervisionStatus::Recovering);
        assert_eq!(second.status, SupervisionStatus::Quarantined);
        assert_eq!(second.recovery_streak, 2);
        assert_eq!(second.max_risk, CycleRisk::Critical);
        assert!(second.flags.has(SupervisionFlags::RECOVERY_STREAK_LIMIT));
    }

    #[test]
    fn non_recovery_cycle_resets_recovery_streak_but_keeps_watch_status() {
        let mut ledger = SupervisionLedger::conservative();
        let recovery = gate(
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
        let learning = gate(
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

        ledger.record(&recovery);
        let snapshot = ledger.record(&learning);

        assert_eq!(snapshot.status, SupervisionStatus::Watching);
        assert_eq!(snapshot.recovery_streak, 0);
        assert_eq!(snapshot.max_recovery_streak_seen, 1);
        assert_eq!(snapshot.learning_allowed, 1);
        assert!(snapshot.flags.has(SupervisionFlags::CRITICAL_SEEN));
    }

    #[test]
    fn summary_hash_changes_with_mode_sequence() {
        let recovery = gate(
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
        let observe = gate(
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

        let mut a = SupervisionLedger::conservative();
        let mut b = SupervisionLedger::conservative();
        a.record(&recovery);
        a.record(&observe);
        b.record(&observe);
        b.record(&recovery);

        assert_ne!(a.snapshot().summary_hash, b.snapshot().summary_hash);
    }

    #[test]
    fn forged_learning_decision_is_observed_not_counted_as_learning() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = RuntimeGateDecision {
            mode: RuntimeGateMode::LearningAllowed,
            flags: RuntimeGateFlags(RuntimeGateFlags::NO_EFFECT_AUTH),
            readiness_level: crate::cognition::readiness::ReadinessLevel::AgenticCandidate,
            cycle_risk: CycleRisk::Nominal,
            trace_id: 42,
            checkpoint_required: false,
            learning_budget_ticks: 4,
            learning_scale: FixedPoint::ONE,
        };

        let snapshot = ledger.record(&decision);

        assert_eq!(snapshot.last_mode, RuntimeGateMode::ObserveOnly);
        assert_eq!(snapshot.learning_allowed, 0);
        assert_eq!(snapshot.observed, 1);
        assert!(snapshot.flags.has(SupervisionFlags::HAS_OBSERVED));
        assert!(!snapshot.flags.has(SupervisionFlags::HAS_LEARNING));
    }

    #[test]
    fn forged_recovery_mode_without_evidence_is_observed() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = RuntimeGateDecision {
            mode: RuntimeGateMode::RecoveryOnly,
            flags: RuntimeGateFlags::empty(),
            readiness_level: crate::cognition::readiness::ReadinessLevel::AgenticCandidate,
            cycle_risk: CycleRisk::Nominal,
            trace_id: 77,
            checkpoint_required: false,
            learning_budget_ticks: 0,
            learning_scale: FixedPoint::ZERO,
        };

        let snapshot = ledger.record(&decision);

        assert_eq!(snapshot.last_mode, RuntimeGateMode::ObserveOnly);
        assert_eq!(snapshot.recovery, 0);
        assert_eq!(snapshot.observed, 1);
        assert_eq!(snapshot.recovery_streak, 0);
        assert!(!snapshot.flags.has(SupervisionFlags::HAS_RECOVERY));
    }

    #[test]
    fn forged_blocked_mode_without_readiness_flag_is_observed() {
        let mut ledger = SupervisionLedger::conservative();
        let decision = RuntimeGateDecision {
            mode: RuntimeGateMode::Blocked,
            flags: RuntimeGateFlags::empty(),
            readiness_level: crate::cognition::readiness::ReadinessLevel::AgenticCandidate,
            cycle_risk: CycleRisk::Nominal,
            trace_id: 88,
            checkpoint_required: false,
            learning_budget_ticks: 0,
            learning_scale: FixedPoint::ZERO,
        };

        let snapshot = ledger.record(&decision);

        assert_eq!(snapshot.last_mode, RuntimeGateMode::ObserveOnly);
        assert_eq!(snapshot.blocked, 0);
        assert_eq!(snapshot.observed, 1);
        assert!(!snapshot.flags.has(SupervisionFlags::HAS_BLOCKED));
        assert!(snapshot.flags.has(SupervisionFlags::HAS_OBSERVED));
    }
}
