//! Deterministic scenario evaluation for HKL-2 cognition.
//!
//! Scenarios exercise the cognition controller across learning, exploration,
//! budget starvation, and recovery paths. They provide a tiny repeatable eval
//! surface before larger agent-like loops are allowed to grow.

use crate::cognition::audit::{CycleAuditFlags, CycleAuditRecord, CycleRisk};
use crate::cognition::controller::{CognitiveController, CycleSignals};
use crate::cognition::executive::ExecutiveAction;
use crate::cognition::planner::PlanStepKind;
use crate::core::math::FixedPoint;
use crate::training::trainer::{TrainStepReport, TrainStepStatus};

pub const MAX_SCENARIOS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenarioKind {
    HealthyLearning,
    NoveltyExploration,
    SaturatedLossRecovery,
    SafetyPressureRecovery,
    BudgetStarvedLearning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScenarioExpectation {
    pub action: ExecutiveAction,
    pub risk: CycleRisk,
    pub may_apply_learning: bool,
    pub must_recover: bool,
    pub external_effects_allowed: bool,
    pub checkpoint_required: bool,
    pub has_learning_budget: bool,
    pub first_step: PlanStepKind,
}

#[derive(Clone, Copy, Debug)]
pub struct CognitiveScenario {
    pub kind: ScenarioKind,
    pub preview_report: TrainStepReport,
    pub signals: CycleSignals,
    pub max_plan_budget_ticks: u16,
    pub expected: ScenarioExpectation,
}

impl CognitiveScenario {
    pub const fn empty() -> Self {
        Self {
            kind: ScenarioKind::HealthyLearning,
            preview_report: report_const(0, 0, 0, 0, TrainStepStatus::Empty, FixedPoint::ZERO),
            signals: CycleSignals::neutral(),
            max_plan_budget_ticks: 0,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Idle,
                risk: CycleRisk::Elevated,
                may_apply_learning: false,
                must_recover: false,
                external_effects_allowed: false,
                checkpoint_required: false,
                has_learning_budget: false,
                first_step: PlanStepKind::Idle,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScenarioMismatchFlags(pub u16);

impl ScenarioMismatchFlags {
    pub const ACTION: u16 = 1 << 0;
    pub const RISK: u16 = 1 << 1;
    pub const APPLY: u16 = 1 << 2;
    pub const RECOVER: u16 = 1 << 3;
    pub const FIRST_STEP: u16 = 1 << 4;
    pub const EXTERNAL_EFFECTS: u16 = 1 << 5;
    pub const CHECKPOINT: u16 = 1 << 6;
    pub const LEARNING_BUDGET: u16 = 1 << 7;

    pub const fn empty() -> Self {
        Self(0)
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
pub struct ScenarioOutcome {
    pub kind: ScenarioKind,
    pub audit: CycleAuditRecord,
    pub passed: bool,
    pub mismatches: ScenarioMismatchFlags,
}

impl ScenarioOutcome {
    pub const fn empty() -> Self {
        Self {
            kind: ScenarioKind::HealthyLearning,
            audit: CycleAuditRecord::empty(),
            passed: false,
            mismatches: ScenarioMismatchFlags::empty(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScenarioSuiteReport {
    pub outcomes: [ScenarioOutcome; MAX_SCENARIOS],
    pub requested: usize,
    pub len: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub score: ScenarioSuiteScore,
    pub summary_hash: u32,
}

impl ScenarioSuiteReport {
    pub const fn empty() -> Self {
        Self {
            outcomes: [ScenarioOutcome::empty(); MAX_SCENARIOS],
            requested: 0,
            len: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            score: ScenarioSuiteScore::empty(),
            summary_hash: 0,
        }
    }

    pub const fn all_evaluated_and_passed(&self) -> bool {
        self.len > 0 && self.failed == 0 && self.skipped == 0 && self.passed == self.len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScenarioSuiteScore {
    pub earned_points: u16,
    pub possible_points: u16,
    pub capability_per_mille: u16,
    pub learning_points: u16,
    pub exploration_points: u16,
    pub recovery_points: u16,
    pub restraint_points: u16,
}

impl ScenarioSuiteScore {
    pub const fn empty() -> Self {
        Self {
            earned_points: 0,
            possible_points: 0,
            capability_per_mille: 0,
            learning_points: 0,
            exploration_points: 0,
            recovery_points: 0,
            restraint_points: 0,
        }
    }
}

pub fn run_default_scenarios() -> ScenarioSuiteReport {
    evaluate_scenarios(&default_scenarios())
}

pub fn evaluate_scenarios(scenarios: &[CognitiveScenario]) -> ScenarioSuiteReport {
    let mut report = ScenarioSuiteReport::empty();
    let mut hash = 0x9e3779b9_u32;
    report.requested = scenarios.len();
    report.skipped = scenarios.len().saturating_sub(MAX_SCENARIOS);

    for scenario in scenarios.iter().take(MAX_SCENARIOS) {
        let outcome = evaluate_scenario(*scenario);
        hash = mix(hash, scenario_id(outcome.kind));
        hash = mix(hash, bool_id(outcome.passed));
        hash = mix(hash, action_id(outcome.audit.action));
        hash = mix(hash, risk_id(outcome.audit.risk));
        hash = mix(hash, step_id(outcome.audit.first_step));
        hash = mix(hash, outcome.audit.trace_id);
        hash = mix(hash, outcome.audit.flags.0 as u32);
        hash = mix(hash, outcome.audit.budget_ticks as u32);
        hash = mix(hash, outcome.audit.learning_budget_ticks as u32);
        hash = mix(hash, outcome.audit.plan_len as u32);
        hash = mix(hash, outcome.audit.tokens_seen as u32);
        hash = mix(hash, outcome.mismatches.0 as u32);

        if outcome.passed {
            report.passed += 1;
        } else {
            report.failed += 1;
        }
        report.outcomes[report.len] = outcome;
        report.len += 1;
    }

    report.score = score_outcomes(&report.outcomes, report.len, report.skipped);
    hash = mix(hash, report.score.capability_per_mille as u32);
    hash = mix(hash, report.score.earned_points as u32);
    hash = mix(hash, report.score.possible_points as u32);
    report.summary_hash = hash;
    report
}

pub fn evaluate_scenario(scenario: CognitiveScenario) -> ScenarioOutcome {
    let mut controller = CognitiveController::conservative(scenario.max_plan_budget_ticks);
    let cycle = controller.preview_cycle(scenario.preview_report, scenario.signals);
    let audit = CycleAuditRecord::from_cycle(&cycle);
    let mut mismatches = ScenarioMismatchFlags::empty();

    mismatches.set(
        ScenarioMismatchFlags::ACTION,
        audit.action != scenario.expected.action,
    );
    mismatches.set(
        ScenarioMismatchFlags::RISK,
        audit.risk != scenario.expected.risk,
    );
    mismatches.set(
        ScenarioMismatchFlags::APPLY,
        cycle.may_apply_learning != scenario.expected.may_apply_learning,
    );
    mismatches.set(
        ScenarioMismatchFlags::RECOVER,
        cycle.must_recover != scenario.expected.must_recover,
    );
    mismatches.set(
        ScenarioMismatchFlags::EXTERNAL_EFFECTS,
        audit.allows_external_effects() != scenario.expected.external_effects_allowed,
    );
    mismatches.set(
        ScenarioMismatchFlags::CHECKPOINT,
        audit.flags.has(CycleAuditFlags::CHECKPOINT_REQUIRED)
            != scenario.expected.checkpoint_required,
    );
    mismatches.set(
        ScenarioMismatchFlags::LEARNING_BUDGET,
        (audit.learning_budget_ticks > 0) != scenario.expected.has_learning_budget,
    );
    mismatches.set(
        ScenarioMismatchFlags::FIRST_STEP,
        audit.first_step != scenario.expected.first_step,
    );

    ScenarioOutcome {
        kind: scenario.kind,
        audit,
        passed: mismatches.is_empty(),
        mismatches,
    }
}

pub const fn default_scenarios() -> [CognitiveScenario; MAX_SCENARIOS] {
    [
        CognitiveScenario {
            kind: ScenarioKind::HealthyLearning,
            preview_report: report_const(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            signals: CycleSignals::neutral(),
            max_plan_budget_ticks: 16,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Learn,
                risk: CycleRisk::Elevated,
                may_apply_learning: true,
                must_recover: false,
                external_effects_allowed: true,
                checkpoint_required: true,
                has_learning_budget: true,
                first_step: PlanStepKind::SafetyCheck,
            },
        },
        CognitiveScenario {
            kind: ScenarioKind::NoveltyExploration,
            preview_report: report_const(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            signals: CycleSignals {
                curiosity: FixedPoint::from_f32(0.8),
                novelty: FixedPoint::from_f32(0.3),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::ZERO,
            },
            max_plan_budget_ticks: 32,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Explore,
                risk: CycleRisk::Elevated,
                may_apply_learning: true,
                must_recover: false,
                external_effects_allowed: true,
                checkpoint_required: true,
                has_learning_budget: true,
                first_step: PlanStepKind::SafetyCheck,
            },
        },
        CognitiveScenario {
            kind: ScenarioKind::SaturatedLossRecovery,
            preview_report: report_const(
                4,
                0,
                0,
                1,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(8192.0),
            ),
            signals: CycleSignals::neutral(),
            max_plan_budget_ticks: 16,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Recover,
                risk: CycleRisk::Critical,
                may_apply_learning: false,
                must_recover: true,
                external_effects_allowed: false,
                checkpoint_required: true,
                has_learning_budget: false,
                first_step: PlanStepKind::SafetyCheck,
            },
        },
        CognitiveScenario {
            kind: ScenarioKind::SafetyPressureRecovery,
            preview_report: report_const(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            signals: CycleSignals {
                curiosity: FixedPoint::from_f32(0.8),
                novelty: FixedPoint::from_f32(0.8),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::from_f32(0.9),
            },
            max_plan_budget_ticks: 16,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Recover,
                risk: CycleRisk::Critical,
                may_apply_learning: false,
                must_recover: true,
                external_effects_allowed: false,
                checkpoint_required: true,
                has_learning_budget: false,
                first_step: PlanStepKind::SafetyCheck,
            },
        },
        CognitiveScenario {
            kind: ScenarioKind::BudgetStarvedLearning,
            preview_report: report_const(
                16,
                0,
                0,
                0,
                TrainStepStatus::Complete,
                FixedPoint::from_f32(4.0),
            ),
            signals: CycleSignals::neutral(),
            max_plan_budget_ticks: 2,
            expected: ScenarioExpectation {
                action: ExecutiveAction::Learn,
                risk: CycleRisk::Nominal,
                may_apply_learning: false,
                must_recover: false,
                external_effects_allowed: false,
                checkpoint_required: false,
                has_learning_budget: false,
                first_step: PlanStepKind::SafetyCheck,
            },
        },
    ]
}

const fn report_const(
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

fn mix(hash: u32, value: u32) -> u32 {
    hash.rotate_left(5) ^ value.wrapping_mul(0x85eb_ca6b)
}

fn scenario_id(kind: ScenarioKind) -> u32 {
    match kind {
        ScenarioKind::HealthyLearning => 1,
        ScenarioKind::NoveltyExploration => 2,
        ScenarioKind::SaturatedLossRecovery => 3,
        ScenarioKind::SafetyPressureRecovery => 4,
        ScenarioKind::BudgetStarvedLearning => 5,
    }
}

fn bool_id(value: bool) -> u32 {
    if value { 1 } else { 0 }
}

fn action_id(action: ExecutiveAction) -> u32 {
    match action {
        ExecutiveAction::Learn => 11,
        ExecutiveAction::Explore => 12,
        ExecutiveAction::Consolidate => 13,
        ExecutiveAction::Recover => 14,
        ExecutiveAction::Idle => 15,
    }
}

fn risk_id(risk: CycleRisk) -> u32 {
    match risk {
        CycleRisk::Nominal => 21,
        CycleRisk::Elevated => 22,
        CycleRisk::Critical => 23,
    }
}

fn step_id(step: PlanStepKind) -> u32 {
    match step {
        PlanStepKind::SafetyCheck => 31,
        PlanStepKind::PreviewTraining => 32,
        PlanStepKind::ApplyLearning => 33,
        PlanStepKind::ExploreProbe => 34,
        PlanStepKind::ConsolidateMemory => 35,
        PlanStepKind::RecoverState => 36,
        PlanStepKind::EmitTelemetry => 37,
        PlanStepKind::Idle => 38,
    }
}

const EXPECTATION_POINTS: u16 = 8;

fn score_outcomes(
    outcomes: &[ScenarioOutcome; MAX_SCENARIOS],
    len: usize,
    skipped: usize,
) -> ScenarioSuiteScore {
    let mut score = ScenarioSuiteScore::empty();

    for outcome in outcomes.iter().take(len.min(MAX_SCENARIOS)) {
        score.possible_points = score.possible_points.saturating_add(EXPECTATION_POINTS);
        score.earned_points = score
            .earned_points
            .saturating_add(outcome_score_points(*outcome));

        match outcome.kind {
            ScenarioKind::HealthyLearning => {
                if outcome.passed
                    && outcome.audit.flags.has(CycleAuditFlags::APPLY_LEARNING)
                    && outcome.audit.learning_budget_ticks > 0
                {
                    score.learning_points = score.learning_points.saturating_add(1);
                }
            }
            ScenarioKind::NoveltyExploration => {
                if outcome.passed
                    && outcome.audit.action == ExecutiveAction::Explore
                    && outcome.audit.flags.has(CycleAuditFlags::APPLY_LEARNING)
                {
                    score.exploration_points = score.exploration_points.saturating_add(1);
                }
            }
            ScenarioKind::SaturatedLossRecovery | ScenarioKind::SafetyPressureRecovery => {
                if outcome.passed
                    && outcome.audit.risk == CycleRisk::Critical
                    && outcome.audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED)
                    && !outcome.audit.allows_external_effects()
                {
                    score.recovery_points = score.recovery_points.saturating_add(1);
                }
            }
            ScenarioKind::BudgetStarvedLearning => {
                if outcome.passed
                    && !outcome.audit.flags.has(CycleAuditFlags::APPLY_LEARNING)
                    && !outcome.audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED)
                    && outcome.audit.learning_budget_ticks == 0
                {
                    score.restraint_points = score.restraint_points.saturating_add(1);
                }
            }
        }
    }

    if skipped > 0 {
        score.possible_points = score.possible_points.saturating_add(count_points(skipped));
    }

    if score.possible_points > 0 {
        score.capability_per_mille =
            ((score.earned_points as u32 * 1000) / score.possible_points as u32) as u16;
    }

    score
}

fn outcome_score_points(outcome: ScenarioOutcome) -> u16 {
    EXPECTATION_POINTS.saturating_sub(outcome.mismatches.0.count_ones() as u16)
}

fn count_points(count: usize) -> u16 {
    let capped = count.min((u16::MAX / EXPECTATION_POINTS) as usize) as u16;
    capped.saturating_mul(EXPECTATION_POINTS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::audit::CycleAuditFlags;

    #[test]
    fn default_scenarios_all_pass() {
        let report = run_default_scenarios();

        assert_eq!(report.len, MAX_SCENARIOS);
        assert_eq!(report.requested, MAX_SCENARIOS);
        assert_eq!(report.passed, MAX_SCENARIOS);
        assert_eq!(report.failed, 0);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.score.capability_per_mille, 1000);
        assert_eq!(report.score.learning_points, 1);
        assert_eq!(report.score.exploration_points, 1);
        assert_eq!(report.score.recovery_points, 2);
        assert_eq!(report.score.restraint_points, 1);
        assert_ne!(report.summary_hash, 0);
        assert!(report.all_evaluated_and_passed());
    }

    #[test]
    fn oversized_scenario_suites_report_skipped_cases() {
        let scenarios = [
            default_scenarios()[0],
            default_scenarios()[1],
            default_scenarios()[2],
            default_scenarios()[3],
            default_scenarios()[4],
            default_scenarios()[0],
        ];

        let report = evaluate_scenarios(&scenarios);

        assert_eq!(report.len, MAX_SCENARIOS);
        assert_eq!(report.requested, MAX_SCENARIOS + 1);
        assert_eq!(report.skipped, 1);
        assert!(report.score.capability_per_mille < 1000);
        assert!(!report.all_evaluated_and_passed());
    }

    #[test]
    fn skipped_score_penalty_scales_with_all_ignored_scenarios() {
        let scenarios = [default_scenarios()[0]; MAX_SCENARIOS + 10];

        let report = evaluate_scenarios(&scenarios);

        assert_eq!(report.len, MAX_SCENARIOS);
        assert_eq!(report.requested, MAX_SCENARIOS + 10);
        assert_eq!(report.skipped, 10);
        assert_eq!(
            report.score.possible_points,
            count_points(MAX_SCENARIOS + 10)
        );
        assert!(report.score.capability_per_mille < 500);
    }

    #[test]
    fn mismatch_flags_report_wrong_expectation() {
        let mut scenario = default_scenarios()[0];
        scenario.expected.action = ExecutiveAction::Recover;

        let outcome = evaluate_scenario(scenario);

        assert!(!outcome.passed);
        assert!(outcome.mismatches.0 & ScenarioMismatchFlags::ACTION != 0);
        assert!(outcome_score_points(outcome) < EXPECTATION_POINTS);
    }

    #[test]
    fn mismatch_flags_report_effect_and_budget_expectations() {
        let mut scenario = default_scenarios()[0];
        scenario.expected.external_effects_allowed = false;
        scenario.expected.checkpoint_required = false;
        scenario.expected.has_learning_budget = false;

        let outcome = evaluate_scenario(scenario);

        assert!(!outcome.passed);
        assert!(outcome.mismatches.0 & ScenarioMismatchFlags::EXTERNAL_EFFECTS != 0);
        assert!(outcome.mismatches.0 & ScenarioMismatchFlags::CHECKPOINT != 0);
        assert!(outcome.mismatches.0 & ScenarioMismatchFlags::LEARNING_BUDGET != 0);
    }

    #[test]
    fn recovery_scenarios_are_critical_and_non_effectful() {
        let report = run_default_scenarios();

        for outcome in report.outcomes.iter().take(report.len) {
            if matches!(
                outcome.kind,
                ScenarioKind::SaturatedLossRecovery | ScenarioKind::SafetyPressureRecovery
            ) {
                assert_eq!(outcome.audit.risk, CycleRisk::Critical);
                assert!(outcome.audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED));
                assert!(!outcome.audit.allows_external_effects());
            }
        }
    }

    #[test]
    fn budget_starved_scenario_blocks_learning_without_recovery() {
        let scenario = default_scenarios()[4];
        let outcome = evaluate_scenario(scenario);

        assert!(outcome.passed);
        assert_eq!(outcome.kind, ScenarioKind::BudgetStarvedLearning);
        assert!(!outcome.audit.flags.has(CycleAuditFlags::APPLY_LEARNING));
        assert!(!outcome.audit.flags.has(CycleAuditFlags::RECOVERY_REQUIRED));
        assert_eq!(outcome.audit.learning_budget_ticks, 0);
    }

    #[test]
    fn scenario_hash_changes_when_effective_budget_changes() {
        let mut budget_one = default_scenarios()[4];
        budget_one.max_plan_budget_ticks = 1;

        let budget_two = default_scenarios()[4];

        let one = evaluate_scenarios(&[budget_one]);
        let two = evaluate_scenarios(&[budget_two]);

        assert_eq!(one.outcomes[0].audit.budget_ticks, 1);
        assert_eq!(two.outcomes[0].audit.budget_ticks, 2);
        assert_eq!(one.outcomes[0].mismatches, two.outcomes[0].mismatches);
        assert_ne!(one.summary_hash, two.summary_hash);
    }

    #[test]
    fn suite_score_drops_when_expectations_do_not_match() {
        let mut scenarios = default_scenarios();
        scenarios[0].expected.action = ExecutiveAction::Recover;

        let report = evaluate_scenarios(&scenarios);

        assert_eq!(report.len, MAX_SCENARIOS);
        assert_eq!(report.failed, 1);
        assert!(report.score.capability_per_mille < 1000);
        assert!(report.score.earned_points < report.score.possible_points);
    }
}
