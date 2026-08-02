//! Readiness gates for HKL-2 cognition scenario suites.
//!
//! A readiness gate converts deterministic scenario scores into an explicit
//! maturity level. This keeps larger agent-like loops behind measurable
//! learning, exploration, recovery, and restraint evidence.

use crate::cognition::scenario::{MAX_SCENARIOS, ScenarioSuiteReport};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReadinessLevel {
    Blocked,
    ObserveOnly,
    LearningReady,
    AdaptiveReady,
    AgenticCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessFlags(pub u16);

impl ReadinessFlags {
    pub const NO_SCENARIOS: u16 = 1 << 0;
    pub const SKIPPED_SCENARIOS: u16 = 1 << 1;
    pub const FAILED_SCENARIOS: u16 = 1 << 2;
    pub const LOW_SCORE: u16 = 1 << 3;
    pub const MISSING_LEARNING: u16 = 1 << 4;
    pub const MISSING_EXPLORATION: u16 = 1 << 5;
    pub const MISSING_RECOVERY: u16 = 1 << 6;
    pub const MISSING_RESTRAINT: u16 = 1 << 7;

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
pub struct ReadinessPolicy {
    pub min_learning_per_mille: u16,
    pub min_adaptive_per_mille: u16,
    pub min_agentic_per_mille: u16,
    pub required_recovery_points: u16,
}

impl ReadinessPolicy {
    pub const fn conservative() -> Self {
        Self {
            min_learning_per_mille: 600,
            min_adaptive_per_mille: 800,
            min_agentic_per_mille: 1000,
            required_recovery_points: 2,
        }
    }

    pub const fn normalized(self) -> Self {
        let min_learning_per_mille = clamp_per_mille(self.min_learning_per_mille);
        let mut min_adaptive_per_mille = clamp_per_mille(self.min_adaptive_per_mille);
        if min_adaptive_per_mille < min_learning_per_mille {
            min_adaptive_per_mille = min_learning_per_mille;
        }
        let mut min_agentic_per_mille = clamp_per_mille(self.min_agentic_per_mille);
        if min_agentic_per_mille < min_adaptive_per_mille {
            min_agentic_per_mille = min_adaptive_per_mille;
        }

        Self {
            min_learning_per_mille,
            min_adaptive_per_mille,
            min_agentic_per_mille,
            required_recovery_points: clamp_recovery_points(self.required_recovery_points),
        }
    }
}

const fn clamp_per_mille(value: u16) -> u16 {
    if value > 1000 { 1000 } else { value }
}

const fn clamp_recovery_points(value: u16) -> u16 {
    if value as usize > MAX_SCENARIOS {
        MAX_SCENARIOS as u16
    } else {
        value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReadinessReport {
    pub level: ReadinessLevel,
    pub flags: ReadinessFlags,
    pub capability_per_mille: u16,
    pub evaluated: usize,
    pub requested: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub learning_points: u16,
    pub exploration_points: u16,
    pub recovery_points: u16,
    pub restraint_points: u16,
}

impl ReadinessReport {
    pub const fn blocked() -> Self {
        Self {
            level: ReadinessLevel::Blocked,
            flags: ReadinessFlags(ReadinessFlags::NO_SCENARIOS),
            capability_per_mille: 0,
            evaluated: 0,
            requested: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            learning_points: 0,
            exploration_points: 0,
            recovery_points: 0,
            restraint_points: 0,
        }
    }

    pub const fn permits_agentic_loop(&self) -> bool {
        matches!(self.level, ReadinessLevel::AgenticCandidate)
            && self.flags.is_empty()
            && self.evaluated > 0
            && self.requested == self.evaluated
            && self.passed == self.evaluated
            && self.failed == 0
            && self.skipped == 0
            && self.capability_per_mille == 1000
            && self.learning_points > 0
            && self.exploration_points > 0
            && self.recovery_points > 0
            && self.restraint_points > 0
    }
}

pub fn evaluate_readiness(suite: &ScenarioSuiteReport) -> ReadinessReport {
    evaluate_readiness_with_policy(suite, ReadinessPolicy::conservative())
}

pub fn evaluate_readiness_with_policy(
    suite: &ScenarioSuiteReport,
    policy: ReadinessPolicy,
) -> ReadinessReport {
    let policy = policy.normalized();
    if suite.len == 0 {
        return ReadinessReport::blocked();
    }

    let score = suite.score;
    let mut flags = ReadinessFlags::empty();
    flags.set(ReadinessFlags::NO_SCENARIOS, suite.len == 0);
    flags.set(ReadinessFlags::SKIPPED_SCENARIOS, suite.skipped > 0);
    flags.set(ReadinessFlags::FAILED_SCENARIOS, suite.failed > 0);
    flags.set(
        ReadinessFlags::LOW_SCORE,
        score.capability_per_mille < policy.min_learning_per_mille,
    );
    flags.set(ReadinessFlags::MISSING_LEARNING, score.learning_points == 0);
    flags.set(
        ReadinessFlags::MISSING_EXPLORATION,
        score.exploration_points == 0,
    );
    flags.set(
        ReadinessFlags::MISSING_RECOVERY,
        score.recovery_points < policy.required_recovery_points,
    );
    flags.set(
        ReadinessFlags::MISSING_RESTRAINT,
        score.restraint_points == 0,
    );

    let structural_block = suite.failed > 0 || suite.skipped > 0;
    let level = if structural_block || flags.has(ReadinessFlags::NO_SCENARIOS) {
        ReadinessLevel::Blocked
    } else if score.capability_per_mille >= policy.min_agentic_per_mille
        && score.learning_points > 0
        && score.exploration_points > 0
        && score.recovery_points >= policy.required_recovery_points
        && score.restraint_points > 0
    {
        ReadinessLevel::AgenticCandidate
    } else if score.capability_per_mille >= policy.min_adaptive_per_mille
        && score.learning_points > 0
        && score.recovery_points > 0
    {
        ReadinessLevel::AdaptiveReady
    } else if score.capability_per_mille >= policy.min_learning_per_mille
        && score.learning_points > 0
    {
        ReadinessLevel::LearningReady
    } else {
        ReadinessLevel::ObserveOnly
    };

    ReadinessReport {
        level,
        flags,
        capability_per_mille: score.capability_per_mille,
        evaluated: suite.len,
        requested: suite.requested,
        passed: suite.passed,
        failed: suite.failed,
        skipped: suite.skipped,
        learning_points: score.learning_points,
        exploration_points: score.exploration_points,
        recovery_points: score.recovery_points,
        restraint_points: score.restraint_points,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::executive::ExecutiveAction;
    use crate::cognition::scenario::{MAX_SCENARIOS, ScenarioKind};
    use crate::cognition::scenario::{default_scenarios, evaluate_scenarios};

    #[test]
    fn default_suite_is_agentic_candidate() {
        let suite = evaluate_scenarios(&default_scenarios());
        let readiness = evaluate_readiness(&suite);

        assert_eq!(readiness.level, ReadinessLevel::AgenticCandidate);
        assert!(readiness.flags.is_empty());
        assert!(readiness.permits_agentic_loop());
        assert_eq!(readiness.capability_per_mille, 1000);
        assert_eq!(readiness.recovery_points, 2);
    }

    #[test]
    fn skipped_scenarios_block_readiness() {
        let scenarios = [
            default_scenarios()[0],
            default_scenarios()[1],
            default_scenarios()[2],
            default_scenarios()[3],
            default_scenarios()[4],
            default_scenarios()[0],
        ];
        let suite = evaluate_scenarios(&scenarios);
        let readiness = evaluate_readiness(&suite);

        assert_eq!(suite.requested, MAX_SCENARIOS + 1);
        assert_eq!(readiness.level, ReadinessLevel::Blocked);
        assert!(readiness.flags.has(ReadinessFlags::SKIPPED_SCENARIOS));
        assert!(!readiness.permits_agentic_loop());
    }

    #[test]
    fn failed_scenario_blocks_readiness() {
        let mut scenarios = default_scenarios();
        scenarios[0].expected.action = ExecutiveAction::Recover;

        let suite = evaluate_scenarios(&scenarios);
        let readiness = evaluate_readiness(&suite);

        assert_eq!(readiness.level, ReadinessLevel::Blocked);
        assert!(readiness.flags.has(ReadinessFlags::FAILED_SCENARIOS));
        assert!(!readiness.permits_agentic_loop());
    }

    #[test]
    fn missing_recovery_caps_at_adaptive_ready() {
        let scenarios = [default_scenarios()[0], default_scenarios()[1]];
        let suite = evaluate_scenarios(&scenarios);
        let readiness = evaluate_readiness_with_policy(
            &suite,
            ReadinessPolicy {
                min_learning_per_mille: 500,
                min_adaptive_per_mille: 900,
                min_agentic_per_mille: 1000,
                required_recovery_points: 1,
            },
        );

        assert_eq!(suite.outcomes[0].kind, ScenarioKind::HealthyLearning);
        assert_eq!(readiness.level, ReadinessLevel::LearningReady);
        assert!(readiness.flags.has(ReadinessFlags::MISSING_RECOVERY));
    }

    #[test]
    fn empty_suite_is_blocked() {
        let suite = evaluate_scenarios(&[]);
        let readiness = evaluate_readiness(&suite);

        assert_eq!(readiness.level, ReadinessLevel::Blocked);
        assert!(readiness.flags.has(ReadinessFlags::NO_SCENARIOS));
    }

    #[test]
    fn readiness_policy_normalizes_inverted_thresholds() {
        let policy = ReadinessPolicy {
            min_learning_per_mille: 900,
            min_adaptive_per_mille: 100,
            min_agentic_per_mille: 50,
            required_recovery_points: u16::MAX,
        }
        .normalized();

        assert_eq!(policy.min_learning_per_mille, 900);
        assert_eq!(policy.min_adaptive_per_mille, 900);
        assert_eq!(policy.min_agentic_per_mille, 900);
        assert_eq!(policy.required_recovery_points, MAX_SCENARIOS as u16);
    }

    #[test]
    fn inverted_policy_cannot_promote_failed_suite_to_agentic() {
        let mut scenarios = default_scenarios();
        scenarios[0].expected.action = ExecutiveAction::Recover;
        let suite = evaluate_scenarios(&scenarios);
        let readiness = evaluate_readiness_with_policy(
            &suite,
            ReadinessPolicy {
                min_learning_per_mille: 1000,
                min_adaptive_per_mille: 0,
                min_agentic_per_mille: 0,
                required_recovery_points: 0,
            },
        );

        assert_eq!(readiness.level, ReadinessLevel::Blocked);
        assert!(readiness.flags.has(ReadinessFlags::FAILED_SCENARIOS));
        assert!(!readiness.permits_agentic_loop());
    }

    #[test]
    fn permits_agentic_loop_rejects_forged_empty_report() {
        let mut readiness = ReadinessReport::blocked();
        readiness.level = ReadinessLevel::AgenticCandidate;
        readiness.flags = ReadinessFlags::empty();
        readiness.capability_per_mille = 1000;
        readiness.learning_points = 1;
        readiness.exploration_points = 1;
        readiness.recovery_points = 2;
        readiness.restraint_points = 1;

        assert!(!readiness.permits_agentic_loop());
    }

    #[test]
    fn permits_agentic_loop_rejects_inconsistent_counts() {
        let suite = evaluate_scenarios(&default_scenarios());
        let mut readiness = evaluate_readiness(&suite);
        assert!(readiness.permits_agentic_loop());

        readiness.requested = readiness.evaluated + 1;
        assert!(!readiness.permits_agentic_loop());

        readiness = evaluate_readiness(&suite);
        readiness.passed = readiness.passed.saturating_sub(1);
        assert!(!readiness.permits_agentic_loop());
    }
}
