//! Reusable HKL-2 cognition episode runner.
//!
//! The runner is the single dry-run authorization path for longer loops:
//! controller, audit, runtime gate, and supervision ledger are composed before
//! any caller is allowed to mutate model weights or trigger exploratory work.

use crate::cognition::audit::CycleAuditRecord;
use crate::cognition::controller::{CognitiveController, CognitiveCycleReport, CycleSignals};
use crate::cognition::readiness::ReadinessReport;
use crate::cognition::runtime_gate::{RuntimeGateDecision, RuntimeGateMode, evaluate_runtime_gate};
use crate::cognition::supervisor::{SupervisionLedger, SupervisionSnapshot, SupervisionStatus};
use crate::training::trainer::TrainStepReport;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeRecommendation {
    ApplyLearning,
    Probe,
    Recover,
    Observe,
    Stop,
}

#[derive(Clone, Copy, Debug)]
pub struct CognitiveEpisodeCycle {
    pub cycle: CognitiveCycleReport,
    pub audit: CycleAuditRecord,
    pub runtime_gate: RuntimeGateDecision,
    pub supervision: SupervisionSnapshot,
    pub recommendation: EpisodeRecommendation,
}

impl CognitiveEpisodeCycle {
    pub fn allows_model_update(&self) -> bool {
        matches!(self.recommendation, EpisodeRecommendation::ApplyLearning)
            && self.runtime_gate.permits_learning()
            && !matches!(self.supervision.status, SupervisionStatus::Quarantined)
    }

    pub fn allows_probe(&self) -> bool {
        matches!(self.recommendation, EpisodeRecommendation::Probe)
            && self.runtime_gate.permits_exploration()
            && !matches!(self.supervision.status, SupervisionStatus::Quarantined)
    }

    pub const fn should_stop_unattended(&self) -> bool {
        matches!(self.recommendation, EpisodeRecommendation::Stop)
    }
}

pub struct CognitiveEpisodeRunner {
    pub readiness: ReadinessReport,
    pub controller: CognitiveController,
    pub supervisor: SupervisionLedger,
}

impl CognitiveEpisodeRunner {
    pub fn conservative(readiness: ReadinessReport, max_plan_budget_ticks: u16) -> Self {
        Self::new(
            readiness,
            CognitiveController::conservative(max_plan_budget_ticks),
            SupervisionLedger::conservative(),
        )
    }

    pub const fn new(
        readiness: ReadinessReport,
        controller: CognitiveController,
        supervisor: SupervisionLedger,
    ) -> Self {
        Self {
            readiness,
            controller,
            supervisor,
        }
    }

    pub fn preview_cycle(
        &mut self,
        preview_report: TrainStepReport,
        signals: CycleSignals,
    ) -> CognitiveEpisodeCycle {
        let cycle = self.controller.preview_cycle(preview_report, signals);
        let audit = CycleAuditRecord::from_cycle(&cycle);
        let runtime_gate = evaluate_runtime_gate(&self.readiness, &audit);
        let supervision = self.supervisor.record(&runtime_gate);
        let recommendation = recommend(&runtime_gate, &supervision);

        CognitiveEpisodeCycle {
            cycle,
            audit,
            runtime_gate,
            supervision,
            recommendation,
        }
    }

    pub const fn supervision(&self) -> SupervisionSnapshot {
        self.supervisor.snapshot()
    }

    pub const fn readiness(&self) -> ReadinessReport {
        self.readiness
    }
}

fn recommend(
    runtime_gate: &RuntimeGateDecision,
    supervision: &SupervisionSnapshot,
) -> EpisodeRecommendation {
    if matches!(runtime_gate.mode, RuntimeGateMode::Blocked)
        || matches!(supervision.status, SupervisionStatus::Quarantined)
    {
        EpisodeRecommendation::Stop
    } else if runtime_gate.requires_recovery() {
        EpisodeRecommendation::Recover
    } else if runtime_gate.permits_learning() {
        EpisodeRecommendation::ApplyLearning
    } else if runtime_gate.permits_exploration() {
        EpisodeRecommendation::Probe
    } else {
        EpisodeRecommendation::Observe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::readiness::{ReadinessReport, evaluate_readiness};
    use crate::cognition::scenario::run_default_scenarios;
    use crate::core::math::FixedPoint;
    use crate::training::trainer::TrainStepStatus;

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

    #[test]
    fn healthy_episode_cycle_recommends_learning() {
        let mut runner = CognitiveEpisodeRunner::conservative(mature_readiness(), 16);

        let episode = runner.preview_cycle(
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

        assert_eq!(episode.recommendation, EpisodeRecommendation::ApplyLearning);
        assert!(episode.allows_model_update());
        assert!(!episode.allows_probe());
        assert_eq!(episode.supervision.status, SupervisionStatus::Stable);
        assert_eq!(runner.supervision().learning_allowed, 1);
    }

    #[test]
    fn exploration_episode_recommends_probe_without_model_update() {
        let mut runner = CognitiveEpisodeRunner::conservative(mature_readiness(), 16);

        let episode = runner.preview_cycle(
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
                novelty: FixedPoint::from_f32(0.35),
                prediction_error: FixedPoint::from_f32(0.2),
                safety_pressure: FixedPoint::ZERO,
            },
        );

        assert_eq!(episode.recommendation, EpisodeRecommendation::Probe);
        assert!(!episode.allows_model_update());
        assert!(episode.allows_probe());
        assert_eq!(episode.supervision.exploration_allowed, 1);
    }

    #[test]
    fn blocked_readiness_stops_unattended_episode() {
        let mut runner = CognitiveEpisodeRunner::conservative(ReadinessReport::blocked(), 16);

        let episode = runner.preview_cycle(
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

        assert_eq!(episode.recommendation, EpisodeRecommendation::Stop);
        assert!(episode.should_stop_unattended());
        assert!(!episode.allows_model_update());
        assert_eq!(episode.runtime_gate.mode, RuntimeGateMode::Blocked);
    }

    #[test]
    fn repeated_recovery_stops_after_quarantine() {
        let mut runner = CognitiveEpisodeRunner::conservative(mature_readiness(), 16);
        let saturated = report(
            4,
            0,
            0,
            1,
            TrainStepStatus::Complete,
            FixedPoint::from_f32(8192.0),
        );

        let first = runner.preview_cycle(saturated, CycleSignals::neutral());
        let second = runner.preview_cycle(saturated, CycleSignals::neutral());

        assert_eq!(first.recommendation, EpisodeRecommendation::Recover);
        assert_eq!(first.supervision.status, SupervisionStatus::Recovering);
        assert_eq!(second.recommendation, EpisodeRecommendation::Stop);
        assert!(second.should_stop_unattended());
        assert!(!second.allows_model_update());
        assert_eq!(second.supervision.status, SupervisionStatus::Quarantined);
    }

    #[test]
    fn budget_starved_episode_observes_only() {
        let mut runner = CognitiveEpisodeRunner::conservative(mature_readiness(), 2);

        let episode = runner.preview_cycle(
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

        assert_eq!(episode.recommendation, EpisodeRecommendation::Observe);
        assert!(!episode.allows_model_update());
        assert!(!episode.allows_probe());
        assert_eq!(episode.supervision.observed, 1);
    }
}
