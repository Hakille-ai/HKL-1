//! Bounded action planner for HKL-2 executive decisions.
//!
//! The planner converts an executive decision into a small auditable sequence of
//! steps. It does not execute effects; it only records what a higher-level loop
//! may attempt next, within a deterministic step and time budget.

use crate::cognition::executive::{ExecutiveAction, ExecutiveDecision, ExecutiveReason};
use crate::core::math::FixedPoint;

pub const MAX_PLAN_STEPS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanStepKind {
    SafetyCheck,
    PreviewTraining,
    ApplyLearning,
    ExploreProbe,
    ConsolidateMemory,
    RecoverState,
    EmitTelemetry,
    Idle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub kind: PlanStepKind,
    pub priority: FixedPoint,
    pub budget_ticks: u16,
    pub requires_checkpoint: bool,
}

impl PlanStep {
    pub const fn idle() -> Self {
        Self {
            kind: PlanStepKind::Idle,
            priority: FixedPoint::ZERO,
            budget_ticks: 0,
            requires_checkpoint: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CognitivePlan {
    pub steps: [PlanStep; MAX_PLAN_STEPS],
    pub len: usize,
    pub source_action: ExecutiveAction,
    pub source_reason: ExecutiveReason,
    pub total_budget_ticks: u16,
    pub rollback_required: bool,
    pub trace_id: u32,
}

impl CognitivePlan {
    pub const fn empty() -> Self {
        Self {
            steps: [PlanStep::idle(); MAX_PLAN_STEPS],
            len: 0,
            source_action: ExecutiveAction::Idle,
            source_reason: ExecutiveReason::NoWork,
            total_budget_ticks: 0,
            rollback_required: false,
            trace_id: 0,
        }
    }

    pub fn first_step(&self) -> PlanStep {
        if self.len == 0 {
            PlanStep::idle()
        } else {
            self.steps[0]
        }
    }
}

pub struct CognitivePlanner {
    pub max_budget_ticks: u16,
    pub last_plan: CognitivePlan,
}

impl CognitivePlanner {
    pub fn new(max_budget_ticks: u16) -> Self {
        Self {
            max_budget_ticks,
            last_plan: CognitivePlan::empty(),
        }
    }

    pub fn plan(&mut self, decision: &ExecutiveDecision) -> CognitivePlan {
        let mut plan = CognitivePlan {
            source_action: decision.action,
            source_reason: decision.reason,
            trace_id: trace_id(decision),
            ..CognitivePlan::empty()
        };

        match decision.action {
            ExecutiveAction::Idle => {
                plan.push(step(
                    PlanStepKind::Idle,
                    FixedPoint::from_f32(0.1),
                    1,
                    false,
                ));
                plan.push(step(
                    PlanStepKind::EmitTelemetry,
                    FixedPoint::from_f32(0.1),
                    1,
                    false,
                ));
            }
            ExecutiveAction::Learn => {
                plan.push(safety_step(decision.priority));
                plan.push(step(
                    PlanStepKind::PreviewTraining,
                    decision.priority,
                    3,
                    false,
                ));
                plan.push(step(
                    PlanStepKind::ApplyLearning,
                    decision.learning_scale,
                    scaled_ticks(4, decision.learning_scale),
                    true,
                ));
                plan.push(telemetry_step(decision.priority));
            }
            ExecutiveAction::Explore => {
                plan.push(safety_step(decision.priority));
                plan.push(step(
                    PlanStepKind::ExploreProbe,
                    decision.exploration_scale,
                    scaled_ticks(5, decision.exploration_scale),
                    true,
                ));
                plan.push(step(
                    PlanStepKind::PreviewTraining,
                    decision.priority,
                    3,
                    false,
                ));
                plan.push(step(
                    PlanStepKind::ApplyLearning,
                    decision.learning_scale,
                    scaled_ticks(3, decision.learning_scale),
                    true,
                ));
                plan.push(telemetry_step(decision.priority));
            }
            ExecutiveAction::Consolidate => {
                plan.push(safety_step(decision.priority));
                plan.push(step(
                    PlanStepKind::ConsolidateMemory,
                    decision.priority,
                    5,
                    true,
                ));
                plan.push(telemetry_step(decision.priority));
            }
            ExecutiveAction::Recover => {
                plan.rollback_required = true;
                plan.push(safety_step(FixedPoint::ONE));
                plan.push(step(PlanStepKind::RecoverState, decision.priority, 5, true));
                plan.push(telemetry_step(decision.priority));
            }
        }

        plan.clamp_budget(self.max_budget_ticks);
        self.last_plan = plan;
        plan
    }
}

impl CognitivePlan {
    fn push(&mut self, next: PlanStep) {
        if self.len >= MAX_PLAN_STEPS {
            return;
        }
        self.total_budget_ticks = self.total_budget_ticks.saturating_add(next.budget_ticks);
        self.rollback_required |=
            next.requires_checkpoint && next.kind == PlanStepKind::RecoverState;
        self.steps[self.len] = next;
        self.len += 1;
    }

    fn clamp_budget(&mut self, max_budget_ticks: u16) {
        if self.total_budget_ticks <= max_budget_ticks {
            return;
        }

        let mut remaining = max_budget_ticks;
        let mut compacted = [PlanStep::idle(); MAX_PLAN_STEPS];
        let mut compacted_len = 0;
        let mut total_budget_ticks = 0_u16;
        let mut rollback_required = self.rollback_required;

        for i in 0..self.len {
            let allowed = self.steps[i].budget_ticks.min(remaining);
            if allowed > 0 {
                let mut bounded = self.steps[i];
                bounded.budget_ticks = allowed;
                total_budget_ticks = total_budget_ticks.saturating_add(allowed);
                rollback_required |=
                    bounded.requires_checkpoint && bounded.kind == PlanStepKind::RecoverState;
                compacted[compacted_len] = bounded;
                compacted_len += 1;
            }
            remaining = remaining.saturating_sub(allowed);
        }

        self.steps = compacted;
        self.len = compacted_len;
        self.total_budget_ticks = total_budget_ticks;
        self.rollback_required = rollback_required;
    }
}

fn safety_step(priority: FixedPoint) -> PlanStep {
    step(
        PlanStepKind::SafetyCheck,
        priority.max(FixedPoint::HALF),
        2,
        false,
    )
}

fn telemetry_step(priority: FixedPoint) -> PlanStep {
    step(PlanStepKind::EmitTelemetry, priority, 1, false)
}

fn step(
    kind: PlanStepKind,
    priority: FixedPoint,
    budget_ticks: u16,
    requires_checkpoint: bool,
) -> PlanStep {
    PlanStep {
        kind,
        priority: clamp01(priority),
        budget_ticks,
        requires_checkpoint,
    }
}

fn scaled_ticks(base: u16, scale: FixedPoint) -> u16 {
    let clamped = clamp01(scale);
    let variable = ((base as i32 * clamped.to_bits()) >> FixedPoint::FRAC_BITS) as u16;
    base.saturating_add(variable).max(1)
}

fn clamp01(value: FixedPoint) -> FixedPoint {
    value.clamp(FixedPoint::ZERO, FixedPoint::ONE)
}

fn trace_id(decision: &ExecutiveDecision) -> u32 {
    let mut h = 0x811c9dc5_u32;
    h = mix(h, action_id(decision.action));
    h = mix(h, reason_id(decision.reason));
    h = mix(h, decision.priority.to_bits() as u32);
    h = mix(h, decision.learning_scale.to_bits() as u32);
    h = mix(h, decision.exploration_scale.to_bits() as u32);
    for signal in decision.trace.iter().take(decision.trace_len) {
        h = mix(h, signal.intensity.to_bits() as u32);
    }
    h
}

fn mix(hash: u32, value: u32) -> u32 {
    hash ^ value.wrapping_mul(0x01000193)
}

fn action_id(action: ExecutiveAction) -> u32 {
    match action {
        ExecutiveAction::Learn => 1,
        ExecutiveAction::Explore => 2,
        ExecutiveAction::Consolidate => 3,
        ExecutiveAction::Recover => 4,
        ExecutiveAction::Idle => 5,
    }
}

fn reason_id(reason: ExecutiveReason) -> u32 {
    match reason {
        ExecutiveReason::HealthyLearning => 11,
        ExecutiveReason::CuriosityNovelty => 12,
        ExecutiveReason::TrainingThrottled => 13,
        ExecutiveReason::TrainingHalted => 14,
        ExecutiveReason::SafetyPressure => 15,
        ExecutiveReason::NoWork => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognition::executive::{CausalSignal, CausalSignalKind, MAX_CAUSAL_SIGNALS};

    fn decision(
        action: ExecutiveAction,
        reason: ExecutiveReason,
        priority: FixedPoint,
        learning_scale: FixedPoint,
        exploration_scale: FixedPoint,
    ) -> ExecutiveDecision {
        ExecutiveDecision {
            action,
            reason,
            priority,
            learning_scale,
            exploration_scale,
            trace: [CausalSignal {
                kind: CausalSignalKind::Guard,
                intensity: FixedPoint::from_f32(0.2),
            }; MAX_CAUSAL_SIGNALS],
            trace_len: MAX_CAUSAL_SIGNALS,
        }
    }

    #[test]
    fn learn_plan_previews_then_applies_learning() {
        let mut planner = CognitivePlanner::new(16);
        let plan = planner.plan(&decision(
            ExecutiveAction::Learn,
            ExecutiveReason::HealthyLearning,
            FixedPoint::from_f32(0.5),
            FixedPoint::ONE,
            FixedPoint::ZERO,
        ));

        assert_eq!(plan.len, 4);
        assert_eq!(plan.steps[0].kind, PlanStepKind::SafetyCheck);
        assert_eq!(plan.steps[1].kind, PlanStepKind::PreviewTraining);
        assert_eq!(plan.steps[2].kind, PlanStepKind::ApplyLearning);
        assert!(plan.steps[2].requires_checkpoint);
        assert!(!plan.rollback_required);
    }

    #[test]
    fn explore_plan_contains_probe_before_learning() {
        let mut planner = CognitivePlanner::new(32);
        let plan = planner.plan(&decision(
            ExecutiveAction::Explore,
            ExecutiveReason::CuriosityNovelty,
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.5),
            FixedPoint::from_f32(0.9),
        ));

        assert_eq!(plan.steps[1].kind, PlanStepKind::ExploreProbe);
        assert_eq!(plan.steps[2].kind, PlanStepKind::PreviewTraining);
        assert_eq!(plan.steps[3].kind, PlanStepKind::ApplyLearning);
        assert!(plan.trace_id != 0);
    }

    #[test]
    fn recover_plan_requires_rollback_path() {
        let mut planner = CognitivePlanner::new(16);
        let plan = planner.plan(&decision(
            ExecutiveAction::Recover,
            ExecutiveReason::TrainingHalted,
            FixedPoint::from_f32(0.95),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
        ));

        assert!(plan.rollback_required);
        assert_eq!(plan.steps[0].kind, PlanStepKind::SafetyCheck);
        assert_eq!(plan.steps[1].kind, PlanStepKind::RecoverState);
        assert_eq!(plan.steps[2].kind, PlanStepKind::EmitTelemetry);
    }

    #[test]
    fn idle_plan_is_short_and_non_mutating() {
        let mut planner = CognitivePlanner::new(16);
        let plan = planner.plan(&decision(
            ExecutiveAction::Idle,
            ExecutiveReason::NoWork,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
        ));

        assert_eq!(plan.len, 2);
        assert_eq!(plan.first_step().kind, PlanStepKind::Idle);
        assert!(!plan.steps[0].requires_checkpoint);
        assert!(!plan.rollback_required);
    }

    #[test]
    fn planner_clamps_total_budget() {
        let mut planner = CognitivePlanner::new(4);
        let plan = planner.plan(&decision(
            ExecutiveAction::Explore,
            ExecutiveReason::CuriosityNovelty,
            FixedPoint::ONE,
            FixedPoint::ONE,
            FixedPoint::ONE,
        ));

        assert_eq!(plan.total_budget_ticks, 4);
        let mut sum = 0_u16;
        for step in plan.steps.iter().take(plan.len) {
            sum = sum.saturating_add(step.budget_ticks);
        }
        assert_eq!(sum, 4);
    }

    #[test]
    fn planner_removes_zero_budget_steps_after_clamp() {
        let mut planner = CognitivePlanner::new(2);
        let plan = planner.plan(&decision(
            ExecutiveAction::Explore,
            ExecutiveReason::CuriosityNovelty,
            FixedPoint::ONE,
            FixedPoint::ONE,
            FixedPoint::ONE,
        ));

        assert_eq!(plan.total_budget_ticks, 2);
        assert_eq!(plan.len, 1);
        assert_eq!(plan.steps[0].kind, PlanStepKind::SafetyCheck);
        assert!(
            !plan
                .steps
                .iter()
                .take(plan.len)
                .any(|step| step.kind == PlanStepKind::ApplyLearning)
        );
    }

    #[test]
    fn planner_preserves_recovery_requirement_after_budget_clamp() {
        let mut planner = CognitivePlanner::new(1);
        let plan = planner.plan(&decision(
            ExecutiveAction::Recover,
            ExecutiveReason::SafetyPressure,
            FixedPoint::ONE,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
        ));

        assert_eq!(plan.total_budget_ticks, 1);
        assert_eq!(plan.len, 1);
        assert_eq!(plan.steps[0].kind, PlanStepKind::SafetyCheck);
        assert!(plan.rollback_required);
    }

    #[test]
    fn priorities_are_clamped() {
        let mut planner = CognitivePlanner::new(16);
        let plan = planner.plan(&decision(
            ExecutiveAction::Learn,
            ExecutiveReason::HealthyLearning,
            FixedPoint::from_f32(5.0),
            FixedPoint::from_f32(2.0),
            FixedPoint::ZERO,
        ));

        for step in plan.steps.iter().take(plan.len) {
            assert!(step.priority >= FixedPoint::ZERO);
            assert!(step.priority <= FixedPoint::ONE);
        }
    }
}
