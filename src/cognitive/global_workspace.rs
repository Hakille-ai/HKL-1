//! Global workspace ignition for cross-module cognitive arbitration.
//!
//! This module implements a small deterministic Global Workspace Theory layer:
//! specialist modules submit competing candidates, the workspace scores them
//! from salience, confidence, novelty, reward, risk, and metabolic cost, then
//! broadcasts the winning conscious frame to attention, action, and safety
//! systems.

use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

pub const MAX_WORKSPACE_CANDIDATES: usize = 16;
pub const WORKSPACE_BROADCAST_HISTORY: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecialistId {
    Sensory,
    Actor,
    Predictor,
    Memory,
    Safety,
    Swarm,
    Language,
    Vision,
    Audio,
    Power,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkspaceMode {
    Quiescent,
    Exploring,
    Focused,
    Guarded,
    Crisis,
}

#[derive(Clone, Copy, Debug)]
pub struct WorkspaceCandidate {
    pub specialist: SpecialistId,
    pub content_id: u32,
    pub target_layer: u8,
    pub action_hint: Option<u8>,
    pub salience: FixedPoint,
    pub confidence: FixedPoint,
    pub novelty: FixedPoint,
    pub expected_reward: FixedPoint,
    pub risk: FixedPoint,
    pub energy_cost: FixedPoint,
    pub timestamp_ms: u32,
    pub valid: bool,
}

impl WorkspaceCandidate {
    pub const fn empty() -> Self {
        Self {
            specialist: SpecialistId::Unknown,
            content_id: 0,
            target_layer: 0,
            action_hint: None,
            salience: FixedPoint::ZERO,
            confidence: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            expected_reward: FixedPoint::ZERO,
            risk: FixedPoint::ZERO,
            energy_cost: FixedPoint::ZERO,
            timestamp_ms: 0,
            valid: false,
        }
    }

    pub fn new(
        specialist: SpecialistId,
        content_id: u32,
        target_layer: u8,
        action_hint: Option<u8>,
    ) -> Self {
        Self {
            specialist,
            content_id,
            target_layer,
            action_hint,
            valid: true,
            ..Self::empty()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BroadcastFrame {
    pub specialist: SpecialistId,
    pub content_id: u32,
    pub target_layer: u8,
    pub action_hint: Option<u8>,
    pub score: FixedPoint,
    pub ignition_strength: FixedPoint,
    pub mode: WorkspaceMode,
    pub timestamp_ms: u32,
    pub valid: bool,
}

impl BroadcastFrame {
    pub const fn empty() -> Self {
        Self {
            specialist: SpecialistId::Unknown,
            content_id: 0,
            target_layer: 0,
            action_hint: None,
            score: FixedPoint::ZERO,
            ignition_strength: FixedPoint::ZERO,
            mode: WorkspaceMode::Quiescent,
            timestamp_ms: 0,
            valid: false,
        }
    }
}

pub struct GlobalWorkspace {
    pub candidates: [WorkspaceCandidate; MAX_WORKSPACE_CANDIDATES],
    pub candidate_count: usize,
    pub last_broadcast: BroadcastFrame,
    pub broadcast_history: [BroadcastFrame; WORKSPACE_BROADCAST_HISTORY],
    pub history_idx: usize,
    pub ignition_threshold: FixedPoint,
    pub stability: FixedPoint,
    pub integration_pressure: FixedPoint,
    pub safety_bias: FixedPoint,
    pub broadcast_count: u32,
    pub rejected_count: u32,
}

impl GlobalWorkspace {
    pub const fn new() -> Self {
        Self {
            candidates: [WorkspaceCandidate::empty(); MAX_WORKSPACE_CANDIDATES],
            candidate_count: 0,
            last_broadcast: BroadcastFrame::empty(),
            broadcast_history: [BroadcastFrame::empty(); WORKSPACE_BROADCAST_HISTORY],
            history_idx: 0,
            ignition_threshold: FixedPoint::from_f32(0.32),
            stability: FixedPoint::from_f32(0.5),
            integration_pressure: FixedPoint::ZERO,
            safety_bias: FixedPoint::from_f32(0.25),
            broadcast_count: 0,
            rejected_count: 0,
        }
    }

    pub fn reset_cycle(&mut self) {
        self.candidate_count = 0;
        for slot in self.candidates.iter_mut() {
            *slot = WorkspaceCandidate::empty();
        }
    }

    pub fn submit(&mut self, mut candidate: WorkspaceCandidate) -> bool {
        if self.candidate_count >= MAX_WORKSPACE_CANDIDATES {
            self.rejected_count = self.rejected_count.saturating_add(1);
            return false;
        }
        candidate.salience = candidate.salience.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        candidate.confidence = candidate
            .confidence
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        candidate.novelty = candidate.novelty.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        candidate.expected_reward = candidate
            .expected_reward
            .clamp(FixedPoint::from_f32(-1.0), FixedPoint::ONE);
        candidate.risk = candidate.risk.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        candidate.energy_cost = candidate
            .energy_cost
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        candidate.valid = true;

        self.candidates[self.candidate_count] = candidate;
        self.candidate_count += 1;
        true
    }

    pub fn score(&self, candidate: &WorkspaceCandidate) -> FixedPoint {
        if !candidate.valid {
            return FixedPoint::from_f32(-4.0);
        }

        let mut score = candidate.salience * FixedPoint::from_f32(0.30)
            + candidate.confidence * FixedPoint::from_f32(0.20)
            + candidate.novelty * FixedPoint::from_f32(0.18)
            + candidate.expected_reward * FixedPoint::from_f32(0.17);

        score = score - candidate.energy_cost * FixedPoint::from_f32(0.10);

        if candidate.specialist == SpecialistId::Safety {
            score += self.safety_bias + candidate.risk * FixedPoint::from_f32(0.30);
        } else {
            score -= candidate.risk * FixedPoint::from_f32(0.28);
        }

        score + self.integration_pressure * FixedPoint::from_f32(0.05)
    }

    pub fn ignite(&mut self, now_ms: u32) -> Option<BroadcastFrame> {
        let mut best_idx: Option<usize> = None;
        let mut best_score = FixedPoint::from_f32(-4.0);

        for (idx, candidate) in self.candidates[..self.candidate_count].iter().enumerate() {
            let score = self.score(candidate);
            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        let Some(idx) = best_idx else {
            self.last_broadcast = BroadcastFrame::empty();
            return None;
        };

        if best_score < self.ignition_threshold {
            self.rejected_count = self.rejected_count.saturating_add(1);
            self.last_broadcast = BroadcastFrame::empty();
            return None;
        }

        let winner = self.candidates[idx];
        let ignition_strength =
            (best_score - self.ignition_threshold).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        let mode = self.derive_mode(&winner, best_score);
        let frame = BroadcastFrame {
            specialist: winner.specialist,
            content_id: winner.content_id,
            target_layer: winner.target_layer,
            action_hint: winner.action_hint,
            score: best_score,
            ignition_strength,
            mode,
            timestamp_ms: now_ms,
            valid: true,
        };

        self.last_broadcast = frame;
        self.broadcast_history[self.history_idx] = frame;
        self.history_idx = (self.history_idx + 1) % WORKSPACE_BROADCAST_HISTORY;
        self.broadcast_count = self.broadcast_count.saturating_add(1);
        self.integration_pressure = self.integration_pressure * FixedPoint::from_f32(0.85)
            + ignition_strength * FixedPoint::from_f32(0.15);
        self.stability = match mode {
            WorkspaceMode::Focused => (self.stability + FixedPoint::from_f32(0.03))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE),
            WorkspaceMode::Crisis | WorkspaceMode::Guarded => (self.stability
                - FixedPoint::from_f32(0.05))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE),
            _ => self.stability,
        };
        Some(frame)
    }

    pub fn submit_network_state(
        &mut self,
        now_ms: u32,
        prediction_error: FixedPoint,
        novelty: FixedPoint,
        action: Option<u8>,
        action_confidence: FixedPoint,
        energy_level: FixedPoint,
    ) -> Option<BroadcastFrame> {
        self.reset_cycle();

        let mut predictor = WorkspaceCandidate::new(SpecialistId::Predictor, now_ms, 2, None);
        predictor.salience = prediction_error;
        predictor.confidence =
            (FixedPoint::ONE - prediction_error).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        predictor.novelty = novelty;
        predictor.expected_reward = (FixedPoint::from_f32(0.4) - prediction_error)
            .clamp(FixedPoint::from_f32(-1.0), FixedPoint::ONE);
        predictor.energy_cost = FixedPoint::from_f32(0.08);
        predictor.timestamp_ms = now_ms;
        self.submit(predictor);

        if let Some(action_hint) = action {
            let mut actor =
                WorkspaceCandidate::new(SpecialistId::Actor, now_ms ^ 0xA17C, 4, Some(action_hint));
            actor.salience = action_confidence;
            actor.confidence = action_confidence;
            actor.novelty = novelty * FixedPoint::from_f32(0.35);
            actor.expected_reward = action_confidence * FixedPoint::from_f32(0.5);
            actor.energy_cost = FixedPoint::from_f32(0.12);
            actor.timestamp_ms = now_ms;
            self.submit(actor);
        }

        let risk = (prediction_error + (FixedPoint::ONE - energy_level))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        if risk > FixedPoint::from_f32(0.35) {
            let mut safety =
                WorkspaceCandidate::new(SpecialistId::Safety, now_ms ^ 0x5AFE, 6, None);
            safety.salience = risk;
            safety.confidence = FixedPoint::from_f32(0.85);
            safety.risk = risk;
            safety.expected_reward = FixedPoint::from_f32(0.2);
            safety.energy_cost = FixedPoint::from_f32(0.03);
            safety.timestamp_ms = now_ms;
            self.submit(safety);
        }

        if novelty > FixedPoint::from_f32(0.25) {
            let mut explorer =
                WorkspaceCandidate::new(SpecialistId::Sensory, now_ms ^ 0xE970, 1, None);
            explorer.salience = novelty;
            explorer.confidence = FixedPoint::from_f32(0.55);
            explorer.novelty = novelty;
            explorer.expected_reward = novelty * FixedPoint::from_f32(0.3);
            explorer.energy_cost = FixedPoint::from_f32(0.05);
            explorer.timestamp_ms = now_ms;
            self.submit(explorer);
        }

        self.ignite(now_ms)
    }

    fn derive_mode(&self, winner: &WorkspaceCandidate, score: FixedPoint) -> WorkspaceMode {
        if winner.specialist == SpecialistId::Safety && winner.risk > FixedPoint::from_f32(0.75) {
            return WorkspaceMode::Crisis;
        }
        if winner.specialist == SpecialistId::Safety || winner.risk > FixedPoint::from_f32(0.45) {
            return WorkspaceMode::Guarded;
        }
        if winner.novelty > FixedPoint::from_f32(0.55) {
            return WorkspaceMode::Exploring;
        }
        if score > FixedPoint::from_f32(0.55) {
            return WorkspaceMode::Focused;
        }
        WorkspaceMode::Quiescent
    }
}

pub static mut GLOBAL_WORKSPACE: MaybeUninit<GlobalWorkspace> = MaybeUninit::uninit();
static INITIALIZED_GLOBAL_WORKSPACE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_global_workspace() {
    unsafe {
        if !INITIALIZED_GLOBAL_WORKSPACE.load(core::sync::atomic::Ordering::Relaxed) {
            GLOBAL_WORKSPACE.write(GlobalWorkspace::new());
            INITIALIZED_GLOBAL_WORKSPACE.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn global_workspace() -> &'static mut GlobalWorkspace {
    unsafe {
        if !INITIALIZED_GLOBAL_WORKSPACE.load(core::sync::atomic::Ordering::Relaxed) {
            init_global_workspace();
        }
        &mut *GLOBAL_WORKSPACE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_selects_highest_scoring_candidate() {
        let mut gw = GlobalWorkspace::new();
        let mut weak = WorkspaceCandidate::new(SpecialistId::Actor, 1, 4, Some(7));
        weak.salience = FixedPoint::from_f32(0.2);
        weak.confidence = FixedPoint::from_f32(0.2);
        let mut strong = WorkspaceCandidate::new(SpecialistId::Predictor, 2, 2, None);
        strong.salience = FixedPoint::from_f32(0.9);
        strong.confidence = FixedPoint::from_f32(0.8);
        strong.novelty = FixedPoint::from_f32(0.4);
        assert!(gw.submit(weak));
        assert!(gw.submit(strong));

        let frame = gw.ignite(42).expect("strong candidate should ignite");
        assert_eq!(frame.specialist, SpecialistId::Predictor);
        assert_eq!(frame.content_id, 2);
        assert!(frame.valid);
    }

    #[test]
    fn safety_candidate_gets_priority_under_risk() {
        let mut gw = GlobalWorkspace::new();
        let mut actor = WorkspaceCandidate::new(SpecialistId::Actor, 10, 4, Some(1));
        actor.salience = FixedPoint::from_f32(0.8);
        actor.confidence = FixedPoint::from_f32(0.9);
        actor.risk = FixedPoint::from_f32(0.7);
        let mut safety = WorkspaceCandidate::new(SpecialistId::Safety, 11, 6, None);
        safety.salience = FixedPoint::from_f32(0.65);
        safety.confidence = FixedPoint::from_f32(0.8);
        safety.risk = FixedPoint::from_f32(0.8);
        assert!(gw.submit(actor));
        assert!(gw.submit(safety));

        let frame = gw.ignite(100).expect("safety frame should ignite");
        assert_eq!(frame.specialist, SpecialistId::Safety);
        assert_eq!(frame.mode, WorkspaceMode::Crisis);
    }

    #[test]
    fn submit_network_state_generates_broadcast() {
        let mut gw = GlobalWorkspace::new();
        let frame = gw.submit_network_state(
            9,
            FixedPoint::from_f32(0.42),
            FixedPoint::from_f32(0.5),
            Some(33),
            FixedPoint::from_f32(0.7),
            FixedPoint::from_f32(0.8),
        );
        assert!(frame.is_some());
        assert_eq!(gw.broadcast_count, 1);
        assert!(gw.last_broadcast.valid);
    }

    #[test]
    fn rejects_when_capacity_is_full() {
        let mut gw = GlobalWorkspace::new();
        for idx in 0..MAX_WORKSPACE_CANDIDATES {
            assert!(gw.submit(WorkspaceCandidate::new(
                SpecialistId::Unknown,
                idx as u32,
                0,
                None,
            )));
        }
        assert!(!gw.submit(WorkspaceCandidate::new(SpecialistId::Unknown, 99, 0, None)));
        assert_eq!(gw.rejected_count, 1);
    }
}
