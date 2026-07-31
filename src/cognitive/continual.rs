//! Continual Learning & Anti-Catastrophic Forgetting Module for HKL-1.
//! Implements Offline Experience Replay (Sharp Wave-Ripples SWR), Few-Shot Fast-Weights Adaptation,
//! Meta-Learning hyperparameter auto-tuning, and Elastic Weight Consolidation (EWC) Fisher Information Protection.

use crate::cognitive::episodic::ReplayExperience;
use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;

pub const MAX_PROTECTED_SYNAPSES: usize = 256;

/// Protected Synapse under Elastic Weight Consolidation (EWC)
#[derive(Clone, Copy, Debug)]
pub struct EwcProtectedSynapse {
    pub source_id: NeuronId,
    pub target_id: NeuronId,
    pub optimal_weight: FixedPoint,
    pub fisher_importance: FixedPoint,
}

/// Elastic Weight Consolidation (EWC) Anti-Catastrophic Forgetting Engine
pub struct ElasticWeightConsolidation {
    pub protected_synapses: [Option<EwcProtectedSynapse>; MAX_PROTECTED_SYNAPSES],
    pub count: usize,
    pub ewc_lambda: FixedPoint, // EWC regularization strength
}

impl ElasticWeightConsolidation {
    pub fn new() -> Self {
        Self {
            protected_synapses: [None; MAX_PROTECTED_SYNAPSES],
            count: 0,
            ewc_lambda: FixedPoint::from_f32(10.0),
        }
    }

    /// Register a critical synapse with its Fisher Information Importance score F_ij
    pub fn protect_synapse(
        &mut self,
        source_id: NeuronId,
        target_id: NeuronId,
        optimal_weight: FixedPoint,
        fisher_importance: FixedPoint,
    ) {
        if self.count < MAX_PROTECTED_SYNAPSES {
            self.protected_synapses[self.count] = Some(EwcProtectedSynapse {
                source_id,
                target_id,
                optimal_weight,
                fisher_importance,
            });
            self.count += 1;
        }
    }

    /// Calculate EWC penalty gradient adjustment to prevent catastrophic forgetting
    pub fn compute_ewc_penalty_delta(
        &self,
        source_id: NeuronId,
        target_id: NeuronId,
        current_weight: FixedPoint,
    ) -> FixedPoint {
        for i in 0..self.count {
            if let Some(p) = self.protected_synapses[i] {
                if p.source_id == source_id && p.target_id == target_id {
                    let weight_diff = current_weight - p.optimal_weight;
                    // EWC Penalty = lambda * F_ij * (w - w_opt)
                    return self.ewc_lambda * p.fisher_importance * weight_diff;
                }
            }
        }
        FixedPoint::ZERO
    }
}

/// Offline Replay Engine (Sharp Wave-Ripples SWR 150-250 Hz)
pub struct OfflineReplayEngine {
    pub replay_count: u32,
    pub active_swr: bool,
}

impl OfflineReplayEngine {
    pub fn new() -> Self {
        Self {
            replay_count: 0,
            active_swr: false,
        }
    }

    /// Execute SWR replay phase on episodic memory traces during rest/sleep
    pub fn execute_swr_replay(&mut self, trace: &ReplayExperience) -> (FixedPoint, u32) {
        self.active_swr = true;
        self.replay_count += 1;
        // Replay returns reinforced reward signal and high-frequency SWR pulse tick
        (trace.reward * FixedPoint::from_f32(1.5), 200) // 200 Hz SWR frequency
    }
}

/// Few-Shot Learning Adapter (Fast-Weights Plasticity Booster)
pub struct FewShotAdapter {
    pub boost_factor: FixedPoint,
    pub shots_count: u32,
}

impl FewShotAdapter {
    pub fn new() -> Self {
        Self {
            boost_factor: FixedPoint::from_f32(4.0), // 4x STDP boost for 1-shot learning
            shots_count: 0,
        }
    }

    /// Compute boosted learning rate for rapid few-shot adaptation
    pub fn get_boosted_lr(
        &mut self,
        base_lr: FixedPoint,
        prediction_error: FixedPoint,
    ) -> FixedPoint {
        if prediction_error > FixedPoint::from_f32(0.5) {
            self.shots_count += 1;
            base_lr * self.boost_factor
        } else {
            base_lr
        }
    }
}

/// Meta-Learning Auto-Tuner for Hyperparameters
pub struct MetaLearningEngine {
    pub current_eta_stdp: FixedPoint,
    pub current_theta_da: FixedPoint,
    pub adaptation_rate: FixedPoint,
}

impl MetaLearningEngine {
    pub fn new() -> Self {
        Self {
            current_eta_stdp: FixedPoint::from_f32(0.01),
            current_theta_da: FixedPoint::from_f32(0.20),
            adaptation_rate: FixedPoint::from_f32(0.005),
        }
    }

    /// Auto-tune STDP learning rate and Dopamine threshold based on global performance error
    pub fn adapt_hyperparameters(&mut self, global_error: FixedPoint) {
        if global_error > FixedPoint::from_f32(0.3) {
            // Increase plasticity for exploration
            self.current_eta_stdp += self.adaptation_rate;
            self.current_theta_da -= self.adaptation_rate;
        } else {
            // Consolidate for exploitation
            self.current_eta_stdp =
                (self.current_eta_stdp - self.adaptation_rate).max(FixedPoint::from_f32(0.001));
            self.current_theta_da =
                (self.current_theta_da + self.adaptation_rate).min(FixedPoint::from_f32(0.80));
        }
    }
}

/// Master Continual Learning Engine
pub struct ContinualLearningEngine {
    pub offline_replay: OfflineReplayEngine,
    pub few_shot: FewShotAdapter,
    pub meta_learning: MetaLearningEngine,
    pub ewc: ElasticWeightConsolidation,
}

impl ContinualLearningEngine {
    pub fn new() -> Self {
        Self {
            offline_replay: OfflineReplayEngine::new(),
            few_shot: FewShotAdapter::new(),
            meta_learning: MetaLearningEngine::new(),
            ewc: ElasticWeightConsolidation::new(),
        }
    }
}
