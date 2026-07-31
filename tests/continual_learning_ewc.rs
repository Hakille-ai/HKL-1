#![cfg(feature = "std")]

use hkl1::cognitive::continual::{
    ElasticWeightConsolidation, FewShotAdapter, MetaLearningEngine, OfflineReplayEngine,
};
use hkl1::cognitive::continual_learning;
use hkl1::cognitive::episodic::ReplayExperience;
use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;

#[test]
fn test_offline_experience_replay_swr() {
    let mut replay_engine = OfflineReplayEngine::new();

    let sample_trace = ReplayExperience {
        state_hash: 0x12345678,
        action: 2,
        next_state_hash: 0x87654321,
        reward: FixedPoint::from_f32(0.8),
        td_error: FixedPoint::from_f32(0.1),
    };

    let (reinforced_reward, swr_freq_hz) = replay_engine.execute_swr_replay(&sample_trace);

    assert!(replay_engine.active_swr);
    assert_eq!(replay_engine.replay_count, 1);
    assert!(reinforced_reward > sample_trace.reward, "Replay must reinforce reward!");
    assert_eq!(swr_freq_hz, 200, "SWR frequency must be ~200 Hz");
}

#[test]
fn test_few_shot_fast_weights_adaptation() {
    let mut adapter = FewShotAdapter::new();
    let base_lr = FixedPoint::from_f32(0.01);

    // Low error -> Normal learning rate
    let lr_normal = adapter.get_boosted_lr(base_lr, FixedPoint::from_f32(0.1));
    assert_eq!(lr_normal, base_lr);

    // High prediction error (>0.5) -> 4x Boosted Fast-Weight learning rate
    let lr_boosted = adapter.get_boosted_lr(base_lr, FixedPoint::from_f32(0.8));
    assert!(lr_boosted > base_lr);
    assert_eq!(adapter.shots_count, 1);
}

#[test]
fn test_meta_learning_hyperparameter_autotuning() {
    let mut meta = MetaLearningEngine::new();

    let initial_eta = meta.current_eta_stdp;
    let initial_theta = meta.current_theta_da;

    // High global error -> Increases STDP learning rate for exploration
    meta.adapt_hyperparameters(FixedPoint::from_f32(0.6));
    assert!(meta.current_eta_stdp > initial_eta);
    assert!(meta.current_theta_da < initial_theta);

    // Low global error -> Reduces STDP learning rate for consolidation
    meta.adapt_hyperparameters(FixedPoint::from_f32(0.01));
    meta.adapt_hyperparameters(FixedPoint::from_f32(0.01));
    assert!(meta.current_eta_stdp <= initial_eta);
}

#[test]
fn test_elastic_weight_consolidation_ewc_protection() {
    let mut ewc = ElasticWeightConsolidation::new();

    let src = NeuronId::new(10);
    let tgt = NeuronId::new(20);
    let opt_w = FixedPoint::from_f32(0.75);
    let fisher_imp = FixedPoint::from_f32(0.95);

    ewc.protect_synapse(src, tgt, opt_w, fisher_imp);
    assert_eq!(ewc.count, 1);

    // Attempting to mutate weight away from optimal optimal_weight
    let mutated_w = FixedPoint::from_f32(0.20);
    let penalty = ewc.compute_ewc_penalty_delta(src, tgt, mutated_w);

    assert!(penalty < FixedPoint::ZERO, "EWC penalty must oppose weight perturbation!");
}

#[test]
fn test_full_continual_learning_engine() {
    let engine = continual_learning();

    assert_eq!(engine.offline_replay.replay_count, 0);
    assert_eq!(engine.ewc.count, 0);
}
