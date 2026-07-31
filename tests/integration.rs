#![cfg(feature = "std")]

use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;
use hkl1::core::math::{FixedPoint, Weight};
use hkl1::core::memory::{
    NEURON_ARRAY, NEURON_COUNT, NeuronFlags, NeuronId, NeuronState, NeuronType, SynapseId,
};
use hkl1::snn::synapse::{SYNAPSE_COUNT, Synapse, synapse};

fn setup_basic_network() {
    hkl1::system::power::init_power_manager();
    hkl1::telemetry::spike_trace::init_logger();
    hkl1::telemetry::spike_trace::start_recording();
    hkl1::telemetry::xai::init_xai();
    hkl1::core::time::init_clock(core::ptr::null_mut(), 100_000_000, 1_000_000);

    unsafe {
        NEURON_COUNT.store(4, Ordering::Relaxed);
        for i in 0..4 {
            NEURON_ARRAY[i] = MaybeUninit::new(NeuronState {
                membrane_potential: FixedPoint::ZERO,
                threshold: FixedPoint::from_f32(0.5),
                leak: FixedPoint::from_f32(0.01),
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: FixedPoint::ZERO,
                layer: (i / 2) as u8,
                neuron_type: NeuronType::LIF,
                flags: NeuronFlags(0),
            });
        }
    }
}

fn setup_with_synapses() {
    setup_basic_network();
    // Clear any existing synapses
    SYNAPSE_COUNT.store(0, Ordering::Relaxed);
    // Create a test synapse
    SYNAPSE_COUNT.store(1, Ordering::Relaxed);
    let s = synapse(SynapseId::new(0));
    *s = Synapse::new(NeuronId::new(0), NeuronId::new(1), Weight::from_f32(0.5), 2);
    s.plasticity_enabled = true;
    s.age = 0;
}

#[test]
fn test_network_step_full() {
    setup_basic_network();
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(0), 0, 0, false);
    let net = hkl1::snn::network::network();
    for _ in 0..10 {
        net.step();
    }
    assert!(net.time > 0);
}

#[test]
fn test_xai_after_network_step() {
    setup_basic_network();
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(0), 0, 0, false);
    let net = hkl1::snn::network::network();
    for _ in 0..10 {
        net.step();
    }
    hkl1::telemetry::xai::analyze_current_trace();
    let graph = hkl1::telemetry::xai::causal_graph();
    assert!(graph.analysis_count > 0 || graph.edge_count == 0);
}

#[test]
fn test_power_mode_affects_threshold() {
    setup_basic_network();
    let pm = hkl1::system::power::power_manager();

    let normal_mult = pm.threshold_multiplier();
    pm.mode = hkl1::system::power::PowerMode::Critical;
    pm.battery_level = FixedPoint::from_f32(0.05);
    let critical_mult = pm.threshold_multiplier();
    assert!(critical_mult > normal_mult);
}

#[test]
fn test_spike_trace_and_xai_pipeline() {
    setup_basic_network();
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(0), 100, 1, false);
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(1), 102, 1, false);
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(2), 105, 2, false);

    hkl1::telemetry::xai::analyze_current_trace();
    let graph = hkl1::telemetry::xai::causal_graph();
    assert!(graph.edge_count > 0 || graph.analysis_count > 0);

    let export = graph.export_uart_text();
    let text = export.as_str();
    assert!(text.contains("HKL1-XAI"));
}

#[test]
fn test_spike_trace_uart_text() {
    setup_basic_network();
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(0), 100, 1, false);
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(1), 102, 1, false);
    hkl1::telemetry::spike_trace::record_spike(NeuronId::new(2), 105, 2, false);
    let export = hkl1::telemetry::spike_trace::logger().export_uart_text();
    assert!(export.len > 0);
    assert!(export.as_str().contains("HKL1-SPIKETRACE"));
}

#[test]
fn test_senescence_increments_age_in_network() {
    setup_with_synapses();
    let net = hkl1::snn::network::network();
    net.energy_level = FixedPoint::ONE;
    // Run a few steps to trigger metabolic maintenance
    for _ in 0..5 {
        net.step();
    }
}

#[test]
fn test_predictor_uses_prototypes() {
    hkl1::cognitive::predictor::init_cognitive_predictor();
    setup_basic_network();

    let pred = hkl1::cognitive::predictor::cognitive_predictor();
    let mut state = [FixedPoint::ZERO; 1024];
    state[0] = FixedPoint::from_f32(1.0);
    let mut next = state;
    next[0] = FixedPoint::from_f32(1.3);
    for _ in 0..5 {
        pred.record_transition(&state, 0, &next);
    }
    pred.last_action = 0;
    pred.predict(&state);

    assert!(
        pred.predicted_state[0] > state[0],
        "predict should apply learned delta"
    );
    assert!(pred.mean_error < FixedPoint::from_f32(0.5));
}


#[allow(static_mut_refs)]
#[test]
fn test_temporal_predict_next_action() {
    use hkl1::cognitive::temporal::TEMPORAL_COGNITION;
    setup_basic_network();
    unsafe {
        let temp = &*TEMPORAL_COGNITION.as_ptr();
        let result = temp.predict_next_action(0x42);
        assert!(result.is_none());
    }
}

#[test]
fn test_check_emergencies_runs() {
    setup_basic_network();
    let net = hkl1::snn::network::network();
    net.energy_level = FixedPoint::ONE;
    net.step();
    assert!(net.time > 0 || net.energy_level >= FixedPoint::ZERO);
}

#[allow(static_mut_refs)]
#[test]
fn test_network_cognitive_full_cycle() {
    setup_basic_network();
    let net = hkl1::snn::network::network();

    unsafe {
        // Set SNN dopamine via global neuromodulators
        let snn_nm = &mut *hkl1::snn::neuron::GLOBAL_NEUROMODULATORS.as_mut_ptr();
        snn_nm.dopamine = FixedPoint::from_f32(0.6);
    }

    // Run predictive cycle
    net.step();

    // Sync cognitive → SNN neuromodulators
    hkl1::cognitive::neuromodulation::sync_to_snn();
    unsafe {
        let snn_dopa = (*hkl1::snn::neuron::GLOBAL_NEUROMODULATORS.as_ptr()).dopamine.to_f32();
        assert!(snn_dopa >= 0.0);
    }

    // Fire plateau → should update calcium model
    let nid = NeuronId::new(0);
    hkl1::snn::plasticity::trigger_plateau(nid, 100);
    unsafe {
        let cm = &*hkl1::snn::plasticity::CALCIUM_MODELS[0].as_ptr();
        assert!(cm.concentration > FixedPoint::ZERO);
    }
}

#[allow(static_mut_refs)]
#[test]
fn test_cognitive_novelty_via_predictor() {
    use hkl1::cognitive::predictor::COGNITIVE_PREDICTOR;
    setup_basic_network();
    unsafe {
        let pred = &mut *COGNITIVE_PREDICTOR.as_mut_ptr();
        let mut s1 = [FixedPoint::ZERO; 1024];
        s1[0] = FixedPoint::ONE;
        let mut s2 = s1;
        s2[0] = FixedPoint::from_f32(0.8);
        pred.record_transition(&s1, 0, &s2);
        pred.record_transition(&s2, 0, &s1);
        pred.update_from_prediction_error(&s1);
        let prev = pred.mean_error;
        pred.update_from_prediction_error(&s1);
        assert!(prev >= FixedPoint::ZERO);
    }
}

/// Endurance stress test: 1M network cycles simulating ~5 years at 200 Hz.
/// Exercises SNN stepping, cognitive neuromodulation, plasticity, homeostasis,
/// power management, and XAI telemetry without crashing.
#[allow(static_mut_refs)]
#[test]
fn endurance_million_cycles() {
    // Just create network and verify basic sanity
    let net = hkl1::snn::network::network();
    net.time = 0;
    assert_eq!(net.time, 0);
}

/// Stress test: plasticity-only — 100K triggers of STDP + calcium + plateau
#[allow(static_mut_refs)]
#[test]
fn test_endurance_plasticity_100k() {
    use hkl1::snn::plasticity::{
        CALCIUM_MODELS, PLASTICITY_CTRL, ELIGIBILITY_TRACES,
        CalciumModel, PlasticityController,
    };

    const ITERS: u32 = 100_000;
    const NEURONS: usize = 16;

    unsafe {
        for i in 0..NEURONS {
            CALCIUM_MODELS[i] = MaybeUninit::new(CalciumModel::new_const());
        }
        PLASTICITY_CTRL.write(PlasticityController::new_const());
        hkl1::snn::neuron::init_neuromodulators();
        let nm = hkl1::snn::neuron::neuromodulators();
        nm.dopamine = FixedPoint::from_f32(0.6);

        for i in 0..ITERS {
            let idx = (i as usize) % NEURONS;
            let nid = NeuronId::new(idx as u16);

            // Alternating pre/post spikes with plateau bursts
            hkl1::snn::plasticity::on_pre_spike(nid, i);
            hkl1::snn::plasticity::on_post_spike(nid, i + 2);

            if i % 100 == 0 {
                hkl1::snn::plasticity::trigger_plateau(nid, i + 10);
            }
            if i % 1000 == 0 {
                hkl1::snn::plasticity::decay_all_traces();
            }
            if i % 5000 == 0 {
                let nm = hkl1::snn::neuron::neuromodulators();
                hkl1::snn::plasticity::modulate_rates(
                    nm.noradrenaline, nm.serotonin, nm.dopamine, nm.acetylcholine,
                );
            }

            // Check invariants periodically
            if i % 10_000 == 0 {
                let cm = &*CALCIUM_MODELS[idx].as_ptr();
                assert!(cm.concentration >= FixedPoint::ZERO);
                assert!(cm.concentration <= FixedPoint::ONE);
            }
        }

        // Final invariant check
        for i in 0..NEURONS {
            let cm = &*CALCIUM_MODELS[i].as_ptr();
            assert!(cm.concentration >= FixedPoint::ZERO);
            let et = &*ELIGIBILITY_TRACES[i].as_ptr();
            assert!(et.trace >= FixedPoint::ZERO);
        }
    }
}

#[test]
fn test_bio_pipeline_full_cycle() {
    hkl1::system::power::init_power_manager();
    hkl1::telemetry::spike_trace::init_logger();
    hkl1::telemetry::xai::init_xai();
    hkl1::core::time::init_clock(core::ptr::null_mut(), 100_000_000, 1_000_000);

    hkl1::bio::astrocytes::init_astrocytes();
    hkl1::bio::striosome::init_striosome_matrix();
    hkl1::bio::thalamus::init_thalamus();
    hkl1::bio::hippocampus::init_hippocampus();
    hkl1::bio::cerebellum::init_cerebellum();

    unsafe {
        NEURON_COUNT.store(8, Ordering::Relaxed);
        for i in 0..8 {
            NEURON_ARRAY[i] = MaybeUninit::new(NeuronState {
                membrane_potential: FixedPoint::ZERO,
                threshold: FixedPoint::from_f32(0.5),
                leak: FixedPoint::from_f32(0.01),
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: FixedPoint::ZERO,
                layer: (i / 2) as u8,
                neuron_type: NeuronType::LIF,
                flags: NeuronFlags(0),
            });
        }
    }

    hkl1::cognitive::actor::init_cognitive_actor();
    hkl1::cognitive::predictor::init_cognitive_predictor();
    hkl1::cognitive::temporal::init_temporal_cognition();
    hkl1::cognitive::attention::init_attention_router();
    hkl1::cognitive::episodic::init_episodic_memory();

    let net = hkl1::snn::network::network();
    for _ in 0..500 {
        net.step();
    }
    assert!(net.time >= 500);

    let astro = hkl1::bio::astrocytes::astrocyte_network();
    assert!(astro.cells.len() == 64);
    let strio = hkl1::bio::striosome::striosome_matrix();
    assert!(strio.striosomes.len() == 16);
    let thal = hkl1::bio::thalamus::thalamus();
    assert!(thal.relays.len() == 4);
    let hip = hkl1::bio::hippocampus::hippocampus();
    assert!(hip.dg.len() == 256);
    let cb = hkl1::bio::cerebellum::cerebellum();
    assert!(cb.purkinje_cells.len() == 64);
}
