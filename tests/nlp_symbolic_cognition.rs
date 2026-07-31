#![cfg(feature = "std")]

use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;
use hkl1::nlp::dialogue_engine;
use hkl1::nlp::spike_decoder::SpikeTextDecoder;
use hkl1::nlp::spike_token::SpikeTextEncoder;
use hkl1::nlp::symbolic_graph::SymbolicKnowledgeGraph;
use hkl1::nlp::verbalizer::{CognitiveStateSummary, NeuromodulatedVerbalizer};

#[test]
fn test_spike_token_encoding_phase_timing() {
    let mut encoder = SpikeTextEncoder::new(NeuronId::new(0));

    let text = b"Hello";
    let events = encoder.encode_text(text, 1000);

    assert_eq!(events, 5, "Must encode 5 character spikes");
    assert_eq!(encoder.event_count, 5);
}

#[test]
fn test_spike_token_decoding_wta() {
    let mut decoder = SpikeTextDecoder::new();

    let mut motor_potentials = [FixedPoint::ZERO; 256];
    motor_potentials[b'A' as usize] = FixedPoint::from_f32(0.8);
    motor_potentials[b'B' as usize] = FixedPoint::from_f32(0.3);

    let decoded = decoder.decode_firing_rates(&motor_potentials);

    assert_eq!(
        decoded,
        Some(b'A'),
        "WTA must select token with highest firing potential"
    );
    assert_eq!(decoder.get_response_text(), b"A");
}

#[test]
fn test_neuromodulated_verbalizer_modes() {
    // 1. Stable Mode
    let state_stable = CognitiveStateSummary {
        dopamine: FixedPoint::from_f32(0.8),
        serotonin: FixedPoint::from_f32(0.7),
        noradrenaline: FixedPoint::from_f32(0.2),
        acetylcholine: FixedPoint::from_f32(0.5),
        prediction_error: FixedPoint::from_f32(0.01),
        curiosity: FixedPoint::from_f32(0.2),
        boredom: FixedPoint::from_f32(0.1),
    };

    let (buf_s, len_s) = NeuromodulatedVerbalizer::verbalize_state(&state_stable);
    let str_s = std::str::from_utf8(&buf_s[..len_s]).unwrap();
    assert!(str_s.contains("[STABLE]"));

    // 2. Alert/Crisis Mode
    let state_alert = CognitiveStateSummary {
        dopamine: FixedPoint::from_f32(0.3),
        serotonin: FixedPoint::from_f32(0.2),
        noradrenaline: FixedPoint::from_f32(0.9), // High NA
        acetylcholine: FixedPoint::from_f32(0.8),
        prediction_error: FixedPoint::from_f32(0.65),
        curiosity: FixedPoint::from_f32(0.8),
        boredom: FixedPoint::from_f32(0.0),
    };

    let (buf_a, len_a) = NeuromodulatedVerbalizer::verbalize_state(&state_alert);
    let str_a = std::str::from_utf8(&buf_a[..len_a]).unwrap();
    assert!(str_a.contains("[ALERT/CRISIS]"));
}

#[test]
fn test_symbolic_knowledge_graph_spreading_activation() {
    let mut graph = SymbolicKnowledgeGraph::new();

    let cat_id = graph.add_concept(b"cat");
    let mouse_id = graph.add_concept(b"mouse");

    // Add triple ("cat", "chases", "mouse") — Relation ID 2
    graph.add_triple(b"cat", 2, b"mouse");

    assert_eq!(graph.concept_count, 2);
    assert_eq!(graph.triple_count, 1);

    // Activate "cat" with 1.0 energy
    graph.activate_and_propagate(cat_id, FixedPoint::ONE);

    // Verify spreading activation reached "mouse" concept node
    assert_eq!(graph.concepts[cat_id as usize].activation, FixedPoint::ONE);
    assert!(graph.concepts[mouse_id as usize].activation > FixedPoint::ZERO);
}

#[test]
fn test_dialogue_engine_full_interaction() {
    let engine = dialogue_engine();

    let state = CognitiveStateSummary {
        dopamine: FixedPoint::from_f32(0.75),
        serotonin: FixedPoint::from_f32(0.60),
        noradrenaline: FixedPoint::from_f32(0.30),
        acetylcholine: FixedPoint::from_f32(0.50),
        prediction_error: FixedPoint::from_f32(0.05),
        curiosity: FixedPoint::from_f32(0.40),
        boredom: FixedPoint::from_f32(0.10),
    };

    let (buf, len) = engine.process_user_prompt(b"Hello", 1000, &state);
    let verbal_str = std::str::from_utf8(&buf[..len]).unwrap();

    assert!(verbal_str.contains("State:"));
    assert_eq!(engine.encoder.event_count, 5);
}
