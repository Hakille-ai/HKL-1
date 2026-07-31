//! HKL-1 Neuromorphic SNN & Cognitive Engine Integration Demo
//! Demonstrates bare-metal spiking neural network stepping, multimodal sensory input,
//! cognitive neuromodulation, and XAI causal graph telemetry.

use hkl1::core::math::FixedPoint;
use hkl1::snn::network::network;

fn main() {
    println!("==================================================");
    println!("    HKL-1 Neuromorphic Engine - Bare-Metal SNN    ");
    println!("==================================================");

    // 1. Initialize Subsystems
    hkl1::system::power::init_power_manager();
    hkl1::telemetry::spike_trace::init_logger();
    hkl1::telemetry::xai::init_xai();
    hkl1::cognitive::temporal::init_temporal_cognition();
    hkl1::cognitive::predictor::init_cognitive_predictor();

    let net = network();

    // 2. Hardware auto-adaptation
    let hw_profile = net.auto_adapt_hardware();
    println!(
        "[HW Detector] Cores: {}, Recommended Neurons: {}, Synapses: {}",
        hw_profile.cpu_cores,
        hw_profile.recommended_max_neurons,
        hw_profile.recommended_max_synapses
    );

    // 3. Step simulation for 100 cycles (100 ms)
    println!("\n[SNN Engine] Running 100 simulation steps...");
    for step in 1..=100 {
        net.step();

        if step % 25 == 0 {
            println!(
                "  Step {:3} | Time: {:4} ms | Energy: {:.2} | Neuromodulators -> DA: {:.2}, 5-HT: {:.2}, NA: {:.2}, ACh: {:.2}",
                step,
                net.time,
                net.energy_level.to_f32(),
                net.neuromodulators.dopamine.to_f32(),
                net.neuromodulators.serotonin.to_f32(),
                net.neuromodulators.noradrenaline.to_f32(),
                net.neuromodulators.acetylcholine.to_f32(),
            );
        }
    }

    // 4. Multimodal Audio Processing Demo
    println!("\n[Auditory Engine] Processing 16 kHz PCM audio waveform...");
    hkl1::audio::init_audio_engine();
    let audio_eng = hkl1::audio::audio_engine();
    let dummy_pcm = [500i16; 512];
    let (_bands, formants, pitch) = audio_eng.process_audio_stream(&dummy_pcm, 100);
    println!(
        "  Extracted Formants -> F1: {:.1} Hz, F2: {:.1} Hz | Pitch F0: {:.1} Hz ({:?})",
        formants.f1_hz.to_f32(),
        formants.f2_hz.to_f32(),
        pitch.f0_hz.to_f32(),
        pitch.voice_category
    );

    // 5. NLP & Symbolic Cognition Dialogue Demo
    println!("\n[NLP Engine] Running neuro-symbolic dialogue engine...");
    hkl1::nlp::init_dialogue_engine();
    let nlp_eng = hkl1::nlp::dialogue_engine();
    let cog_state = hkl1::nlp::verbalizer::CognitiveStateSummary {
        dopamine: net.neuromodulators.dopamine,
        serotonin: net.neuromodulators.serotonin,
        noradrenaline: net.neuromodulators.noradrenaline,
        acetylcholine: net.neuromodulators.acetylcholine,
        prediction_error: FixedPoint::from_f32(0.1),
        curiosity: FixedPoint::from_f32(0.8),
        boredom: FixedPoint::ZERO,
    };
    let (verbal_buf, v_len) = nlp_eng.process_user_prompt(b"status", net.time, &cog_state);
    let verbal_str = core::str::from_utf8(&verbal_buf[..v_len]).unwrap_or("OK");
    println!("  Verbalized State: {}", verbal_str);

    // 6. XAI Causal Graph Telemetry
    println!("\n[XAI Telemetry] Querying causal graph feature attributions...");
    let xai = hkl1::telemetry::xai::causal_graph();
    let avg_confidence = xai.avg_confidence;
    println!(
        "  Causal Edges: {}, Avg Confidence: {:.3}",
        xai.edge_count,
        avg_confidence.to_f32()
    );

    println!("\n==================================================");
    println!("           HKL-1 Demo Run Completed               ");
    println!("==================================================");
}
