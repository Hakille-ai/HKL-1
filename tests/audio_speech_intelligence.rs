#![cfg(feature = "std")]

use hkl1::audio::a1_formants::{FormantExtractor, VowelClass};
use hkl1::audio::cochlea::{CochleaEngine, NUM_COCHLEAR_BANDS};
use hkl1::audio::pitch_rhythm::PitchRhythmEngine;

use hkl1::audio::voice_synth::SpikeVoiceSynthesizer;
use hkl1::audio::audio_engine;
use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;

#[test]
fn test_cochlea_gammatone_erb_filtering() {
    let mut cochlea = CochleaEngine::new(NeuronId::new(0));

    // Generate 512 PCM samples of a 1000 Hz sine wave tone
    let sample_rate = 16000.0;
    let freq = 1000.0;
    let mut pcm = [0i16; 512];

    for n in 0..512 {
        let phase = 2.0 * 3.1415926 * freq * (n as f32) / sample_rate;
        pcm[n] = (phase.sin() * 20000.0) as i16;
    }

    let responses = cochlea.process_audio_samples(&pcm, 1000);

    // Band ~11 is centered at 1030 Hz
    assert!(responses[11].energy > FixedPoint::ZERO, "1kHz band must receive energy");
    assert!(cochlea.event_count > 0, "PFM hair cells must generate spikes for 1kHz tone");
}

#[test]
fn test_formant_extraction_and_vowel_classification() {
    // Vowel /a/ Formants: F1 ~ 730 Hz, F2 ~ 1090 Hz
    let vowel_a = FormantExtractor::classify_vowel(730.0, 1090.0);
    assert_eq!(vowel_a, VowelClass::VowelA);

    // Vowel /i/ Formants: F1 ~ 270 Hz, F2 ~ 2290 Hz
    let vowel_i = FormantExtractor::classify_vowel(270.0, 2290.0);
    assert_eq!(vowel_i, VowelClass::VowelI);

    // Vowel /u/ Formants: F1 ~ 300 Hz, F2 ~ 870 Hz
    let vowel_u = FormantExtractor::classify_vowel(300.0, 870.0);
    assert_eq!(vowel_u, VowelClass::VowelU);
}

#[test]
fn test_pitch_f0_autocorrelation_estimation() {
    let pitch_engine = PitchRhythmEngine::new();

    // Generate 512 PCM samples of 120 Hz Male Voice pitch tone
    let sample_rate = 16000.0;
    let f0 = 120.0;
    let mut pcm = [0i16; 512];

    for n in 0..512 {
        let phase = 2.0 * 3.1415926 * f0 * (n as f32) / sample_rate;
        pcm[n] = (phase.sin() * 20000.0) as i16;
    }

    let estimated_f0 = pitch_engine.estimate_pitch_f0(&pcm);
    assert!(estimated_f0 > FixedPoint::from_f32(100.0));
    assert!(estimated_f0 < FixedPoint::from_f32(140.0));
}

#[test]
fn test_speech_onset_and_rhythm_detection() {
    let mut pitch_engine = PitchRhythmEngine::new();

    let pcm_quiet = [100i16; 512];
    let pcm_loud = [20000i16; 512];

    let profile1 = pitch_engine.process_pitch_rhythm(&pcm_quiet);
    assert!(!profile1.speech_onset_detected);

    let profile2 = pitch_engine.process_pitch_rhythm(&pcm_loud);
    assert!(profile2.speech_onset_detected, "Sudden energy onset must be detected!");
}

#[test]
fn test_spiking_voice_pcm_synthesizer() {
    let mut synth = SpikeVoiceSynthesizer::new();

    let f0 = FixedPoint::from_f32(140.0);
    let f1 = FixedPoint::from_f32(700.0);
    let f2 = FixedPoint::from_f32(1200.0);

    let pcm_out = synth.synthesize_waveform(f0, f1, f2);

    assert_eq!(pcm_out.len(), 512);

    // Verify non-silent synthesized PCM samples
    let mut non_zero = false;
    for &sample in pcm_out {
        if sample != 0 {
            non_zero = true;
            break;
        }
    }
    assert!(non_zero, "Synthesizer must generate non-zero PCM audio samples");
}

#[test]
fn test_full_audio_engine_pipeline() {
    let engine = audio_engine();
    let pcm = [5000i16; 512];

    let (bands, formants, pitch) = engine.process_audio_stream(&pcm, 1000);

    assert_eq!(bands.len(), NUM_COCHLEAR_BANDS);
    assert!(formants.f1_hz >= FixedPoint::ZERO);
    assert!(pitch.syllabic_rate_hz > FixedPoint::ZERO);
}
