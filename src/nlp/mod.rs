//! NLP & Symbolic Cognition Module for HKL-1.
//! Integrates spike token encoding, Winner-Take-All text decoding,
//! neuromodulated state verbalization, and neuro-symbolic knowledge graphs.

pub mod dialogue_engine;
pub mod spike_decoder;
pub mod spike_token;
pub mod symbolic_graph;
pub mod verbalizer;

use core::mem::MaybeUninit;
pub use dialogue_engine::DialogueEngine;

pub static mut DIALOGUE_ENGINE: MaybeUninit<DialogueEngine> = MaybeUninit::uninit();

static INITIALIZED_DIALOGUE_ENGINE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_dialogue_engine() {
    unsafe {
        if !INITIALIZED_DIALOGUE_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            DIALOGUE_ENGINE.write(DialogueEngine::new());
            INITIALIZED_DIALOGUE_ENGINE.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn dialogue_engine() -> &'static mut DialogueEngine {
    unsafe {
        if !INITIALIZED_DIALOGUE_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            init_dialogue_engine();
        }
        &mut *DIALOGUE_ENGINE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;
    use crate::core::memory::NeuronId;
    use crate::nlp::verbalizer::CognitiveStateSummary;

    #[test]
    fn spike_vocabulary_has_128_ascii() {
        let vocab = crate::nlp::spike_token::SpikeVocabulary::new();
        assert_eq!(vocab.tokens[65].symbol[0], b'A');
        assert_eq!(vocab.tokens[97].symbol[0], b'a');
    }

    #[test]
    fn spike_text_encoder_emits_events() {
        let mut enc = crate::nlp::spike_token::SpikeTextEncoder::new(NeuronId::new(100));
        let count = enc.encode_text(b"HELLO", 0);
        assert_eq!(count, 5, "5 chars should produce 5 spike events");
    }

    #[test]
    fn spike_text_encoder_handles_empty() {
        let mut enc = crate::nlp::spike_token::SpikeTextEncoder::new(NeuronId::new(100));
        let count = enc.encode_text(b"", 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn dialogue_engine_new_state() {
        let de = DialogueEngine::new();
        assert_eq!(de.verbal_len, 0);
        assert!(de.last_verbalization.iter().all(|&b| b == 0));
    }

    #[test]
    fn dialogue_engine_process_prompt() {
        let mut de = DialogueEngine::new();
        let cs = CognitiveStateSummary {
            dopamine: FixedPoint::ZERO,
            serotonin: FixedPoint::HALF,
            noradrenaline: FixedPoint::from_f32(0.3),
            acetylcholine: FixedPoint::from_f32(0.7),
            prediction_error: FixedPoint::ZERO,
            curiosity: FixedPoint::ZERO,
            boredom: FixedPoint::ZERO,
        };
        let (_resp, len) = de.process_user_prompt(b"hello", 100, &cs);
        assert!(len > 0, "process_user_prompt must produce verbal output");
    }
}
