//! Spiking Auditory & Speech Intelligence Module for HKL-1.
//! Integrates 32-band Gammatone ERB cochlear filter bank, A1 tonotopic map,
//! formant extraction (F1, F2), F0 pitch autocorrelation, and Klatt voice synthesis.

pub mod a1_formants;
pub mod cochlea;
pub mod pitch_rhythm;
pub mod voice_synth;

use core::mem::MaybeUninit;
use a1_formants::{FormantExtractor, FormantProfile};
use cochlea::{BandResponse, CochleaEngine, NUM_COCHLEAR_BANDS};
use pitch_rhythm::{PitchRhythmEngine, PitchRhythmProfile};
use voice_synth::SpikeVoiceSynthesizer;
use crate::core::memory::NeuronId;


/// Unified Auditory & Speech Intelligence Engine
pub struct AudioEngine {
    pub cochlea: CochleaEngine,
    pub pitch_rhythm: PitchRhythmEngine,
    pub voice_synth: SpikeVoiceSynthesizer,
    pub last_formants: FormantProfile,
    pub last_pitch_rhythm: PitchRhythmProfile,
}

impl AudioEngine {
    pub fn new() -> Self {
        Self {
            cochlea: CochleaEngine::new(NeuronId::new(0)),
            pitch_rhythm: PitchRhythmEngine::new(),
            voice_synth: SpikeVoiceSynthesizer::new(),
            last_formants: FormantProfile {
                f1_hz: crate::core::math::FixedPoint::ZERO,
                f2_hz: crate::core::math::FixedPoint::ZERO,
                f3_hz: crate::core::math::FixedPoint::ZERO,
                vowel: a1_formants::VowelClass::Unknown,
            },
            last_pitch_rhythm: PitchRhythmProfile {
                f0_hz: crate::core::math::FixedPoint::ZERO,
                voice_category: pitch_rhythm::VoiceCategory::Unvoiced,
                speech_onset_detected: false,
                syllabic_rate_hz: crate::core::math::FixedPoint::ZERO,
            },
        }
    }

    /// Process input PCM audio samples, extract cochlear bands, formants, pitch F0, and compute voice synthesis
    pub fn process_audio_stream(
        &mut self,
        pcm_samples: &[i16],
        timestamp: u32,
    ) -> (
        [BandResponse; NUM_COCHLEAR_BANDS],
        FormantProfile,
        PitchRhythmProfile,
    ) {
        // 1. Cochlear Gammatone Filtering & PFM Hair Cell Spiking
        let band_responses = self.cochlea.process_audio_samples(pcm_samples, timestamp);

        // 2. Cortex A1 Formant Extraction (F1, F2, F3) & Vowel Classification
        let formants = FormantExtractor::extract_formants(&band_responses);
        self.last_formants = formants;

        // 3. F0 Pitch Autocorrelation & Syllabic Rhythm Detection
        let pitch_rhythm_profile = self.pitch_rhythm.process_pitch_rhythm(pcm_samples);
        self.last_pitch_rhythm = pitch_rhythm_profile;

        // 4. Voice Resonator Waveform Synthesis
        let _pcm_out = self.voice_synth.synthesize_waveform(
            pitch_rhythm_profile.f0_hz,
            formants.f1_hz,
            formants.f2_hz,
        );

        (band_responses, formants, pitch_rhythm_profile)
    }
}

// ---------------------------------------------------------------------------
// Global Instance
// ---------------------------------------------------------------------------
pub static mut AUDIO_ENGINE: MaybeUninit<AudioEngine> = MaybeUninit::uninit();

static INITIALIZED_AUDIO_ENGINE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_audio_engine() {
    unsafe {
        if !INITIALIZED_AUDIO_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            AUDIO_ENGINE.write(AudioEngine::new());
            INITIALIZED_AUDIO_ENGINE.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn audio_engine() -> &'static mut AudioEngine {
    unsafe {
        if !INITIALIZED_AUDIO_ENGINE.load(core::sync::atomic::Ordering::Relaxed) {
            init_audio_engine();
        }
        &mut *AUDIO_ENGINE.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;

    #[test]
    fn audio_engine_new_state() {
        let ae = AudioEngine::new();
        assert_eq!(ae.last_formants.f1_hz, FixedPoint::ZERO);
        assert!(ae.last_pitch_rhythm.voice_category == pitch_rhythm::VoiceCategory::Unvoiced);
    }

    #[test]
    fn audio_engine_process_silence() {
        let mut ae = AudioEngine::new();
        let samples = [0i16; 256];
        let (bands, formants, pitch) = ae.process_audio_stream(&samples, 0);
        assert_eq!(bands.len(), NUM_COCHLEAR_BANDS);
        assert!(formants.f1_hz >= FixedPoint::ZERO);
        assert!(pitch.f0_hz >= FixedPoint::ZERO);
    }

    #[test]
    fn audio_cochlea_band_count() {
        let ce = CochleaEngine::new(NeuronId::new(0));
        assert_eq!(ce.band_energies.len(), NUM_COCHLEAR_BANDS);
    }

    #[test]
    fn audio_formant_extractor_default() {
        let band = BandResponse { frequency_hz: FixedPoint::ZERO, energy: FixedPoint::ZERO, hair_cell_activation: FixedPoint::ZERO };
        let bands = [band; NUM_COCHLEAR_BANDS];
        let pf = FormantExtractor::extract_formants(&bands);
        assert!(pf.f1_hz >= FixedPoint::ZERO);
    }

    #[test]
    fn audio_voice_synth_default() {
        let mut vs = SpikeVoiceSynthesizer::new();
        let pcm = vs.synthesize_waveform(FixedPoint::from_f32(200.0), FixedPoint::from_f32(500.0), FixedPoint::from_f32(1500.0));
        assert!(!pcm.is_empty());
    }
}
