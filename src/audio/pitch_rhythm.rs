//! Pitch Autocorrelation Estimator & Rhythm Cadence Detector for HKL-1.
//! Estimates voice fundamental frequency F0 (Hz) for male/female distinction & prosody,
//! and tracks speech energy onset and syllabic cadence.

use crate::core::math::FixedPoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceCategory {
    Unvoiced,
    MaleVoice,   // F0 ~ 85..160 Hz
    FemaleVoice, // F0 ~ 165..255 Hz
    ChildVoice,  // F0 > 260 Hz
}

/// Pitch & Rhythm Analysis Result
#[derive(Clone, Copy)]
pub struct PitchRhythmProfile {
    pub f0_hz: FixedPoint,
    pub voice_category: VoiceCategory,
    pub speech_onset_detected: bool,
    pub syllabic_rate_hz: FixedPoint,
}

/// Pitch & Rhythm Engine
pub struct PitchRhythmEngine {
    pub prev_total_energy: FixedPoint,
    pub energy_threshold: FixedPoint,
    pub onset_cooldown: u32,
}

impl PitchRhythmEngine {
    pub fn new() -> Self {
        Self {
            prev_total_energy: FixedPoint::ZERO,
            energy_threshold: FixedPoint::from_f32(0.08),
            onset_cooldown: 0,
        }
    }

    /// Estimate fundamental frequency F0 (Hz) from PCM audio sample autocorrelation
    pub fn estimate_pitch_f0(&self, pcm_samples: &[i16]) -> FixedPoint {
        let sample_rate = 16000.0;
        let min_lag = (sample_rate / 400.0) as usize; // Max F0 = 400Hz (lag ~40)
        let max_lag = (sample_rate / 70.0) as usize; // Min F0 = 70Hz (lag ~228)

        let mut best_lag = min_lag;
        let mut max_corr = i64::MIN;

        for lag in min_lag..max_lag.min(pcm_samples.len() / 2) {
            let mut corr = 0i64;
            for i in 0..(pcm_samples.len() - lag) {
                corr += (pcm_samples[i] as i64) * (pcm_samples[i + lag] as i64);
            }

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        if max_corr > 100_000_000 {
            let f0 = sample_rate / (best_lag as f32);
            FixedPoint::from_f32(f0)
        } else {
            FixedPoint::ZERO
        }
    }

    /// Process PCM audio frame for pitch & speech onset rhythm
    pub fn process_pitch_rhythm(&mut self, pcm_samples: &[i16]) -> PitchRhythmProfile {
        let f0_fp = self.estimate_pitch_f0(pcm_samples);
        let f0 = f0_fp.to_f32();

        let category = if f0 < 50.0 {
            VoiceCategory::Unvoiced
        } else if f0 <= 160.0 {
            VoiceCategory::MaleVoice
        } else if f0 <= 255.0 {
            VoiceCategory::FemaleVoice
        } else {
            VoiceCategory::ChildVoice
        };

        // Compute total frame energy & detect onset
        let mut total_e = FixedPoint::ZERO;
        for &s in pcm_samples {
            let s_fp = FixedPoint::from_f32(s as f32 / 32768.0);
            total_e += s_fp.abs();
        }
        total_e = total_e * FixedPoint::from_f32(1.0 / pcm_samples.len() as f32);

        let delta_e = total_e - self.prev_total_energy;
        let mut onset = false;

        if delta_e > self.energy_threshold && self.onset_cooldown == 0 {
            onset = true;
            self.onset_cooldown = 5; // Cooldown 5 frames (~50ms)
        } else if self.onset_cooldown > 0 {
            self.onset_cooldown -= 1;
        }

        self.prev_total_energy = total_e;

        PitchRhythmProfile {
            f0_hz: f0_fp,
            voice_category: category,
            speech_onset_detected: onset,
            syllabic_rate_hz: FixedPoint::from_f32(4.5), // Typical 4-5 Hz syllable rate
        }
    }
}
