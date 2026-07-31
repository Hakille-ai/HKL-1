//! Spiking Formant Voice Resonator & PCM Audio Synthesizer for HKL-1.
//! Converts Layer 4 motor/vocal neuron spiking potentials into 16-bit 16kHz PCM audio waveforms.

use crate::core::math::FixedPoint;

pub const SYNTH_BUFFER_SIZE: usize = 512; // 32ms buffer at 16kHz

/// Spiking Voice Resonator & Formant Synthesizer
pub struct SpikeVoiceSynthesizer {
    pub sample_rate_hz: f32,
    pub pcm_output: [i16; SYNTH_BUFFER_SIZE],
    pub phase: f32,
}

impl SpikeVoiceSynthesizer {
    pub fn new() -> Self {
        Self {
            sample_rate_hz: 16000.0,
            pcm_output: [0; SYNTH_BUFFER_SIZE],
            phase: 0.0,
        }
    }

    /// Synthesize 512 PCM samples (16-bit 16kHz) from vocal motor neuron spike potentials (F0 pitch, F1, F2)
    pub fn synthesize_waveform(
        &mut self,
        vocal_pitch_f0: FixedPoint,
        formant_f1: FixedPoint,
        formant_f2: FixedPoint,
    ) -> &[i16; SYNTH_BUFFER_SIZE] {
        let f0 = vocal_pitch_f0.to_f32().clamp(70.0, 400.0);
        let f1 = formant_f1.to_f32().clamp(200.0, 1200.0);
        let f2 = formant_f2.to_f32().clamp(600.0, 3000.0);

        let dt = 1.0 / self.sample_rate_hz;

        for n in 0..SYNTH_BUFFER_SIZE {
            self.phase += f0 * dt;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }

            // Glottal pulse excitation waveform (sawtooth approximation)
            let glottal_pulse = 2.0 * self.phase - 1.0;

            // Formant resonance filter modulations (F1 + F2)
            let dt_fp = FixedPoint::ONE / FixedPoint::from_f32(self.sample_rate_hz);
            let t = FixedPoint::from_int(n as i32) * dt_fp;
            let r1 = (FixedPoint::TAU * FixedPoint::from_f32(f1) * t).sin();
            let r2 =
                (FixedPoint::TAU * FixedPoint::from_f32(f2) * t).sin() * FixedPoint::from_f32(0.5);

            let acoustic_val = glottal_pulse * (r1.to_f32() + r2.to_f32());
            let pcm_val = (acoustic_val * 16384.0).clamp(-32768.0, 32767.0) as i16;

            self.pcm_output[n] = pcm_val;
        }

        &self.pcm_output
    }
}
