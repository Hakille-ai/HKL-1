//! Spiking Formant Voice Resonator & PCM Audio Synthesizer for HKL-1.
//! Converts Layer 4 motor/vocal neuron spiking potentials into 16-bit 16kHz PCM audio waveforms.

use crate::core::math::FixedPoint;

pub const SYNTH_BUFFER_SIZE: usize = 512; // 32ms buffer at 16kHz

pub struct BiquadFilter {
    pub x1: f32,
    pub x2: f32,
    pub y1: f32,
    pub y2: f32,
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadFilter {
    pub fn from_resonance(freq_hz: f32, bandwidth_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * core::f32::consts::PI * freq_hz / sample_rate;
        let q = freq_hz / bandwidth_hz;
        // prevent div by zero
        let q_safe = if q < 0.1 { 0.1 } else { q };
        let w0_fp = FixedPoint::from_f32(w0);
        let alpha = w0_fp.sin().to_f32() / (2.0 * q_safe);
        let a0 = 1.0 + alpha;

        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * w0_fp.cos().to_f32()) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let y = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Spiking Voice Resonator & Formant Synthesizer
pub struct SpikeVoiceSynthesizer {
    pub sample_rate_hz: f32,
    pub pcm_output: [i16; SYNTH_BUFFER_SIZE],
    pub phase: f32,
    pub biquad_f1: BiquadFilter,
    pub biquad_f2: BiquadFilter,
}

impl SpikeVoiceSynthesizer {
    pub fn new() -> Self {
        Self {
            sample_rate_hz: 16000.0,
            pcm_output: [0; SYNTH_BUFFER_SIZE],
            phase: 0.0,
            biquad_f1: BiquadFilter::from_resonance(500.0, 50.0, 16000.0),
            biquad_f2: BiquadFilter::from_resonance(1500.0, 50.0, 16000.0),
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

        let mut new_f1 = BiquadFilter::from_resonance(f1, 50.0, self.sample_rate_hz);
        new_f1.x1 = self.biquad_f1.x1;
        new_f1.x2 = self.biquad_f1.x2;
        new_f1.y1 = self.biquad_f1.y1;
        new_f1.y2 = self.biquad_f1.y2;
        self.biquad_f1 = new_f1;

        let mut new_f2 = BiquadFilter::from_resonance(f2, 50.0, self.sample_rate_hz);
        new_f2.x1 = self.biquad_f2.x1;
        new_f2.x2 = self.biquad_f2.x2;
        new_f2.y1 = self.biquad_f2.y1;
        new_f2.y2 = self.biquad_f2.y2;
        self.biquad_f2 = new_f2;

        let dt = 1.0 / self.sample_rate_hz;

        for n in 0..SYNTH_BUFFER_SIZE {
            self.phase += f0 * dt;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }

            // Glottal pulse excitation waveform (sawtooth approximation)
            let glottal_pulse = 2.0 * self.phase - 1.0;

            // Formant resonance filter modulations (F1 + F2)
            let f1_out = self.biquad_f1.process(glottal_pulse);
            let f2_out = self.biquad_f2.process(glottal_pulse);

            let acoustic_val = f1_out + f2_out * 0.5;
            let pcm_val = (acoustic_val * 16384.0).clamp(-32768.0, 32767.0) as i16;

            self.pcm_output[n] = pcm_val;
        }

        &self.pcm_output
    }
}
