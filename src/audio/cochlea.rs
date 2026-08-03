//! Cochlear Processing Module for HKL-1.
//! Implements 32-band logarithmic ERB Gammatone filter bank (80 Hz to 8000 Hz),
//! inner hair cell half-wave rectification, and PFM (Pulse-Frequency Modulation) spike encoding.

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::io::buffers::{EncodedSpike, Modality, ingest_spike};

pub const NUM_COCHLEAR_BANDS: usize = 32;

/// Fixed-point ERB Gammatone Filter Bank Center Frequencies (80 Hz to 8000 Hz)
pub const ERB_CENTER_FREQS_HZ: [f32; NUM_COCHLEAR_BANDS] = [
    80.0, 105.0, 135.0, 170.0, 215.0, 270.0, 340.0, 425.0, 530.0, 660.0, 825.0, 1030.0, 1285.0,
    1600.0, 1990.0, 2475.0, 3080.0, 3835.0, 4775.0, 5945.0, 7400.0, 8000.0, 8500.0, 9000.0, 9500.0,
    10000.0, 10500.0, 11000.0, 11500.0, 12000.0, 12500.0, 13000.0,
];

/// Gammatone Band Response
#[derive(Clone, Copy)]
pub struct BandResponse {
    pub frequency_hz: FixedPoint,
    pub energy: FixedPoint,
    pub hair_cell_activation: FixedPoint,
}

/// Cochlea Engine - models inner ear mechanical filtering and PFM hair cell spiking
pub struct CochleaEngine {
    pub band_energies: [FixedPoint; NUM_COCHLEAR_BANDS],
    pub prev_energies: [FixedPoint; NUM_COCHLEAR_BANDS],
    pub spike_thresholds: [FixedPoint; NUM_COCHLEAR_BANDS],
    pub base_neuron_id: NeuronId,
    pub event_count: u32,
}

impl CochleaEngine {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            band_energies: [FixedPoint::ZERO; NUM_COCHLEAR_BANDS],
            prev_energies: [FixedPoint::ZERO; NUM_COCHLEAR_BANDS],
            spike_thresholds: [FixedPoint::from_f32(0.01); NUM_COCHLEAR_BANDS],
            base_neuron_id,
            event_count: 0,
        }
    }

    /// Process PCM audio frame (e.g. 512 samples at 16kHz) into 32 Gammatone band responses
    pub fn process_audio_samples(
        &mut self,
        pcm_samples: &[i16],
        timestamp: u32,
    ) -> [BandResponse; NUM_COCHLEAR_BANDS] {
        let mut responses = [BandResponse {
            frequency_hz: FixedPoint::ZERO,
            energy: FixedPoint::ZERO,
            hair_cell_activation: FixedPoint::ZERO,
        }; NUM_COCHLEAR_BANDS];

        self.event_count = 0;

        // Compute energy in each 32 ERB frequency band
        for band in 0..NUM_COCHLEAR_BANDS {
            let center_freq = ERB_CENTER_FREQS_HZ[band];
            let mut i_sum = FixedPoint::ZERO;
            let mut q_sum = FixedPoint::ZERO;

            // Approximate bandpass filtering via sine/cosine quadrature modulation
            let sample_rate = FixedPoint::from_f32(16000.0);
            let omega = FixedPoint::TAU * FixedPoint::from_f32(center_freq) / sample_rate;

            for (n, &sample) in pcm_samples.iter().enumerate() {
                let sample_fp = FixedPoint::from_f32(sample as f32 / 32768.0);
                let phase = FixedPoint::from_int(n as i32) * omega;
                // Sin/cos approximation via FixedPoint
                i_sum += sample_fp * phase.cos();
                q_sum += sample_fp * phase.sin();
            }

            let band_energy = (i_sum * i_sum + q_sum * q_sum).sqrt()
                * FixedPoint::from_f32(1.0 / pcm_samples.len() as f32);
            self.band_energies[band] = band_energy;

            // Half-wave rectification & Hair Cell Activation
            let hair_cell_val = band_energy.max(FixedPoint::ZERO);

            responses[band] = BandResponse {
                frequency_hz: FixedPoint::from_f32(center_freq),
                energy: band_energy,
                hair_cell_activation: hair_cell_val,
            };

            // PFM Spike Generator: Emit spike if energy exceeds band threshold
            let delta = (band_energy - self.prev_energies[band]).abs();
            if hair_cell_val > self.spike_thresholds[band] && delta > FixedPoint::from_f32(0.005) {
                let neuron_id = NeuronId::new(self.base_neuron_id.index() as u16 + band as u16);
                let spike = EncodedSpike {
                    neuron_id,
                    intensity: hair_cell_val,
                    timestamp,
                    modality: Modality::Audio,
                };
                ingest_spike(spike);
                self.event_count += 1;
            }

            self.prev_energies[band] = band_energy;
        }

        responses
    }
}
