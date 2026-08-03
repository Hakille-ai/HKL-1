//! Audio Spike Encoder for HKL-2.
//! Maps 32-band Cochlea Gammatone band responses to 256-dimensional
//! spatio-temporal spike patterns over T timesteps.
#![cfg(feature = "hkl2")]

use crate::audio::cochlea::{CochleaEngine, NUM_COCHLEAR_BANDS};
use crate::core::math::{FixedPoint, XorShift64Star};
use crate::core::memory::NeuronId;
use crate::embedding::spike_embedding::{EMBED_DIM, LIFNeuronLight, TIME_STEPS};
use alloc::vec::Vec;

pub struct AudioSpikeEncoder {
    pub cochlea: CochleaEngine,
    pub projection_weights: Vec<[FixedPoint; EMBED_DIM]>, // NUM_COCHLEAR_BANDS x EMBED_DIM
    pub neurons: [LIFNeuronLight; EMBED_DIM],
}

impl AudioSpikeEncoder {
    pub fn new() -> Self {
        let neurons = core::array::from_fn(|_| {
            LIFNeuronLight::new(FixedPoint::from_f32(0.15), FixedPoint::from_f32(0.9))
        });
        Self {
            cochlea: CochleaEngine::new(NeuronId::new(0)),
            projection_weights: alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; NUM_COCHLEAR_BANDS],
            neurons,
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);
        let k_f32 = 1.0 / (NUM_COCHLEAR_BANDS as f32).sqrt();
        for b in 0..NUM_COCHLEAR_BANDS {
            for d in 0..EMBED_DIM {
                let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k_f32 - k_f32;
                self.projection_weights[b][d] = FixedPoint::from_f32(rand_val);
            }
        }
    }

    /// Encode PCM audio frame into spatio-temporal spike matrix [TIME_STEPS x EMBED_DIM]
    pub fn encode_pcm(&mut self, pcm: &[i16], timestamp: u32) -> [[bool; EMBED_DIM]; TIME_STEPS] {
        let responses = self.cochlea.process_audio_samples(pcm, timestamp);

        // Project 32 ERB band energies -> 256D feature vector
        let mut continuous_feature = [FixedPoint::ZERO; EMBED_DIM];
        for d in 0..EMBED_DIM {
            let mut sum = FixedPoint::ZERO;
            for b in 0..NUM_COCHLEAR_BANDS {
                sum = sum + responses[b].energy * self.projection_weights[b][d];
            }
            continuous_feature[d] = sum;
        }

        // Reset LIF neurons
        for n in self.neurons.iter_mut() {
            n.membrane = FixedPoint::ZERO;
        }

        // Generate spike pattern over T timesteps
        let mut spikes = [[false; EMBED_DIM]; TIME_STEPS];
        for t in 0..TIME_STEPS {
            for d in 0..EMBED_DIM {
                spikes[t][d] = self.neurons[d].step(continuous_feature[d]);
            }
        }

        spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_spike_encoder_encode() {
        let mut encoder = AudioSpikeEncoder::new();
        encoder.init_random(42);

        let mut pcm = [0i16; 512];
        for i in 0..512 {
            pcm[i] = ((i as f32 * 0.1).sin() * 10000.0) as i16;
        }

        let spikes = encoder.encode_pcm(&pcm, 100);
        assert_eq!(spikes.len(), TIME_STEPS);
        assert_eq!(spikes[0].len(), EMBED_DIM);
    }
}
