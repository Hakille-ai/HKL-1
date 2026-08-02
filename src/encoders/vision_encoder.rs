//! Vision Spike Encoder for HKL-2.
//! Maps Retinal Engine ganglion responses (1024 pixels) to 256-dimensional
//! spatio-temporal spike patterns over T timesteps.
#![cfg(feature = "hkl2")]

use crate::core::math::{FixedPoint, XorShift64Star};
use crate::core::memory::NeuronId;
use crate::embedding::spike_embedding::{EMBED_DIM, LIFNeuronLight, TIME_STEPS};
use crate::vision::retina::{RetinalEngine, VISION_PIXELS};
use alloc::vec::Vec;

pub struct VisionSpikeEncoder {
    pub retina: RetinalEngine,
    pub projection_weights: Vec<[FixedPoint; EMBED_DIM]>, // VISION_PIXELS x EMBED_DIM
    pub neurons: [LIFNeuronLight; EMBED_DIM],
}

impl VisionSpikeEncoder {
    pub fn new() -> Self {
        let neurons = core::array::from_fn(|_| {
            LIFNeuronLight::new(FixedPoint::from_f32(0.15), FixedPoint::from_f32(0.9))
        });
        Self {
            retina: RetinalEngine::new(NeuronId::new(0)),
            projection_weights: alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; VISION_PIXELS],
            neurons,
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);
        let k_f32 = 1.0 / (VISION_PIXELS as f32).sqrt();
        for p in 0..VISION_PIXELS {
            for d in 0..EMBED_DIM {
                let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k_f32 - k_f32;
                self.projection_weights[p][d] = FixedPoint::from_f32(rand_val);
            }
        }
    }

    /// Encode 32x32 visual frame into spatio-temporal spike matrix [TIME_STEPS x EMBED_DIM]
    pub fn encode_frame(
        &mut self,
        frame: &[u8; VISION_PIXELS],
        timestamp: u32,
    ) -> [[bool; EMBED_DIM]; TIME_STEPS] {
        let ganglion_responses = self.retina.process_frame(frame, timestamp);

        // Project 1024 ganglion ON/OFF responses -> 256D feature vector
        let mut continuous_feature = [FixedPoint::ZERO; EMBED_DIM];
        for d in 0..EMBED_DIM {
            let mut sum = FixedPoint::ZERO;
            for p in 0..VISION_PIXELS {
                let net_resp =
                    ganglion_responses[p].on_response - ganglion_responses[p].off_response;
                sum = sum + net_resp * self.projection_weights[p][d];
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
    fn test_vision_spike_encoder_encode() {
        let mut encoder = VisionSpikeEncoder::new();
        encoder.init_random(123);

        let mut frame = [0u8; VISION_PIXELS];
        for i in 0..VISION_PIXELS {
            frame[i] = (i % 256) as u8;
        }

        let spikes = encoder.encode_frame(&frame, 200);
        assert_eq!(spikes.len(), TIME_STEPS);
        assert_eq!(spikes[0].len(), EMBED_DIM);
    }
}
