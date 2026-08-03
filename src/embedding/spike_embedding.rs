use crate::core::math::{FixedPoint, XorShift64Star};
use alloc::vec::Vec;

pub const EMBED_DIM: usize = 256;
pub const TIME_STEPS: usize = 4;
pub const VOCAB_SIZE: usize = 4096;

pub struct LIFNeuronLight {
    pub membrane: FixedPoint,
    pub threshold: FixedPoint,
    pub decay: FixedPoint,
}

impl LIFNeuronLight {
    pub fn new(threshold: FixedPoint, decay: FixedPoint) -> Self {
        Self {
            membrane: FixedPoint::ZERO,
            threshold,
            decay,
        }
    }

    pub fn step(&mut self, input: FixedPoint) -> bool {
        self.membrane = self.membrane * self.decay + input;
        if self.membrane >= self.threshold {
            self.membrane = FixedPoint::ZERO;
            true
        } else {
            false
        }
    }
}

pub struct SpikeEmbeddingLayer {
    pub weights: Vec<[FixedPoint; EMBED_DIM]>,
    pub neurons: [LIFNeuronLight; EMBED_DIM],
}

impl SpikeEmbeddingLayer {
    pub fn new() -> Self {
        let neurons = core::array::from_fn(|_| {
            LIFNeuronLight::new(FixedPoint::from_f32(0.15), FixedPoint::from_f32(0.9))
        });
        Self {
            weights: alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; VOCAB_SIZE],
            neurons,
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);
        for i in 0..VOCAB_SIZE {
            for j in 0..EMBED_DIM {
                let rand_val = rng.next_f32() * 0.2 - 0.1;
                self.weights[i][j] = FixedPoint::from_f32(rand_val);
            }
        }
        self.reset_state();
    }

    pub fn reset_state(&mut self) {
        for n in self.neurons.iter_mut() {
            n.membrane = FixedPoint::ZERO;
        }
    }

    pub fn encode(&mut self, token_id: u16) -> [[bool; EMBED_DIM]; TIME_STEPS] {
        self.reset_state();

        let mut output = [[false; EMBED_DIM]; TIME_STEPS];
        let token_id = token_id as usize;
        if token_id >= VOCAB_SIZE {
            return output;
        }

        let current = self.weights[token_id];
        for t in 0..TIME_STEPS {
            for i in 0..EMBED_DIM {
                output[t][i] = self.neurons[i].step(current[i]);
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron_light_spikes() {
        let mut neuron = LIFNeuronLight::new(FixedPoint::from_f32(1.0), FixedPoint::from_f32(0.9));
        assert!(!neuron.step(FixedPoint::from_f32(0.5)));
        assert_eq!(neuron.membrane, FixedPoint::from_f32(0.5));

        assert!(neuron.step(FixedPoint::from_f32(0.6)));
        assert_eq!(neuron.membrane, FixedPoint::ZERO); // reset
    }

    #[test]
    fn test_lif_neuron_light_decay() {
        let mut neuron = LIFNeuronLight::new(FixedPoint::from_f32(1.0), FixedPoint::from_f32(0.5));
        neuron.step(FixedPoint::from_f32(0.8));
        assert_eq!(neuron.membrane, FixedPoint::from_f32(0.8));

        neuron.step(FixedPoint::from_f32(0.0));
        assert_eq!(neuron.membrane, FixedPoint::from_f32(0.4)); // Decayed by 0.5
    }

    #[test]
    fn test_spike_embedding_layer() {
        let mut layer = SpikeEmbeddingLayer::new();
        layer.init_random(42);

        let pattern1 = layer.encode(10);
        let pattern2 = layer.encode(20);

        // Due to randomness and 4 timesteps, the patterns should be different
        assert_ne!(pattern1, pattern2);

        // Encoding the same token again should yield the same pattern (since neurons reset)
        let pattern1_again = layer.encode(10);
        assert_eq!(pattern1, pattern1_again);
    }

    #[test]
    fn spike_embedding_out_of_vocab_is_silent_and_resets_state() {
        let mut layer = SpikeEmbeddingLayer::new();
        layer.init_random(42);
        layer.neurons[0].membrane = FixedPoint::ONE;

        let pattern = layer.encode(VOCAB_SIZE as u16);

        assert!(pattern.iter().flatten().all(|spike| !*spike));
        assert!(layer.neurons.iter().all(|n| n.membrane == FixedPoint::ZERO));
    }

    #[test]
    fn test_spike_embedding_reset_state_clears_membranes() {
        let mut layer = SpikeEmbeddingLayer::new();
        layer.neurons[0].membrane = FixedPoint::ONE;
        layer.neurons[EMBED_DIM - 1].membrane = FixedPoint::from_f32(0.5);

        layer.reset_state();

        assert!(layer.neurons.iter().all(|n| n.membrane == FixedPoint::ZERO));
    }
}
