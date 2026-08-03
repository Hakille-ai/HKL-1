//! Full Spiking Transformer Backbone for HKL-2.

use crate::core::math::{FixedPoint, XorShift64Star};
use crate::embedding::spike_embedding::{EMBED_DIM, SpikeEmbeddingLayer, VOCAB_SIZE};
use crate::transformer::attention::MAX_SEQ_LEN;
use crate::transformer::block::SpikingTransformerBlock;
use alloc::vec;
use alloc::vec::Vec;

/// Output Projection Head (EMBED_DIM -> VOCAB_SIZE)
pub struct OutputProjection {
    pub weights: Vec<FixedPoint>, // VOCAB_SIZE * EMBED_DIM
    pub bias: Vec<FixedPoint>,    // VOCAB_SIZE
}

impl OutputProjection {
    pub fn new() -> Self {
        Self {
            weights: vec![FixedPoint::ZERO; VOCAB_SIZE * EMBED_DIM],
            bias: vec![FixedPoint::ZERO; VOCAB_SIZE],
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        let mut rng = XorShift64Star::new(seed);
        let k_f32 = 1.0 / (EMBED_DIM as f32).sqrt();
        for v in self.weights.iter_mut() {
            let rand_val = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 * k_f32 - k_f32;
            *v = FixedPoint::from_f32(rand_val);
        }
    }

    pub fn forward(&self, input: &[FixedPoint; EMBED_DIM], output: &mut [FixedPoint; VOCAB_SIZE]) {
        for v in 0..VOCAB_SIZE {
            let mut sum = self.bias[v];
            let offset = v * EMBED_DIM;
            for d in 0..EMBED_DIM {
                sum = sum + self.weights[offset + d] * input[d];
            }
            output[v] = sum.relu() + FixedPoint::from_f32(0.001);
        }
    }
}

/// Full Spiking Transformer Model
pub struct SpikingTransformer {
    pub embedding: SpikeEmbeddingLayer,
    pub blocks: Vec<SpikingTransformerBlock>,
    pub head: OutputProjection,
}

impl SpikingTransformer {
    pub fn new(num_layers: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(SpikingTransformerBlock::new());
        }
        Self {
            embedding: SpikeEmbeddingLayer::new(),
            blocks,
            head: OutputProjection::new(),
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        self.embedding.init_random(seed);
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.init_random(seed + 100 + i as u64 * 10);
        }
        self.head.init_random(seed + 999);
        self.reset_state();
    }

    pub fn reset_state(&mut self) {
        self.embedding.reset_state();
        for block in self.blocks.iter_mut() {
            block.reset_state();
        }
    }

    /// Forward pass over sequence of token IDs
    /// Returns firing rates/logits for each token position: [seq_len][VOCAB_SIZE]
    pub fn forward(&mut self, tokens: &[u16]) -> Vec<[FixedPoint; VOCAB_SIZE]> {
        let active_len = tokens.len().min(MAX_SEQ_LEN);
        if active_len == 0 {
            return Vec::new();
        }

        // 1. Encode tokens to spike embeddings [seq_len][EMBED_DIM]
        // Average temporal spikes over TIME_STEPS to get continuous rate embedding per position
        let mut continuous_seq = vec![[FixedPoint::ZERO; EMBED_DIM]; active_len];
        for (pos, &token) in tokens.iter().take(active_len).enumerate() {
            if token as usize >= VOCAB_SIZE {
                continue;
            }
            let spike_matrix = self.embedding.encode(token);
            for d in 0..EMBED_DIM {
                let mut spike_count = 0;
                for t in 0..crate::embedding::spike_embedding::TIME_STEPS {
                    if spike_matrix[t][d] {
                        spike_count += 1;
                    }
                }
                continuous_seq[pos][d] = FixedPoint::from_f32(
                    spike_count as f32 / crate::embedding::spike_embedding::TIME_STEPS as f32,
                );
            }
        }

        // 2. Pass through transformer blocks
        let mut x = continuous_seq;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, active_len);
        }

        // 3. Project to output vocabulary logits
        let mut logits = vec![[FixedPoint::ZERO; VOCAB_SIZE]; active_len];
        for pos in 0..active_len {
            self.head.forward(&x[pos], &mut logits[pos]);
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_transformer_pipeline() {
        let mut model = SpikingTransformer::new(2);
        model.init_random(42);
        let tokens = [10u16, 25u16, 100u16];
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), VOCAB_SIZE);
    }

    #[test]
    fn test_spiking_transformer_clamps_oversized_sequence() {
        let mut model = SpikingTransformer::new(1);
        let tokens = alloc::vec![1u16; MAX_SEQ_LEN + 5];

        let logits = model.forward(&tokens);

        assert_eq!(logits.len(), MAX_SEQ_LEN);
    }

    #[test]
    fn test_spiking_transformer_accepts_out_of_vocab_as_silent_input() {
        let mut model = SpikingTransformer::new(1);
        let tokens = [VOCAB_SIZE as u16];

        let logits = model.forward(&tokens);

        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), VOCAB_SIZE);
    }

    #[test]
    fn test_spiking_transformer_reset_state_propagates_to_layers() {
        let mut model = SpikingTransformer::new(1);
        model.embedding.neurons[0].membrane = FixedPoint::ONE;
        model.blocks[0].attention.q_membranes[0] = FixedPoint::ONE;
        model.blocks[0].feed_forward.sn1_membranes[0] = FixedPoint::ONE;

        model.reset_state();

        assert_eq!(model.embedding.neurons[0].membrane, FixedPoint::ZERO);
        assert_eq!(model.blocks[0].attention.q_membranes[0], FixedPoint::ZERO);
        assert_eq!(
            model.blocks[0].feed_forward.sn1_membranes[0],
            FixedPoint::ZERO
        );
    }
}
