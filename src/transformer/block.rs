use crate::core::math::FixedPoint;
use crate::embedding::spike_embedding::EMBED_DIM;
use crate::transformer::attention::{MAX_SEQ_LEN, SpikingSelfAttention};
use crate::transformer::feed_forward::SpikingFeedForward;
use crate::transformer::norm::LayerNorm;
use alloc::vec::Vec;

pub struct SpikingTransformerBlock {
    pub attention: SpikingSelfAttention,
    pub feed_forward: SpikingFeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl SpikingTransformerBlock {
    pub fn new() -> Self {
        Self {
            attention: SpikingSelfAttention::new(),
            feed_forward: SpikingFeedForward::new(),
            norm1: LayerNorm::new(EMBED_DIM),
            norm2: LayerNorm::new(EMBED_DIM),
        }
    }

    pub fn init_random(&mut self, seed: u64) {
        self.attention.init_random(seed);
        self.feed_forward.init_random(seed + 1);
        self.reset_state();
    }

    pub fn reset_state(&mut self) {
        self.attention.reset_state();
        self.feed_forward.reset_state();
    }

    pub fn forward(
        &mut self,
        input: &[[FixedPoint; EMBED_DIM]],
        seq_len: usize,
    ) -> Vec<[FixedPoint; EMBED_DIM]> {
        let active_len = seq_len.min(input.len()).min(MAX_SEQ_LEN);
        let mut x = input[..active_len].to_vec();

        let attn_out = self.attention.forward(&x, active_len);

        for i in 0..active_len {
            for j in 0..EMBED_DIM {
                x[i][j] = x[i][j] + attn_out[i][j];
            }
            self.norm1.forward(&mut x[i]);
        }

        for i in 0..active_len {
            let ffn_out = self.feed_forward.forward(&x[i]);
            for j in 0..EMBED_DIM {
                x[i][j] = x[i][j] + ffn_out[j];
            }
            self.norm2.forward(&mut x[i]);
        }

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block() {
        let mut block = SpikingTransformerBlock::new();
        block.init_random(123);
        let seq = alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; 2];
        let out = block.forward(&seq, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_transformer_block_clamps_to_input_len() {
        let mut block = SpikingTransformerBlock::new();
        let seq = alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; 1];
        let out = block.forward(&seq, 8);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_transformer_block_clamps_to_max_seq_len() {
        let mut block = SpikingTransformerBlock::new();
        let seq = alloc::vec![[FixedPoint::ZERO; EMBED_DIM]; MAX_SEQ_LEN + 4];
        let out = block.forward(&seq, MAX_SEQ_LEN + 4);
        assert_eq!(out.len(), MAX_SEQ_LEN);
    }

    #[test]
    fn test_transformer_block_reset_state_propagates() {
        let mut block = SpikingTransformerBlock::new();
        block.attention.q_membranes[0] = FixedPoint::ONE;
        block.feed_forward.sn1_membranes[0] = FixedPoint::ONE;

        block.reset_state();

        assert_eq!(block.attention.q_membranes[0], FixedPoint::ZERO);
        assert_eq!(block.feed_forward.sn1_membranes[0], FixedPoint::ZERO);
    }
}
