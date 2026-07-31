//! Spike Token Decoder & Motor Output Reconstruction for HKL-1.
//! Integrates Layer 4 text/motor neuron firing rates, performs Winner-Take-All (WTA) token selection,
//! and reconstructs fluent natural language sentences.

use crate::core::math::FixedPoint;
use crate::nlp::spike_token::VOCAB_SIZE;

pub const MAX_RESPONSE_LEN: usize = 128;

/// Firing Rate Accumulator per token slot
#[derive(Clone, Copy)]
pub struct TokenFiringRate {
    pub token_id: u8,
    pub firing_potential: FixedPoint,
}

/// Winner-Take-All Spike Token Decoder
pub struct SpikeTextDecoder {
    pub integration_threshold: FixedPoint,
    pub output_buffer: [u8; MAX_RESPONSE_LEN],
    pub buffer_len: usize,
}

impl SpikeTextDecoder {
    pub fn new() -> Self {
        Self {
            integration_threshold: FixedPoint::from_f32(0.2),
            output_buffer: [0; MAX_RESPONSE_LEN],
            buffer_len: 0,
        }
    }

    /// Decode Layer 4 output motor/text firing potentials into text characters/tokens
    pub fn decode_firing_rates(&mut self, motor_potentials: &[FixedPoint; VOCAB_SIZE]) -> Option<u8> {
        let mut max_val = FixedPoint::ZERO;
        let mut best_token = 0u8;

        for i in 0..VOCAB_SIZE {
            if motor_potentials[i] > max_val {
                max_val = motor_potentials[i];
                best_token = i as u8;
            }
        }

        if max_val >= self.integration_threshold {
            if self.buffer_len < MAX_RESPONSE_LEN {
                self.output_buffer[self.buffer_len] = best_token;
                self.buffer_len += 1;
            }
            Some(best_token)
        } else {
            None
        }
    }

    /// Clear response buffer
    pub fn clear_buffer(&mut self) {
        self.output_buffer = [0; MAX_RESPONSE_LEN];
        self.buffer_len = 0;
    }

    /// Get current decoded response slice
    pub fn get_response_text(&self) -> &[u8] {
        &self.output_buffer[..self.buffer_len]
    }
}
