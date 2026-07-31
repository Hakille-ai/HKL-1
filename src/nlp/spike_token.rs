//! Spike Tokenizer & Text Encoder for HKL-1.
//! Converts ASCII & subword text tokens into spatio-temporal spike trains
//! with phase-timing position encoding (delta t_pos = pos * tau_phase).

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::io::buffers::{EncodedSpike, Modality, ingest_spike};

pub const VOCAB_SIZE: usize = 256;
pub const MAX_TOKEN_LEN: usize = 32;

/// Spiking Vocabulary Entry
#[derive(Clone, Copy)]
pub struct VocabToken {
    pub id: u16,
    pub symbol: [u8; 16],
    pub len: u8,
    pub neuron_offset: u16,
}

impl VocabToken {
    pub const fn empty() -> Self {
        Self {
            id: 0,
            symbol: [0; 16],
            len: 0,
            neuron_offset: 0,
        }
    }
}

/// Spike Tokenizer & Vocabulary System
pub struct SpikeVocabulary {
    pub tokens: [VocabToken; VOCAB_SIZE],
}

impl SpikeVocabulary {
    pub fn new() -> Self {
        let mut tokens = [VocabToken::empty(); VOCAB_SIZE];
        // Populate ASCII characters 0..127
        for c in 0..128u16 {
            let mut sym = [0u8; 16];
            sym[0] = c as u8;
            tokens[c as usize] = VocabToken {
                id: c,
                symbol: sym,
                len: 1,
                neuron_offset: c,
            };
        }
        Self { tokens }
    }
}

/// Spiking Text Encoder with temporal phase-timing position encoding
pub struct SpikeTextEncoder {
    pub vocab: SpikeVocabulary,
    pub base_neuron_id: NeuronId,
    pub phase_timing_ms: u32, // Delay per position index
    pub event_count: u32,
}

impl SpikeTextEncoder {
    pub fn new(base_neuron_id: NeuronId) -> Self {
        Self {
            vocab: SpikeVocabulary::new(),
            base_neuron_id,
            phase_timing_ms: 5, // 5ms phase delay per token position
            event_count: 0,
        }
    }

    /// Encode input text string into temporal spike trains with position phase delays
    pub fn encode_text(&mut self, text: &[u8], timestamp: u32) -> u32 {
        self.event_count = 0;
        let mut pos = 0u32;

        for &c in text {
            if c == 0 {
                break;
            }
            let token_idx = (c & 0x7F) as u16;
            let neuron_id = NeuronId::new(self.base_neuron_id.index() as u16 + token_idx);

            // Temporal Phase Position Encoding: delta_t_pos = pos * phase_timing_ms
            let token_timestamp = timestamp + pos * self.phase_timing_ms;

            let spike = EncodedSpike {
                neuron_id,
                intensity: FixedPoint::ONE,
                timestamp: token_timestamp,
                modality: Modality::Text,
            };

            if ingest_spike(spike) {
                self.event_count += 1;
            }
            pos += 1;
        }

        self.event_count
    }
}
