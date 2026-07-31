//! Unified NLP & Dialogue Engine for HKL-1.
//! Orchestrates spike text encoding, neuro-symbolic knowledge graph reasoning,
//! neuromodulated state verbalization, and text generation.

use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::nlp::spike_decoder::SpikeTextDecoder;
use crate::nlp::spike_token::SpikeTextEncoder;
use crate::nlp::symbolic_graph::SymbolicKnowledgeGraph;
use crate::nlp::verbalizer::{CognitiveStateSummary, MAX_VERBAL_LEN, NeuromodulatedVerbalizer};

/// Unified NLP Dialogue Engine
pub struct DialogueEngine {
    pub encoder: SpikeTextEncoder,
    pub decoder: SpikeTextDecoder,
    pub knowledge_graph: SymbolicKnowledgeGraph,
    pub last_verbalization: [u8; MAX_VERBAL_LEN],
    pub verbal_len: usize,
}

impl DialogueEngine {
    pub fn new() -> Self {
        Self {
            encoder: SpikeTextEncoder::new(NeuronId::new(0)),
            decoder: SpikeTextDecoder::new(),
            knowledge_graph: SymbolicKnowledgeGraph::new(),
            last_verbalization: [0; MAX_VERBAL_LEN],
            verbal_len: 0,
        }
    }

    /// Process user natural language prompt: encodes spikes, activates symbolic concepts, and generates response
    pub fn process_user_prompt(
        &mut self,
        prompt: &[u8],
        timestamp: u32,
        cognitive_state: &CognitiveStateSummary,
    ) -> ([u8; MAX_VERBAL_LEN], usize) {
        // 1. Encode prompt into temporal spike trains
        self.encoder.encode_text(prompt, timestamp);

        // 2. Synthesize internal state verbalization
        let (verbal_buf, v_len) = NeuromodulatedVerbalizer::verbalize_state(cognitive_state);
        self.last_verbalization = verbal_buf;
        self.verbal_len = v_len;

        // 3. Neuro-Symbolic Activation: add prompt words as concepts & propagate activation
        let concept_id = self.knowledge_graph.add_concept(prompt);
        self.knowledge_graph
            .activate_and_propagate(concept_id, FixedPoint::ONE);

        // 4. Simulate Layer 4 motor potentials decoding
        let mut motor_potentials = [FixedPoint::ZERO; crate::nlp::spike_token::VOCAB_SIZE];
        // Populate motor response based on top active concept
        if self.knowledge_graph.concepts[concept_id as usize].valid {
            motor_potentials[b'O' as usize] = FixedPoint::from_f32(0.8);
            motor_potentials[b'K' as usize] = FixedPoint::from_f32(0.9);
        }
        self.decoder.clear_buffer();
        self.decoder.decode_firing_rates(&motor_potentials);

        (verbal_buf, v_len)
    }
}
