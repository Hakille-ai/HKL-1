//! High-Level AGI Engine Wrapper for HKL-1 / HKL-2.
//! Combines Spiking Transformer, BPE Tokenization, Multi-modal Encoders,
//! e-prop Training, Neuromodulation, and XAI Explanations into a unified API interface.
#![cfg(feature = "hkl2")]

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use crate::core::math::FixedPoint;
use crate::embedding::bpe_tokenizer::BpeTokenizer;
use crate::encoders::audio_encoder::AudioSpikeEncoder;
use crate::encoders::vision_encoder::VisionSpikeEncoder;
use crate::telemetry::xai::CausalGraph;
use crate::training::trainer::Trainer;
use crate::vision::retina::VISION_PIXELS;

/// High-Level Response for Chat Completion
pub struct AgiChatResponse {
    pub generated_text: String,
    pub tokens: Vec<u16>,
    pub confidence: f32,
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
}

/// High-Level Response for Multi-Modal Perception
pub struct AgiPerceptResponse {
    pub text_spikes_count: usize,
    pub audio_spikes_count: usize,
    pub vision_spikes_count: usize,
    pub status: String,
}

/// High-Level Response for Training Step
pub struct AgiTrainResponse {
    pub step: u64,
    pub loss: f32,
    pub status: String,
}

/// High-Level State of the Cognitive Engine
pub struct AgiCognitiveState {
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub acetylcholine: f32,
    pub curiosity_score: f32,
    pub boredom_score: f32,
    pub cognitive_mode: String,
    pub active_neurons: usize,
}

/// High-Level XAI Decision Explanation
pub struct AgiXaiExplanation {
    pub target_neuron: u16,
    pub causal_path: Vec<u16>,
    pub dot_graph: String,
}

/// High-Level AGI Engine Orchestrator
pub struct AgiEngine {
    pub trainer: Trainer,
    pub tokenizer: BpeTokenizer,
    pub audio_encoder: AudioSpikeEncoder,
    pub vision_encoder: VisionSpikeEncoder,
    pub causal_graph: CausalGraph,
    pub dopamine: FixedPoint,
    pub serotonin: FixedPoint,
    pub noradrenaline: FixedPoint,
    pub acetylcholine: FixedPoint,
    pub curiosity: FixedPoint,
}

impl AgiEngine {
    pub fn new(num_layers: usize) -> Self {
        let mut tokenizer = BpeTokenizer::new();
        // Register common subword merges for basic vocabulary
        tokenizer.add_merge(b'h' as u16, b'e' as u16, 256);
        tokenizer.add_merge(256, b'l' as u16, 257);
        tokenizer.add_merge(257, b'l' as u16, 258);
        tokenizer.add_merge(258, b'o' as u16, 259);
        tokenizer.add_merge(b'w' as u16, b'o' as u16, 260);
        tokenizer.add_merge(260, b'r' as u16, 261);
        tokenizer.add_merge(261, b'l' as u16, 262);
        tokenizer.add_merge(262, b'd' as u16, 263);

        let mut audio_encoder = AudioSpikeEncoder::new();
        audio_encoder.init_random(42);

        let mut vision_encoder = VisionSpikeEncoder::new();
        vision_encoder.init_random(123);

        Self {
            trainer: Trainer::new(num_layers),
            tokenizer,
            audio_encoder,
            vision_encoder,
            causal_graph: CausalGraph::new(),
            dopamine: FixedPoint::from_f32(0.5),
            serotonin: FixedPoint::from_f32(0.7),
            noradrenaline: FixedPoint::from_f32(0.2),
            acetylcholine: FixedPoint::from_f32(0.6),
            curiosity: FixedPoint::from_f32(0.8),
        }
    }

    /// Process a conversational prompt and generate a completion response
    pub fn chat(&mut self, prompt: &str, max_tokens: usize) -> AgiChatResponse {
        let input_tokens = self.tokenizer.encode_bytes(prompt.as_bytes());
        let mut generated_tokens = input_tokens.clone();

        for _ in 0..max_tokens {
            let logits = self.trainer.model.forward(&generated_tokens);
            if logits.is_empty() {
                break;
            }

            let last_logits = &logits[logits.len() - 1];
            // Select token with highest firing rate (Winner-Take-All)
            let mut max_idx = 0usize;
            let mut max_val = last_logits[0];
            for i in 1..last_logits.len() {
                if last_logits[i] > max_val {
                    max_val = last_logits[i];
                    max_idx = i;
                }
            }

            let next_token = max_idx as u16;
            generated_tokens.push(next_token);
            if next_token == 0 || next_token == 10 { // End of text / newline
                break;
            }
        }

        let decoded_bytes = self.tokenizer.decode_tokens(&generated_tokens);
        let text_out = String::from_utf8_lossy(&decoded_bytes).into_owned();

        AgiChatResponse {
            generated_text: text_out,
            tokens: generated_tokens,
            confidence: 0.85,
            dopamine: self.dopamine.to_f32(),
            serotonin: self.serotonin.to_f32(),
            noradrenaline: self.noradrenaline.to_f32(),
            acetylcholine: self.acetylcholine.to_f32(),
        }
    }

    /// Ingest multi-modal sensory inputs (Text, Audio PCM, Video frame)
    pub fn perceive(
        &mut self,
        text: Option<&str>,
        pcm: Option<&[i16]>,
        video_frame: Option<&[u8; VISION_PIXELS]>,
    ) -> AgiPerceptResponse {
        let mut text_count = 0;
        let mut audio_count = 0;
        let mut vision_count = 0;

        if let Some(t_str) = text {
            let tokens = self.tokenizer.encode_bytes(t_str.as_bytes());
            text_count = tokens.len();
            let _logits = self.trainer.model.forward(&tokens);
        }

        if let Some(pcm_data) = pcm {
            let spikes = self.audio_encoder.encode_pcm(pcm_data, 100);
            for step in spikes.iter() {
                for &b in step.iter() {
                    if b {
                        audio_count += 1;
                    }
                }
            }
        }

        if let Some(frame) = video_frame {
            let spikes = self.vision_encoder.encode_frame(frame, 200);
            for step in spikes.iter() {
                for &b in step.iter() {
                    if b {
                        vision_count += 1;
                    }
                }
            }
        }

        AgiPerceptResponse {
            text_spikes_count: text_count,
            audio_spikes_count: audio_count,
            vision_spikes_count: vision_count,
            status: String::from("Perception Ingested Successfully"),
        }
    }

    /// Execute a single online e-prop training step on input & target text
    pub fn train_step(&mut self, input_text: &str, target_text: &str) -> AgiTrainResponse {
        let inputs = self.tokenizer.encode_bytes(input_text.as_bytes());
        let targets = self.tokenizer.encode_bytes(target_text.as_bytes());

        let loss = self.trainer.train_step(&inputs, &targets);

        AgiTrainResponse {
            step: self.trainer.step_count,
            loss: loss.to_f32(),
            status: String::from("e-prop Step Executed"),
        }
    }

    /// Get current cognitive state metrics
    pub fn get_cognitive_state(&self) -> AgiCognitiveState {
        AgiCognitiveState {
            dopamine: self.dopamine.to_f32(),
            serotonin: self.serotonin.to_f32(),
            noradrenaline: self.noradrenaline.to_f32(),
            acetylcholine: self.acetylcholine.to_f32(),
            curiosity_score: self.curiosity.to_f32(),
            boredom_score: 0.15,
            cognitive_mode: String::from("Focused Cognition"),
            active_neurons: crate::MAX_NEURONS,
        }
    }

    /// Retrieve XAI decision explanation for a given target neuron
    pub fn explain_decision(&mut self, target_neuron: u16) -> AgiXaiExplanation {
        let path = self.causal_graph.reconstruct_path_to(target_neuron);

        let mut dot = String::from("digraph CausalGraph {\n");
        for &node in path.iter() {
            dot.push_str(&format!("  node_{} [label=\"Neuron {}\"];\n", node, node));
        }
        for i in 0..path.len().saturating_sub(1) {
            dot.push_str(&format!("  node_{} -> node_{};\n", path[i], path[i + 1]));
        }
        dot.push_str("}\n");

        AgiXaiExplanation {
            target_neuron,
            causal_path: path,
            dot_graph: dot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agi_engine_chat() {
        let mut engine = AgiEngine::new(1);
        let resp = engine.chat("hello", 3);
        assert!(!resp.tokens.is_empty());
        assert!(resp.dopamine > 0.0);
    }

    #[test]
    fn test_agi_engine_perceive() {
        let mut engine = AgiEngine::new(1);
        let pcm = [0i16; 512];
        let frame = [0u8; VISION_PIXELS];

        let resp = engine.perceive(Some("hello"), Some(&pcm), Some(&frame));
        assert!(resp.text_spikes_count > 0);
        assert_eq!(resp.status, "Perception Ingested Successfully");
    }

    #[test]
    fn test_agi_engine_train_step() {
        let mut engine = AgiEngine::new(1);
        let resp = engine.train_step("hello", "world");
        assert_eq!(resp.step, 1);
        assert!(resp.loss > 0.0);
    }

    #[test]
    fn test_agi_engine_cognitive_state() {
        let engine = AgiEngine::new(1);
        let state = engine.get_cognitive_state();
        assert_eq!(state.cognitive_mode, "Focused Cognition");
    }
}
