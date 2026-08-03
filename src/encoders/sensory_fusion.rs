//! Cross-Modal Spiking Sensory Fusion Module (`src/encoders/sensory_fusion.rs`).
//! Fuses Text (256D), Audio Cochlear (256D), and Vision Retinal (256D) spike streams
//! into a unified 512D Spatio-Temporal Cross-Modal Cortex Embedding.
#![cfg(feature = "hkl2")]

use crate::core::math::FixedPoint;
use crate::embedding::spike_embedding::LIFNeuronLight;
use alloc::vec::Vec;

pub const FUSED_DIM: usize = 512;
pub const MODALITY_DIM: usize = 256;

/// Fused Cross-Modal Sensory Spike Frame
#[derive(Debug, Clone)]
pub struct FusedSensoryFrame {
    pub fused_spikes: Vec<[bool; FUSED_DIM]>,
    pub total_active_spikes: usize,
    pub cross_modal_coincidence_score: f32,
}

/// Cross-Modal Spiking Sensory Fusion Engine
pub struct SensoryFusionEngine {
    pub fusion_neurons: Vec<LIFNeuronLight>,
    pub weights_text: Vec<FixedPoint>,
    pub weights_audio: Vec<FixedPoint>,
    pub weights_vision: Vec<FixedPoint>,
}

impl SensoryFusionEngine {
    pub fn new() -> Self {
        let mut fusion_neurons = Vec::with_capacity(FUSED_DIM);
        let mut weights_text = Vec::with_capacity(FUSED_DIM);
        let mut weights_audio = Vec::with_capacity(FUSED_DIM);
        let mut weights_vision = Vec::with_capacity(FUSED_DIM);

        let thresh = FixedPoint::from_f32(0.8);
        let decay = FixedPoint::from_f32(0.9);

        for i in 0..FUSED_DIM {
            let neuron = LIFNeuronLight::new(thresh, decay);
            fusion_neurons.push(neuron);

            let t_w = FixedPoint::from_f32(0.4 + (i % 10) as f32 * 0.05);
            let a_w = FixedPoint::from_f32(0.3 + (i % 8) as f32 * 0.05);
            let v_w = FixedPoint::from_f32(0.3 + (i % 6) as f32 * 0.05);

            weights_text.push(t_w);
            weights_audio.push(a_w);
            weights_vision.push(v_w);
        }

        Self {
            fusion_neurons,
            weights_text,
            weights_audio,
            weights_vision,
        }
    }

    /// Fuse text, audio, and vision spike streams over T timesteps
    pub fn fuse_modalities(
        &mut self,
        text_spikes: &[[bool; MODALITY_DIM]],
        audio_spikes: &[[bool; MODALITY_DIM]],
        vision_spikes: &[[bool; MODALITY_DIM]],
    ) -> FusedSensoryFrame {
        let t_len = text_spikes
            .len()
            .max(audio_spikes.len())
            .max(vision_spikes.len());
        let mut fused_stream = Vec::with_capacity(t_len);
        let mut total_spikes = 0usize;
        let mut coincidence_count = 0usize;

        for step in 0..t_len {
            let mut step_spikes = [false; FUSED_DIM];

            let t_frame = text_spikes.get(step);
            let a_frame = audio_spikes.get(step);
            let v_frame = vision_spikes.get(step);

            for i in 0..FUSED_DIM {
                let mod_idx = i % MODALITY_DIM;
                let t_active = t_frame.is_some_and(|f| f[mod_idx]);
                let a_active = a_frame.is_some_and(|f| f[mod_idx]);
                let v_active = v_frame.is_some_and(|f| f[mod_idx]);

                let mut current_in = FixedPoint::ZERO;
                if t_active {
                    current_in = current_in + self.weights_text[i];
                }
                if a_active {
                    current_in = current_in + self.weights_audio[i];
                }
                if v_active {
                    current_in = current_in + self.weights_vision[i];
                }

                if (t_active as u8 + a_active as u8 + v_active as u8) >= 2 {
                    coincidence_count += 1;
                }

                if self.fusion_neurons[i].step(current_in) {
                    step_spikes[i] = true;
                    total_spikes += 1;
                }
            }

            fused_stream.push(step_spikes);
        }

        let max_coincidence = (t_len * FUSED_DIM) as f32;
        let coincidence_score = if max_coincidence > 0.0 {
            coincidence_count as f32 / max_coincidence
        } else {
            0.0
        };

        FusedSensoryFrame {
            fused_spikes: fused_stream,
            total_active_spikes: total_spikes,
            cross_modal_coincidence_score: coincidence_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory_fusion_engine() {
        let mut engine = SensoryFusionEngine::new();
        let text = [[true; MODALITY_DIM]; 4];
        let audio = [[true; MODALITY_DIM]; 4];
        let vision = [[false; MODALITY_DIM]; 4];

        let fused = engine.fuse_modalities(&text, &audio, &vision);
        assert_eq!(fused.fused_spikes.len(), 4);
        assert!(fused.total_active_spikes > 0);
        assert!(fused.cross_modal_coincidence_score > 0.0);
    }
}
