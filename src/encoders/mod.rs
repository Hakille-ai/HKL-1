//! Multi-modal sensory encoders for HKL-2.
//! Maps raw sensory signals (PCM audio, DVS/Retinal video frames) directly into
//! 256-dimensional spatio-temporal spike embeddings for the Spiking Transformer.
#![cfg(feature = "hkl2")]

pub mod audio_encoder;
pub mod sensory_fusion;
pub mod vision_encoder;
