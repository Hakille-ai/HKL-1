//! Spiking Transformer backbone for HKL-2.
//! Implements Spiking Self-Attention (SSA) inspired by Spikformer,
//! Spiking Feed-Forward networks, and full transformer blocks.
#![cfg(feature = "hkl2")]

pub mod attention;
pub mod backbone;
pub mod block;
pub mod feed_forward;
pub mod norm;
