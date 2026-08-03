//! E-prop learning engine for HKL-2 (Phase 1)
//! Adds gradient-based learning capability to the existing SNN.

#![cfg(feature = "hkl2")]

pub mod eprop;
pub mod loss;
pub mod surrogate;
