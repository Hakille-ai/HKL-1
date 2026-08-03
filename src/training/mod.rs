//! Training pipeline module for HKL-2.
//! Manages datasets, batch loading, forward/loss computation, and e-prop weight updates.
#![cfg(feature = "hkl2")]

pub mod checkpoint;
pub mod corpus;
pub mod data_loader;
pub mod dataset;
pub mod media;
pub mod monitor;
pub mod run;
pub mod trainer;
