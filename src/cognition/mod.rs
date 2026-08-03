//! HKL-2 bounded cognition loops.
//!
//! These modules turn model telemetry into explicit, auditable decisions before
//! any higher-level agent loop is allowed to affect learning or action.
#![cfg(feature = "hkl2")]

pub mod audit;
pub mod controller;
pub mod episode;
pub mod executive;
pub mod metacognition;
pub mod planner;
pub mod readiness;
pub mod runtime_gate;
pub mod scenario;
pub mod supervisor;
