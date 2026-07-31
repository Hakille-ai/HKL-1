//! HKL-1: A neuromorphic AI system for embedded devices.
//! Integrates spiking neural networks, cognitive functions, I/O,
//! system management, safety reflexes, swarm communication, and telemetry.
#![no_std]
#![deny(clippy::correctness, clippy::suspicious, clippy::perf)]
#![warn(clippy::all)]
#![allow(static_mut_refs)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::needless_return)]
#![allow(clippy::unnecessary_cast)]
#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod audio;
pub mod bio;
pub mod bsp;
pub mod cognitive;
pub mod core;
pub mod efpga;
pub mod io;
pub mod nlp;
pub mod safety;
pub mod snn;
pub mod swarm;
pub mod system;
pub mod telemetry;
pub mod vision;

/// HKL-1 Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Global constants
pub const MAX_NEURONS: usize = 4096;
pub const MAX_SYNAPSES: usize = 65536;
pub const RING_BUFFER_SIZE: usize = 4096;
pub const SPIKE_TRACE_BUFFER: usize = 8192;
pub const PERSISTENCE_SLOTS: usize = 3;
