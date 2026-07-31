//! Core primitives: math utilities, memory management, time keeping,
//! entropy generation, cryptographic operations, and cross-platform atomics.
pub mod atomic;
pub mod crypto;
pub mod entropy;
pub mod math;
pub mod memory;
pub mod text;
pub mod time;

pub use atomic::FetchAtomic;
