//! Native HKL Protocol & API Server Module (HKL-NP v1).
//! Provides binary packet framing, unified Cortex Service orchestration,
//! distributed multi-node Swarm cluster management, and multi-threaded TCP/HTTP server.
#![cfg(feature = "hkl2")]

pub mod cortex_service;
pub mod protocol;
pub mod server;
pub mod swarm_cluster;
