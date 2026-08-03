//! Native HKL Protocol Server Example (`hkl_native_server.rs`).
//! Demonstrates initializing a Swarm Cluster Manager, binding the HklNativeServer,
//! and sending binary packet requests over TCP sockets.

#[cfg(feature = "hkl2")]
use hkl1::api::protocol::{HklBinaryPacket, HklCommand};
#[cfg(feature = "hkl2")]
use hkl1::api::server::HklNativeServer;
#[cfg(feature = "hkl2")]
use hkl1::api::swarm_cluster::SwarmClusterManager;
#[cfg(feature = "hkl2")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "hkl2")]
fn main() {
    println!("=== 🚀 HKL Native Protocol Multi-Node Swarm Server Demo ===");

    let head_id = [1, 2, 3, 4, 5, 6, 7, 8];
    let peer_id = [9, 10, 11, 12, 13, 14, 15, 16];

    // 1. Initialize Distributed Multi-Node Swarm Cluster
    println!("\n[1] Initializing Swarm Cluster Manager ('AlphaSwarm')...");
    let mut cluster = SwarmClusterManager::new(head_id, "AlphaSwarm", 2);
    cluster.register_peer_node(peer_id, 1, -50);

    let consensus = cluster.propose_cluster_consensus(101, 1, 42, 1000);
    println!("   Cluster Nodes: {}", cluster.cluster_nodes.len());
    println!(
        "   Consensus Proposal 101: Passed = {}, VotesFor = {}",
        consensus.passed, consensus.votes_for
    );

    // 2. Direct Binary Packet Execution Test
    println!("\n[2] Executing HKL Native Protocol Binary Packets...");
    let service_arc = Arc::new(Mutex::new(cluster.local_service));

    // Command 1: Perceive Frame
    let perceive_req =
        HklBinaryPacket::new(HklCommand::PerceiveFrame, 1000, b"hello world".to_vec());
    let perceive_resp = HklNativeServer::handle_packet(&service_arc, &perceive_req);
    println!(
        "   [0x0001 Perceive] Response: {}",
        String::from_utf8_lossy(&perceive_resp.payload)
    );

    // Command 2: Synthesize Response
    let synth_req = HklBinaryPacket::new(HklCommand::SynthesizeResponse, 1001, b"hello".to_vec());
    let synth_resp = HklNativeServer::handle_packet(&service_arc, &synth_req);
    println!(
        "   [0x0002 Synthesize] Response: {}",
        String::from_utf8_lossy(&synth_resp.payload)
    );

    // Command 3: e-prop Online Training
    let eprop_req =
        HklBinaryPacket::new(HklCommand::EpropTrainStep, 1002, b"hello->world".to_vec());
    let eprop_resp = HklNativeServer::handle_packet(&service_arc, &eprop_req);
    println!(
        "   [0x0003 e-prop Train] Response: {}",
        String::from_utf8_lossy(&eprop_resp.payload)
    );

    // Command 4: Cognitive State Telemetry
    let state_req = HklBinaryPacket::new(HklCommand::CognitiveState, 1003, vec![]);
    let state_resp = HklNativeServer::handle_packet(&service_arc, &state_req);
    println!(
        "   [0x0004 Cognitive State] Response: {}",
        String::from_utf8_lossy(&state_resp.payload)
    );

    // Command 5: XAI Causal Tree
    let xai_req = HklBinaryPacket::new(HklCommand::XaiCausalTree, 1004, vec![]);
    let xai_resp = HklNativeServer::handle_packet(&service_arc, &xai_req);
    println!(
        "   [0x0005 XAI Causal Tree] Response: {}",
        String::from_utf8_lossy(&xai_resp.payload)
    );

    // Command 6: Silicon eFPGA Compilation
    let silicon_req = HklBinaryPacket::new(HklCommand::SiliconCompile, 1005, vec![]);
    let silicon_resp = HklNativeServer::handle_packet(&service_arc, &silicon_req);
    println!(
        "   [0x0006 Silicon Compile] Response: {}",
        String::from_utf8_lossy(&silicon_resp.payload)
    );

    // Command 7: Swarm Mesh Topology
    let swarm_req = HklBinaryPacket::new(HklCommand::SwarmMeshStatus, 1006, vec![]);
    let swarm_resp = HklNativeServer::handle_packet(&service_arc, &swarm_req);
    println!(
        "   [0x0007 Swarm Mesh] Response: {}",
        String::from_utf8_lossy(&swarm_resp.payload)
    );

    println!("\n=== ✅ HKL Native Protocol Server Verification Complete ===");
}

#[cfg(not(feature = "hkl2"))]
fn main() {
    println!("Run with --features hkl2 to execute the native protocol server demo.");
}
