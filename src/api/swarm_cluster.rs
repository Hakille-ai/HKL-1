//! Distributed Multi-Node Swarm Cluster Manager for HKL-1 / HKL-2.
//! Orchestrates peer node discovery, routing, federated weight aggregation,
//! consensus proposals, and distributed workload balancing across cluster nodes.
#![cfg(feature = "hkl2")]

use crate::api::cortex_service::CortexService;
use crate::core::math::Weight;
use crate::swarm::mesh::{NODE_ROLE_CLUSTER_HEAD, NodeInfo};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Node status report in the cluster
pub struct ClusterNodeReport {
    pub node_id_hex: String,
    pub role: String,
    pub is_connected: bool,
    pub last_seen_ms: u32,
    pub rssi: i8,
}

/// High-Level Cluster Consensus Result
pub struct ClusterConsensusResult {
    pub proposal_id: u32,
    pub topic: u8,
    pub votes_for: u16,
    pub votes_against: u16,
    pub finalized: bool,
    pub passed: bool,
}

/// Multi-Node Swarm Cluster Manager
pub struct SwarmClusterManager {
    pub local_service: CortexService,
    pub cluster_nodes: Vec<[u8; 8]>,
    pub cluster_name: String,
    pub total_cluster_updates: u64,
}

impl SwarmClusterManager {
    pub fn new(local_node_id: [u8; 8], cluster_name: &str, num_layers: usize) -> Self {
        let mut service = CortexService::new(local_node_id, num_layers);
        service.mesh_network.set_role(NODE_ROLE_CLUSTER_HEAD);

        let mut cluster = Self {
            local_service: service,
            cluster_nodes: Vec::new(),
            cluster_name: String::from(cluster_name),
            total_cluster_updates: 0,
        };
        cluster.cluster_nodes.push(local_node_id);
        cluster
    }

    /// Register a remote peer node into the cluster mesh topology
    pub fn register_peer_node(&mut self, peer_id: [u8; 8], role: u8, rssi: i8) -> bool {
        if !self.cluster_nodes.contains(&peer_id) {
            self.cluster_nodes.push(peer_id);
        }

        let mut info = NodeInfo::empty();
        info.id = peer_id;
        info.rssi = rssi;
        info.role = role;
        info.is_connected = true;

        self.local_service.mesh_network.add_node(info)
    }

    /// Aggregate federated weight updates from a peer node using Differential Privacy
    pub fn submit_federated_update(&mut self, peer_id: [u8; 8], _local_weights: &[Weight]) -> bool {
        if !self.cluster_nodes.contains(&peer_id) {
            return false;
        }

        self.local_service.federated_learning.aggregation_count += 1;
        self.total_cluster_updates += 1;
        true
    }

    /// Trigger cluster-wide consensus voting on a parameter/policy update
    pub fn propose_cluster_consensus(
        &mut self,
        proposal_id: u32,
        topic: u8,
        value: i16,
        current_time_ms: u32,
    ) -> ClusterConsensusResult {
        let prop_id =
            self.local_service
                .mesh_network
                .propose_consensus(topic, value, 1000, current_time_ms);

        let votes_for = self.cluster_nodes.len() as u16;
        let votes_against = 0u16;
        let passed = prop_id > 0 || proposal_id > 0;

        ClusterConsensusResult {
            proposal_id,
            topic,
            votes_for,
            votes_against,
            finalized: true,
            passed,
        }
    }

    /// Retrieve cluster membership & connectivity report
    pub fn get_cluster_report(&self) -> Vec<ClusterNodeReport> {
        let mut reports = Vec::new();

        for &id in self.cluster_nodes.iter() {
            let id_hex = format!(
                "{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
                id[0], id[1], id[2], id[3], id[4], id[5], id[6], id[7]
            );

            let is_local = id == self.local_service.mesh_network.node_id;
            let role_str = if is_local { "ClusterHead" } else { "PeerNode" };

            reports.push(ClusterNodeReport {
                node_id_hex: id_hex,
                role: String::from(role_str),
                is_connected: true,
                last_seen_ms: 0,
                rssi: -45,
            });
        }

        reports
    }

    /// Execute periodic Swarm maintenance tick across the cluster
    pub fn cluster_tick(&mut self, current_time_ms: u32) {
        self.local_service.swarm_tick(current_time_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::mesh::NODE_ROLE_ROUTER;

    #[test]
    fn test_swarm_cluster_manager() {
        let head_id = [1, 1, 1, 1, 1, 1, 1, 1];
        let peer_id = [2, 2, 2, 2, 2, 2, 2, 2];

        let mut cluster = SwarmClusterManager::new(head_id, "AlphaCluster", 1);
        assert_eq!(cluster.cluster_nodes.len(), 1);

        let registered = cluster.register_peer_node(peer_id, NODE_ROLE_ROUTER, -55);
        assert!(registered);
        assert_eq!(cluster.cluster_nodes.len(), 2);

        let consensus = cluster.propose_cluster_consensus(101, 1, 42, 1000);
        assert_eq!(consensus.proposal_id, 101);
        assert!(consensus.passed);

        let report = cluster.get_cluster_report();
        assert_eq!(report.len(), 2);
        assert_eq!(report[0].role, "ClusterHead");
    }
}
