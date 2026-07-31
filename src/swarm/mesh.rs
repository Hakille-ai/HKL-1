use crate::core::math::{FixedPoint, Weight, XorShift64Star};
use crate::core::memory::SynapseId;
use crate::snn::synapse;

pub const MAX_NODES: usize = 128;
pub const MAX_ROUTES: usize = 512;
pub const GOSSIP_FANOUT: u8 = 3;
pub const DISCOVERY_INTERVAL: u32 = 100;
pub const HEARTBEAT_TIMEOUT: u32 = 500;
pub const ROUTE_TIMEOUT: u32 = 3000;
pub const MAX_RECONNECT_RETRIES: u8 = 5;
pub const CONSENSUS_TIMEOUT: u32 = 200;
pub const NODE_ROLE_LEAF: u8 = 0;
pub const NODE_ROLE_ROUTER: u8 = 1;
pub const NODE_ROLE_CLUSTER_HEAD: u8 = 2;

#[derive(Clone, Copy)]
pub struct NodeInfo {
    pub id: [u8; 8],
    pub rssi: i8,
    pub is_connected: bool,
    pub last_seen: u32,
    pub clock_offset: i32,
    pub round_trip: u32,
    pub protocol_version: u8,
    pub role: u8,
    pub capability_score: u8,
    pub route_hops: u8,
}

impl NodeInfo {
    pub const fn empty() -> Self {
        Self {
            id: [0; 8],
            rssi: 0,
            is_connected: false,
            last_seen: 0,
            clock_offset: 0,
            round_trip: 0,
            protocol_version: 1,
            role: NODE_ROLE_LEAF,
            capability_score: 0,
            route_hops: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RouteEntry {
    pub destination: [u8; 8],
    pub next_hop: [u8; 8],
    pub hop_count: u8,
    pub last_validated: u32,
    pub rssi_avg: i8,
    pub is_active: bool,
    pub failures: u8,
}

impl RouteEntry {
    pub const fn empty() -> Self {
        Self {
            destination: [0; 8],
            next_hop: [0; 8],
            hop_count: 0,
            last_validated: 0,
            rssi_avg: 0,
            is_active: false,
            failures: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ReconnectEntry {
    pub node_id: [u8; 8],
    pub retry_count: u8,
    pub next_attempt: u32,
    pub backoff: u32,
    pub is_pending: bool,
}

impl ReconnectEntry {
    pub const fn empty() -> Self {
        Self {
            node_id: [0; 8],
            retry_count: 0,
            next_attempt: 0,
            backoff: 1000,
            is_pending: false,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ConsensusVote {
    pub proposal_id: u32,
    pub proposer: [u8; 8],
    pub topic: u8,
    pub value: i16,
    pub votes_for: u16,
    pub votes_against: u16,
    pub deadline: u32,
    pub finalised: bool,
    pub result: i16,
}

impl ConsensusVote {
    pub const fn empty() -> Self {
        Self {
            proposal_id: 0,
            proposer: [0; 8],
            topic: 0,
            value: 0,
            votes_for: 0,
            votes_against: 0,
            deadline: 0,
            finalised: false,
            result: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct EmergentObservation {
    pub timestamp: u32,
    pub source: [u8; 8],
    pub pattern_type: u8,
    pub intensity: FixedPoint,
    pub confidence: FixedPoint,
    pub payload: [u8; 16],
}

impl EmergentObservation {
    pub const fn empty() -> Self {
        Self {
            timestamp: 0,
            source: [0; 8],
            pattern_type: 0,
            intensity: FixedPoint::ZERO,
            confidence: FixedPoint::ZERO,
            payload: [0; 16],
        }
    }
}

#[derive(Clone, Copy)]
pub struct GossipMessage {
    pub sender_id: [u8; 8],
    pub seq: u32,
    pub msg_type: u8,
    pub payload: [u8; 128],
    pub payload_len: u8,
    pub timestamp: u32,
    pub hop_count: u8,
    pub originator: [u8; 8],
}

impl GossipMessage {
    pub const fn empty() -> Self {
        Self {
            sender_id: [0; 8],
            seq: 0,
            msg_type: 0,
            payload: [0; 128],
            payload_len: 0,
            timestamp: 0,
            hop_count: 0,
            originator: [0; 8],
        }
    }
}

#[derive(Clone, Copy)]
pub struct RemoteSpike {
    pub neuron_idx: u16,
    pub amplitude: FixedPoint,
    pub source_node: [u8; 8],
    pub timestamp: u32,
    pub hop_count: u8,
}

impl RemoteSpike {
    pub const fn empty() -> Self {
        Self {
            neuron_idx: 0,
            amplitude: FixedPoint::ZERO,
            source_node: [0; 8],
            timestamp: 0,
            hop_count: 0,
        }
    }
}

pub struct MeshNetwork {
    pub connected_nodes: [NodeInfo; MAX_NODES],
    pub node_count: u8,
    pub node_id: [u8; 8],
    pub node_role: u8,
    pub sync_interval_ms: u32,
    pub last_sync: u32,
    pub discovery_counter: u32,
    pub gossip_round: u32,
    pub gossip_queue: [GossipMessage; 128],
    pub gossip_count: u16,
    pub remote_spikes: [RemoteSpike; 512],
    pub remote_spike_count: u16,
    pub clock_drift: i32,
    pub last_clock_sync: u32,
    pub routes: [RouteEntry; MAX_ROUTES],
    pub route_count: u16,
    pub reconnect_queue: [ReconnectEntry; 32],
    pub reconnect_count: u8,
    pub active_proposals: [ConsensusVote; 8],
    pub proposal_count: u8,
    pub emergent_observations: [EmergentObservation; 64],
    pub emergent_count: u16,
    pub collective_agreement: FixedPoint,
    pub network_health: FixedPoint,
}

impl MeshNetwork {
    pub const fn new() -> Self {
        Self {
            connected_nodes: [NodeInfo::empty(); MAX_NODES],
            node_count: 0,
            node_id: [0; 8],
            node_role: NODE_ROLE_LEAF,
            sync_interval_ms: 10000,
            last_sync: 0,
            discovery_counter: 0,
            gossip_round: 0,
            gossip_queue: [GossipMessage::empty(); 128],
            gossip_count: 0,
            remote_spikes: [RemoteSpike::empty(); 512],
            remote_spike_count: 0,
            clock_drift: 0,
            last_clock_sync: 0,
            routes: [RouteEntry::empty(); MAX_ROUTES],
            route_count: 0,
            reconnect_queue: [ReconnectEntry::empty(); 32],
            reconnect_count: 0,
            active_proposals: [ConsensusVote::empty(); 8],
            proposal_count: 0,
            emergent_observations: [EmergentObservation::empty(); 64],
            emergent_count: 0,
            collective_agreement: FixedPoint::from_f32(0.5),
            network_health: FixedPoint::ONE,
        }
    }

    pub fn init(&mut self, id_seed: u64) {
        let mut rng = XorShift64Star::new(id_seed);
        for b in self.node_id.iter_mut() {
            *b = (rng.next_u32() & 0xFF) as u8;
        }
    }

    pub fn set_role(&mut self, role: u8) {
        self.node_role = role;
        if role == NODE_ROLE_CLUSTER_HEAD {
            self.gossip_queue = [GossipMessage::empty(); 128];
        }
    }

    // ------------------------------------------------------------------
    // Node management (scaled to 128 nodes)
    // ------------------------------------------------------------------

    pub fn add_node(&mut self, info: NodeInfo) -> bool {
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].id == info.id {
                self.connected_nodes[i] = info;
                return true;
            }
        }
        if (self.node_count as usize) < MAX_NODES {
            let idx = self.node_count as usize;
            self.connected_nodes[idx] = info;
            self.node_count += 1;
            self.recompute_health();
            true
        } else {
            for i in 0..MAX_NODES {
                if !self.connected_nodes[i].is_connected {
                    self.connected_nodes[i] = info;
                    self.recompute_health();
                    return true;
                }
            }
            false
        }
    }

    pub fn remove_node(&mut self, node_id: &[u8; 8]) -> bool {
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].id == *node_id {
                self.connected_nodes[i].is_connected = false;
                self.connected_nodes[i].rssi = -128;
                self.remove_routes_to(node_id);
                self.recompute_health();
                return true;
            }
        }
        false
    }

    fn recompute_health(&mut self) {
        let mut connected = 0u32;
        let total = self.node_count.max(1) as u32;
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].is_connected {
                connected += 1;
            }
        }
        let route_ratio = if self.route_count > 0 { 1.0 } else { 0.0 };
        let health = (connected as f32 / total as f32) * 0.7 + route_ratio * 0.3;
        self.network_health = FixedPoint::from_f32(health);
    }

    // ------------------------------------------------------------------
    // Multi-hop routing (100+ nodes support)
    // ------------------------------------------------------------------

    pub fn add_route(
        &mut self,
        destination: [u8; 8],
        next_hop: [u8; 8],
        hops: u8,
        rssi: i8,
        now: u32,
    ) {
        if destination == self.node_id || next_hop == self.node_id {
            return;
        }
        for i in 0..self.route_count as usize {
            if self.routes[i].destination == destination {
                if hops < self.routes[i].hop_count || self.routes[i].rssi_avg < rssi {
                    self.routes[i].next_hop = next_hop;
                    self.routes[i].hop_count = hops;
                    self.routes[i].last_validated = now;
                    self.routes[i].rssi_avg = rssi;
                    self.routes[i].is_active = true;
                }
                return;
            }
        }
        if (self.route_count as usize) < MAX_ROUTES {
            let idx = self.route_count as usize;
            self.routes[idx] = RouteEntry {
                destination,
                next_hop,
                hop_count: hops,
                last_validated: now,
                rssi_avg: rssi,
                is_active: true,
                failures: 0,
            };
            self.route_count += 1;
        }
    }

    pub fn find_route(&self, destination: &[u8; 8]) -> Option<&RouteEntry> {
        for i in 0..self.route_count as usize {
            if self.routes[i].destination == *destination && self.routes[i].is_active {
                return Some(&self.routes[i]);
            }
        }
        None
    }

    pub fn update_route_rssi(&mut self, node_id: &[u8; 8], rssi: i8) {
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].id == *node_id {
                self.connected_nodes[i].rssi = rssi;
                break;
            }
        }
        for i in 0..self.route_count as usize {
            if self.routes[i].destination == *node_id {
                self.routes[i].rssi_avg = (self.routes[i].rssi_avg as i16 + rssi as i16 / 2) as i8;
                self.routes[i].last_validated =
                    unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
                break;
            }
        }
    }

    pub fn remove_routes_to(&mut self, node_id: &[u8; 8]) {
        for i in 0..self.route_count as usize {
            if self.routes[i].destination == *node_id || self.routes[i].next_hop == *node_id {
                self.routes[i].is_active = false;
            }
        }
    }

    pub fn route_broadcast(&mut self, msg: &GossipMessage, rng: &mut XorShift64Star, _now: u32) {
        if msg.hop_count >= 5 {
            return;
        }
        let mut fwd = *msg;
        fwd.hop_count += 1;
        fwd.sender_id = self.node_id;
        if let Some(route) = self.find_route(&msg.originator) {
            let target = route.destination;
            let mut candidates: [usize; 8] = [0; 8];
            let mut found = 0;
            for i in 0..self.node_count as usize {
                if self.connected_nodes[i].is_connected
                    && self.connected_nodes[i].id != target
                    && self.connected_nodes[i].id != msg.sender_id
                {
                    candidates[found] = i;
                    found += 1;
                    if found >= 8 {
                        break;
                    }
                }
            }
            if found > 0 {
                let pick = (rng.next_u32() as usize) % found;
                let hop_id = self.connected_nodes[candidates[pick]].id;
                for i in 0..self.node_count as usize {
                    if self.connected_nodes[i].id == hop_id {
                        self.enqueue_gossip(fwd);
                        break;
                    }
                }
            }
        }
        self.enqueue_gossip(fwd);
    }

    // ------------------------------------------------------------------
    // Self-healing
    // ------------------------------------------------------------------

    pub fn mark_node_failed(&mut self, node_id: &[u8; 8], now: u32) {
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].id == *node_id {
                self.connected_nodes[i].is_connected = false;
                break;
            }
        }
        self.remove_routes_to(node_id);
        for r in 0..self.reconnect_count as usize {
            if self.reconnect_queue[r].node_id == *node_id {
                return;
            }
        }
        if self.reconnect_count < 32 {
            let idx = self.reconnect_count as usize;
            self.reconnect_queue[idx] = ReconnectEntry {
                node_id: *node_id,
                retry_count: 0,
                next_attempt: now + 1000,
                backoff: 1000,
                is_pending: true,
            };
            self.reconnect_count += 1;
        }
        self.recompute_health();
    }

    pub fn process_reconnections(&mut self, now: u32) {
        let mut keep = [false; 32];
        let mut new_count: u8 = 0;
        for i in 0..self.reconnect_count as usize {
            if !self.reconnect_queue[i].is_pending {
                continue;
            }
            if self.reconnect_queue[i].next_attempt > now {
                keep[i] = true;
                new_count += 1;
                continue;
            }
            let node_id = self.reconnect_queue[i].node_id;
            let mut found = false;
            for j in 0..self.node_count as usize {
                if self.connected_nodes[j].id == node_id && self.connected_nodes[j].is_connected {
                    found = true;
                    break;
                }
            }
            if found {
                keep[i] = false;
                continue;
            }
            let retry = self.reconnect_queue[i].retry_count;
            if retry >= MAX_RECONNECT_RETRIES {
                keep[i] = false;
                continue;
            }
            let backoff = self.reconnect_queue[i].backoff * 2;
            self.reconnect_queue[i].retry_count = retry + 1;
            self.reconnect_queue[i].next_attempt = now + backoff.min(30000);
            self.reconnect_queue[i].backoff = backoff.min(30000);
            for j in 0..self.node_count as usize {
                if self.connected_nodes[j].id == node_id {
                    self.connected_nodes[j].is_connected = true;
                    self.connected_nodes[j].last_seen = now;
                    break;
                }
            }
            keep[i] = true;
            new_count += 1;
        }
        let mut write_idx = 0;
        for i in 0..self.reconnect_count as usize {
            if keep[i] {
                if write_idx != i {
                    self.reconnect_queue[write_idx] = self.reconnect_queue[i];
                }
                write_idx += 1;
            }
        }
        self.reconnect_count = new_count;
        self.recompute_health();
    }

    pub fn route_failover(&mut self, failed_hop: &[u8; 8]) -> bool {
        let mut repaired = false;
        for i in 0..self.route_count as usize {
            if self.routes[i].next_hop == *failed_hop && self.routes[i].is_active {
                let _dest = self.routes[i].destination;
                let mut best_alt: Option<(i8, usize)> = None;
                for j in 0..self.node_count as usize {
                    if self.connected_nodes[j].is_connected
                        && self.connected_nodes[j].id != *failed_hop
                        && self.connected_nodes[j].id != self.node_id
                    {
                        let rssi = self.connected_nodes[j].rssi;
                        match best_alt {
                            None => best_alt = Some((rssi, j)),
                            Some((best_rssi, _)) if rssi > best_rssi => best_alt = Some((rssi, j)),
                            _ => {}
                        }
                    }
                }
                if let Some((_, alt_idx)) = best_alt {
                    self.routes[i].next_hop = self.connected_nodes[alt_idx].id;
                    self.routes[i].hop_count += 1;
                    self.routes[i].failures = 0;
                    repaired = true;
                } else {
                    self.routes[i].is_active = false;
                }
            }
        }
        self.recompute_health();
        repaired
    }

    pub fn prune_stale_routes(&mut self, now: u32) {
        for i in 0..self.route_count as usize {
            if self.routes[i].is_active && now - self.routes[i].last_validated > ROUTE_TIMEOUT {
                self.routes[i].failures += 1;
                if self.routes[i].failures > 3 {
                    self.routes[i].is_active = false;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Collective intelligence
    // ------------------------------------------------------------------

    pub fn propose_consensus(&mut self, topic: u8, value: i16, duration: u32, now: u32) -> u32 {
        if self.proposal_count >= 8 {
            return 0;
        }
        let pid = self.gossip_round ^ (self.node_id[0] as u32) ^ now;
        let idx = self.proposal_count as usize;
        self.active_proposals[idx] = ConsensusVote {
            proposal_id: pid,
            proposer: self.node_id,
            topic,
            value,
            votes_for: 1,
            votes_against: 0,
            deadline: now + duration,
            finalised: false,
            result: 0,
        };
        self.proposal_count += 1;
        pid
    }

    pub fn cast_vote(&mut self, proposal_id: u32, vote_for: bool) -> bool {
        for i in 0..self.proposal_count as usize {
            if self.active_proposals[i].proposal_id == proposal_id
                && !self.active_proposals[i].finalised
            {
                if vote_for {
                    self.active_proposals[i].votes_for += 1;
                } else {
                    self.active_proposals[i].votes_against += 1;
                }
                return true;
            }
        }
        false
    }

    pub fn finalise_consensus(&mut self, now: u32) {
        for i in 0..self.proposal_count as usize {
            let p = &mut self.active_proposals[i];
            if !p.finalised && now >= p.deadline {
                let total = p.votes_for + p.votes_against;
                if total >= 3 && p.votes_for > p.votes_against {
                    p.result = p.value;
                    self.collective_agreement =
                        FixedPoint::from_f32(p.votes_for as f32 / total as f32);
                } else {
                    p.result = 0;
                }
                p.finalised = true;
            }
        }
        self.prune_finalised_proposals();
    }

    fn prune_finalised_proposals(&mut self) {
        let mut keep = [false; 8];
        let mut new_count: u8 = 0;
        for i in 0..self.proposal_count as usize {
            keep[i] = !self.active_proposals[i].finalised;
            if keep[i] {
                new_count += 1;
            }
        }
        let mut write_idx = 0;
        for i in 0..self.proposal_count as usize {
            if keep[i] {
                if write_idx != i {
                    self.active_proposals[write_idx] = self.active_proposals[i];
                }
                write_idx += 1;
            }
        }
        self.proposal_count = new_count;
    }

    pub fn record_emergent_observation(
        &mut self,
        source: [u8; 8],
        pattern_type: u8,
        intensity: FixedPoint,
        confidence: FixedPoint,
        payload: [u8; 16],
        now: u32,
    ) {
        if self.emergent_count < 64 {
            let idx = self.emergent_count as usize;
            self.emergent_observations[idx] = EmergentObservation {
                timestamp: now,
                source,
                pattern_type,
                intensity,
                confidence,
                payload,
            };
            self.emergent_count += 1;
        }
    }

    pub fn detect_emergent_pattern(&self) -> u8 {
        if self.emergent_count < 3 {
            return 0;
        }
        let mut type_counts: [u16; 16] = [0; 16];
        let mut type_intensity: [f32; 16] = [0.0; 16];
        for i in 0..self.emergent_count as usize {
            let t = self.emergent_observations[i].pattern_type as usize;
            if t < 16 {
                type_counts[t] += 1;
                type_intensity[t] += self.emergent_observations[i].intensity.to_f32();
            }
        }
        let mut best_type = 0u8;
        let mut best_score = 0.0f32;
        for t in 0..16 {
            if type_counts[t] >= 3 {
                let avg = type_intensity[t] / type_counts[t] as f32;
                let score = type_counts[t] as f32 * avg;
                if score > best_score {
                    best_score = score;
                    best_type = t as u8;
                }
            }
        }
        best_type
    }

    pub fn consensus_on_observation(&mut self, now: u32) -> Option<u8> {
        let pattern = self.detect_emergent_pattern();
        if pattern == 0 {
            return None;
        }
        let pid = self.propose_consensus(1, pattern as i16, CONSENSUS_TIMEOUT, now);
        if pid != 0 {
            self.cast_vote(pid, true);
        }
        Some(pattern)
    }

    // ------------------------------------------------------------------
    // Existing mesh operations (extended)
    // ------------------------------------------------------------------

    pub fn discover_peers(&mut self, now: u32) {
        self.discovery_counter += 1;
        if self.discovery_counter < DISCOVERY_INTERVAL {
            return;
        }
        self.discovery_counter = 0;
        let mut failed_ids: [[u8; 8]; 16] = [[0; 8]; 16];
        let mut fail_count = 0;
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].is_connected
                && now > HEARTBEAT_TIMEOUT
                && now - self.connected_nodes[i].last_seen > HEARTBEAT_TIMEOUT
            {
                failed_ids[fail_count] = self.connected_nodes[i].id;
                fail_count += 1;
                if fail_count >= 16 {
                    break;
                }
            }
        }
        for f in 0..fail_count {
            self.mark_node_failed(&failed_ids[f], now);
        }
    }

    pub fn send_heartbeat(&mut self, now: u32) -> GossipMessage {
        let mut msg = GossipMessage::empty();
        msg.sender_id = self.node_id;
        msg.seq = self.gossip_round;
        msg.msg_type = 1;
        msg.originator = self.node_id;
        msg.payload_len = 8;
        msg.payload[0..4].copy_from_slice(&now.to_le_bytes());
        msg.payload[4] = self.node_role;
        msg.payload[5] = self.node_count;
        msg.payload[6] = (self.network_health.to_f32() * 100.0) as u8;
        msg.payload[7] = self.route_count as u8;
        msg.timestamp = now;
        self.gossip_round += 1;
        msg
    }

    pub fn receive_heartbeat(&mut self, msg: &GossipMessage, now: u32) {
        let idx = self.find_node(&msg.sender_id);
        let rtt = now.wrapping_sub(msg.timestamp);
        if let Some(i) = idx {
            self.connected_nodes[i].last_seen = now;
            self.connected_nodes[i].is_connected = true;
            self.connected_nodes[i].round_trip = rtt;
            if msg.payload_len >= 8 {
                self.connected_nodes[i].role = msg.payload[4];
                self.connected_nodes[i].route_hops = msg.payload[7];
            }
            let peer_time = u32::from_le_bytes(msg.payload[0..4].try_into().unwrap_or([0; 4]));
            let estimated_offset = peer_time as i32 - now as i32;
            self.connected_nodes[i].clock_offset = estimated_offset;
            self.add_route(
                msg.sender_id,
                msg.sender_id,
                1,
                self.connected_nodes[i].rssi,
                now,
            );
        } else {
            let mut info = NodeInfo::empty();
            info.id = msg.sender_id;
            info.last_seen = now;
            info.is_connected = true;
            info.round_trip = rtt;
            if msg.payload_len >= 8 {
                info.role = msg.payload[4];
                info.route_hops = msg.payload[7];
            }
            let rssi_est = -((now.wrapping_sub(msg.timestamp) % 80) as i8 + 30);
            info.rssi = rssi_est;
            self.add_node(info);
            self.add_route(msg.sender_id, msg.sender_id, 1, rssi_est, now);
        }
    }

    pub fn gossip_propagate(&mut self, msg: &GossipMessage, rng: &mut XorShift64Star) {
        let mut targets: usize = 0;
        let mut candidates = [0usize; MAX_NODES];
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].is_connected && self.connected_nodes[i].id != msg.sender_id {
                candidates[targets] = i;
                targets += 1;
            }
        }
        if targets == 0 {
            return;
        }
        let fanout = (GOSSIP_FANOUT as usize).min(targets);
        let start = (rng.next_u32() as usize) % targets;
        for f in 0..fanout {
            let idx = candidates[(start + f) % targets];
            let _target_id = self.connected_nodes[idx].id;
            let mut fwd = *msg;
            fwd.sender_id = self.node_id;
            if fwd.originator == [0; 8] {
                fwd.originator = msg.sender_id;
            }
            fwd.hop_count += 1;
            self.enqueue_gossip(fwd);
        }
    }

    pub fn enqueue_gossip(&mut self, msg: GossipMessage) -> bool {
        if self.gossip_count < 128 {
            self.gossip_queue[self.gossip_count as usize] = msg;
            self.gossip_count += 1;
            true
        } else {
            false
        }
    }

    pub fn process_gossip_queue(&mut self, now: u32) {
        let mut keep = [false; 128];
        let mut new_count: u16 = 0;
        for i in 0..self.gossip_count as usize {
            let age = now.wrapping_sub(self.gossip_queue[i].timestamp);
            if age < 1000 {
                keep[i] = true;
                new_count += 1;
            }
        }
        let mut write_idx = 0;
        for i in 0..self.gossip_count as usize {
            if keep[i] {
                if write_idx != i {
                    self.gossip_queue[write_idx] = self.gossip_queue[i];
                }
                write_idx += 1;
            }
        }
        self.gossip_count = new_count;
    }

    pub fn send_remote_spike(
        &mut self,
        neuron_idx: u16,
        amplitude: FixedPoint,
        now: u32,
    ) -> RemoteSpike {
        RemoteSpike {
            neuron_idx,
            amplitude,
            source_node: self.node_id,
            timestamp: now,
            hop_count: 0,
        }
    }

    pub fn receive_remote_spike(&mut self, spike: RemoteSpike) {
        if self.remote_spike_count < 512 {
            self.remote_spikes[self.remote_spike_count as usize] = spike;
            self.remote_spike_count += 1;
        }
    }

    pub fn apply_remote_spikes(&mut self) {
        let count = self.remote_spike_count.min(512);
        for i in 0..count {
            let spike = self.remote_spikes[i as usize];
            if spike.amplitude > FixedPoint::ZERO && spike.hop_count < 3 {
                let nid = crate::core::memory::NeuronId::new(spike.neuron_idx);
                let state = crate::core::memory::neuron_state(nid);
                let attenuation = FixedPoint::from_f32(0.3).pow((spike.hop_count + 1) as u32);
                state.membrane_potential += spike.amplitude * attenuation;
            }
        }
        self.remote_spike_count = 0;
    }

    // ------------------------------------------------------------------
    // Collective intelligence: emergent behavior aggregation
    // ------------------------------------------------------------------

    pub fn aggregate_collective_knowledge(&mut self, now: u32) {
        if self.node_role != NODE_ROLE_CLUSTER_HEAD {
            return;
        }
        if self.emergent_count > 0 {
            let dominant = self.detect_emergent_pattern();
            if dominant > 0 {
                let pid = self.propose_consensus(2, dominant as i16, CONSENSUS_TIMEOUT, now);
                if pid != 0 {
                    for _i in 0..self.node_count.min(10) {
                        let _ = self.collective_agreement;
                    }
                }
            }
        }
        if self.emergent_count > 32 {
            self.emergent_count = 0;
        }
    }

    // ------------------------------------------------------------------
    // Clock sync (unchanged)
    // ------------------------------------------------------------------

    pub fn clock_sync(&mut self, now: u32) -> i32 {
        if now - self.last_clock_sync < self.sync_interval_ms {
            return 0;
        }
        self.last_clock_sync = now;
        let mut offset_sum = 0i64;
        let mut count = 0;
        for i in 0..self.node_count as usize {
            if self.connected_nodes[i].is_connected {
                offset_sum += self.connected_nodes[i].clock_offset as i64;
                count += 1;
            }
        }
        if count > 0 {
            self.clock_drift = (offset_sum / count) as i32;
        }
        self.clock_drift
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    pub fn find_node(&self, id: &[u8; 8]) -> Option<usize> {
        (0..self.node_count as usize).find(|&i| self.connected_nodes[i].id == *id)
    }

    // ------------------------------------------------------------------
    // Weight delta (unchanged)
    // ------------------------------------------------------------------

    pub fn compute_weight_deltas(&self, previous_weights: &[Weight]) -> WeightDelta {
        let mut deltas = WeightDelta {
            entries: [(Weight::ZERO, SynapseId::INVALID); 2048],
            count: 0,
        };
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let limit = count.min(2048);
        let prev_len = previous_weights.len().min(limit as usize);
        for i in 0..prev_len as u16 {
            let id = SynapseId::new(i);
            let current = synapse::synapse_ref(id);
            let delta = current.weight.0 - previous_weights[i as usize].0;
            if delta.abs() > 100 {
                deltas.entries[deltas.count as usize] =
                    (Weight::from_f32(delta as f32 / 256.0), id);
                deltas.count += 1;
            }
        }
        deltas
    }

    pub fn apply_weight_deltas(&self, deltas: &WeightDelta) {
        for i in 0..deltas.count {
            let (delta, id) = deltas.entries[i as usize];
            if id != SynapseId::INVALID {
                let s = synapse::synapse(id);
                s.weight = s
                    .weight
                    .saturating_add(Weight::from_f32(delta.to_f32() * 0.5));
            }
        }
    }
}

pub struct WeightDelta {
    pub entries: [(Weight, SynapseId); 2048],
    pub count: u16,
}

pub static mut MESH: MeshNetwork = MeshNetwork::new();

pub fn mesh() -> &'static mut MeshNetwork {
    unsafe { &mut MESH }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_new() {
        let m = MeshNetwork::new();
        assert_eq!(m.node_count, 0);
        assert_eq!(m.gossip_count, 0);
        assert_eq!(m.remote_spike_count, 0);
        assert_eq!(m.route_count, 0);
        assert_eq!(m.reconnect_count, 0);
    }

    #[test]
    fn test_add_node() {
        let mut m = MeshNetwork::new();
        let info = NodeInfo {
            id: [1, 2, 3, 4, 5, 6, 7, 8],
            rssi: -50,
            is_connected: true,
            last_seen: 100,
            clock_offset: 0,
            round_trip: 0,
            protocol_version: 1,
            role: NODE_ROLE_LEAF,
            capability_score: 0,
            route_hops: 0,
        };
        assert!(m.add_node(info));
        assert_eq!(m.node_count, 1);
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut m = MeshNetwork::new();
        let info = NodeInfo {
            id: [1; 8],
            rssi: -50,
            is_connected: true,
            last_seen: 100,
            clock_offset: 0,
            round_trip: 0,
            protocol_version: 1,
            role: NODE_ROLE_LEAF,
            capability_score: 0,
            route_hops: 0,
        };
        assert!(m.add_node(info));
        let info2 = NodeInfo { rssi: -30, ..info };
        assert!(m.add_node(info2));
        assert_eq!(m.node_count, 1);
    }

    #[test]
    fn test_remove_node() {
        let mut m = MeshNetwork::new();
        let id = [1; 8];
        let info = NodeInfo {
            id,
            is_connected: true,
            ..NodeInfo::empty()
        };
        m.add_node(info);
        assert!(m.remove_node(&id));
        assert!(!m.connected_nodes[0].is_connected);
    }

    #[test]
    fn test_add_route() {
        let mut m = MeshNetwork::new();
        let dest = [2; 8];
        let hop = [3; 8];
        m.add_route(dest, hop, 2, -60, 100);
        assert_eq!(m.route_count, 1);
        let route = m.find_route(&dest);
        assert!(route.is_some());
        assert_eq!(route.unwrap().hop_count, 2);
    }

    #[test]
    fn test_route_failover() {
        let mut m = MeshNetwork::new();
        let dest = [5; 8];
        let hop = [4; 8];
        let alt = [6; 8];
        m.add_route(dest, hop, 1, -50, 100);
        let info = NodeInfo {
            id: alt,
            is_connected: true,
            rssi: -40,
            ..NodeInfo::empty()
        };
        m.add_node(info);
        assert!(m.route_failover(&hop));
        let route = m.find_route(&dest);
        assert!(route.is_some());
        assert_eq!(route.unwrap().next_hop, alt);
    }

    #[test]
    fn test_mark_node_failed_and_reconnect() {
        let mut m = MeshNetwork::new();
        let id = [7; 8];
        let info = NodeInfo {
            id,
            is_connected: true,
            last_seen: 100,
            ..NodeInfo::empty()
        };
        m.add_node(info);
        m.mark_node_failed(&id, 500);
        assert!(!m.connected_nodes[0].is_connected);
        assert_eq!(m.reconnect_count, 1);
        m.process_reconnections(2000);
        assert!(m.connected_nodes[0].is_connected);
    }

    #[test]
    fn test_propose_consensus() {
        let mut m = MeshNetwork::new();
        let pid = m.propose_consensus(1, 42, 100, 50);
        assert!(pid != 0);
        assert_eq!(m.proposal_count, 1);
        assert!(m.cast_vote(pid, true));
        m.finalise_consensus(200);
        assert!(m.active_proposals[0].finalised || m.proposal_count == 0);
    }

    #[test]
    fn test_emergent_observation() {
        let mut m = MeshNetwork::new();
        for i in 0..5 {
            m.record_emergent_observation(
                [i; 8],
                3,
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(0.9),
                [0; 16],
                100 + i as u32 * 10,
            );
        }
        assert_eq!(m.emergent_count, 5);
        let pattern = m.detect_emergent_pattern();
        assert_eq!(pattern, 3);
    }

    #[test]
    fn test_heartbeat_with_routing() {
        let mut m = MeshNetwork::new();
        m.init(42);
        let now = 1000;
        let hb = m.send_heartbeat(now);
        assert_eq!(hb.payload[4], NODE_ROLE_LEAF);

        let mut msg = GossipMessage::empty();
        msg.sender_id = [8; 8];
        msg.payload_len = 8;
        msg.payload[0..4].copy_from_slice(&now.to_le_bytes());
        msg.timestamp = now;
        m.receive_heartbeat(&msg, now + 50);
        assert!(m.find_route(&[8; 8]).is_some());
    }

    #[test]
    fn test_discovery_timeout() {
        let mut m = MeshNetwork::new();
        let info = NodeInfo {
            id: [2; 8],
            rssi: -70,
            is_connected: true,
            last_seen: 10,
            clock_offset: 0,
            round_trip: 5,
            protocol_version: 1,
            role: NODE_ROLE_LEAF,
            capability_score: 0,
            route_hops: 0,
        };
        m.add_node(info);
        m.discovery_counter = DISCOVERY_INTERVAL;
        m.discover_peers(600);
        assert!(!m.connected_nodes[0].is_connected);
    }

    #[test]
    fn test_gossip_queue_full() {
        let mut m = MeshNetwork::new();
        for _ in 0..128 {
            m.enqueue_gossip(GossipMessage::empty());
        }
        assert!(!m.enqueue_gossip(GossipMessage::empty()));
    }

    #[test]
    fn test_process_gossip_expires_old() {
        let mut m = MeshNetwork::new();
        let mut msg = GossipMessage::empty();
        msg.timestamp = 500;
        m.enqueue_gossip(msg);
        m.process_gossip_queue(1000);
        assert_eq!(m.gossip_count, 1);
        m.process_gossip_queue(2000);
        assert_eq!(m.gossip_count, 0);
    }

    #[test]
    fn test_remote_spike_with_hop() {
        let mut m = MeshNetwork::new();
        let spike = m.send_remote_spike(42, FixedPoint::from_f32(0.5), 100);
        assert_eq!(spike.hop_count, 0);
        m.receive_remote_spike(spike);
        assert_eq!(m.remote_spike_count, 1);
    }

    #[test]
    fn test_clock_sync_no_nodes() {
        let mut m = MeshNetwork::new();
        let drift = m.clock_sync(1000);
        assert_eq!(drift, 0);
    }

    #[test]
    fn test_find_node_not_found() {
        let m = MeshNetwork::new();
        assert!(m.find_node(&[0xFF; 8]).is_none());
    }

    #[test]
    fn test_init_sets_node_id() {
        let mut m = MeshNetwork::new();
        m.init(42);
        assert!(m.node_id.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_set_role() {
        let mut m = MeshNetwork::new();
        m.set_role(NODE_ROLE_CLUSTER_HEAD);
        assert_eq!(m.node_role, NODE_ROLE_CLUSTER_HEAD);
    }

    #[test]
    fn test_network_health() {
        let mut m = MeshNetwork::new();
        let info = NodeInfo {
            id: [1; 8],
            is_connected: true,
            ..NodeInfo::empty()
        };
        m.add_node(info);
        assert!(m.network_health > FixedPoint::ZERO);
    }

    #[test]
    fn test_max_nodes_capacity() {
        let mut m = MeshNetwork::new();
        for i in 0..MAX_NODES as u8 {
            let info = NodeInfo {
                id: [i; 8],
                is_connected: true,
                ..NodeInfo::empty()
            };
            m.add_node(info);
        }
        assert_eq!(m.node_count as usize, MAX_NODES);
    }

    #[test]
    fn test_prune_stale_routes() {
        let mut m = MeshNetwork::new();
        m.add_route([9; 8], [10; 8], 1, -50, 100);
        m.prune_stale_routes(5000);
        assert!(!m.routes[0].is_active || m.routes[0].failures > 0);
    }
}
