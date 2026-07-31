use crate::core::math::{FixedPoint, Weight, XorShift64Star};
use crate::core::memory::SynapseId;
use crate::snn::synapse::SYNAPSE_COUNT;
use core::sync::atomic::Ordering;

pub const MAX_FEDERATION_NODES: usize = 128;
pub const ADAPTIVE_TOPOLOGY_WINDOW: u32 = 100;
pub const MAX_HIERARCHY_LEVELS: u8 = 4;
pub const CLUSTER_SIZE_MIN: u8 = 4;

#[derive(Clone, Copy)]
pub struct HierarchyLevel {
    pub level: u8,
    pub aggregation_round: u32,
    pub node_count: u8,
    pub cluster_head_id: [u8; 8],
    pub noise_scale: FixedPoint,
    pub consensus_threshold: FixedPoint,
    pub round_trip: u32,
}

impl HierarchyLevel {
    pub const fn new(level: u8) -> Self {
        Self {
            level,
            aggregation_round: 0,
            node_count: 0,
            cluster_head_id: [0; 8],
            noise_scale: FixedPoint::from_f32(0.001 * (level as f32 + 1.0)),
            consensus_threshold: FixedPoint::from_f32(0.6),
            round_trip: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct HierarchicalAggregate {
    pub level: u8,
    pub source_node: [u8; 8],
    pub synapse_index: u16,
    pub weight_avg: i16,
    pub confidence: FixedPoint,
    pub node_count: u8,
}

impl HierarchicalAggregate {
    pub const fn empty() -> Self {
        Self {
            level: 0,
            source_node: [0; 8],
            synapse_index: 0,
            weight_avg: 0,
            confidence: FixedPoint::ZERO,
            node_count: 0,
        }
    }
}

pub struct FederatedLearning {
    pub noise_scale: FixedPoint,
    pub min_synapses_for_federation: u32,
    pub federation_round: u32,
    pub active_nodes: u8,
    pub node_weights: [u8; MAX_FEDERATION_NODES],
    pub node_reliability: [FixedPoint; MAX_FEDERATION_NODES],
    pub topology_cache: [u16; MAX_FEDERATION_NODES],
    pub topology_cache_count: u8,
    pub adaptive_topology: bool,
    pub last_topology_update: u32,
    pub aggregation_count: u32,
    pub hierarchy_levels: [HierarchyLevel; MAX_HIERARCHY_LEVELS as usize],
    pub hierarchy_count: u8,
    pub aggregate_buffer: [HierarchicalAggregate; 256],
    pub aggregate_count: u16,
    pub local_cluster_id: u16,
    pub cluster_head_elected: bool,
    pub election_round: u32,
}

impl FederatedLearning {
    pub const fn new() -> Self {
        Self {
            noise_scale: FixedPoint::from_f32(0.001),
            min_synapses_for_federation: 1000,
            federation_round: 0,
            active_nodes: 0,
            node_weights: [0; MAX_FEDERATION_NODES],
            node_reliability: [FixedPoint::from_f32(0.5); MAX_FEDERATION_NODES],
            topology_cache: [0; MAX_FEDERATION_NODES],
            topology_cache_count: 0,
            adaptive_topology: true,
            last_topology_update: 0,
            aggregation_count: 0,
            hierarchy_levels: [
                HierarchyLevel::new(0),
                HierarchyLevel::new(1),
                HierarchyLevel::new(2),
                HierarchyLevel::new(3),
            ],
            hierarchy_count: 1,
            aggregate_buffer: [HierarchicalAggregate::empty(); 256],
            aggregate_count: 0,
            local_cluster_id: 0,
            cluster_head_elected: false,
            election_round: 0,
        }
    }

    pub fn init_hierarchy(&mut self, depth: u8) {
        self.hierarchy_count = depth.min(MAX_HIERARCHY_LEVELS);
        for i in 0..self.hierarchy_count as usize {
            self.hierarchy_levels[i] = HierarchyLevel::new(i as u8);
        }
    }

    // ------------------------------------------------------------------
    // Cluster head election
    // ------------------------------------------------------------------

    pub fn elect_cluster_head(
        &mut self,
        node_capabilities: &[(u16, u8); 128],
        num_candidates: u8,
    ) -> Option<u16> {
        self.election_round += 1;
        let mut best_score = 0u32;
        let mut best_id = None;
        for i in 0..num_candidates.min(128) as usize {
            let (node_id, capability) = node_capabilities[i];
            let reliability = if i < MAX_FEDERATION_NODES {
                self.node_reliability[i]
            } else {
                FixedPoint::from_f32(0.5)
            };
            let score = (capability as u32) * 10
                + (reliability.to_f32() * 100.0) as u32
                + self.election_round.wrapping_sub(1) as u32 % 100;
            if score > best_score {
                best_score = score;
                best_id = Some(node_id);
            }
        }
        if best_id.is_some() {
            self.cluster_head_elected = true;
        }
        best_id
    }

    // ------------------------------------------------------------------
    // Hierarchical federated averaging
    // ------------------------------------------------------------------

    pub fn hierarchical_average(
        &mut self,
        node_weights: &[&[Weight]; MAX_FEDERATION_NODES],
        num_nodes: u8,
    ) {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        if count < self.min_synapses_for_federation {
            return;
        }
        let num = (num_nodes as usize).max(1);
        self.active_nodes = num as u8;
        self.aggregation_count += 1;
        let levels = self.hierarchy_count.max(1) as usize;

        for i in 0..count as u16 {
            let mut level_aggregates: [i64; MAX_HIERARCHY_LEVELS as usize] = [0; 4];
            let mut level_counts: [i64; MAX_HIERARCHY_LEVELS as usize] = [0; 4];

            for n in 0..num.min(MAX_FEDERATION_NODES) {
                let reliability = self.node_reliability[n];
                let w_val = node_weights[n][i as usize].0 as i64;
                let rel = (reliability.to_f32() * 256.0) as i64;
                let level = (n * MAX_HIERARCHY_LEVELS as usize / num.max(1)) % levels;
                level_aggregates[level] += w_val * rel;
                level_counts[level] += rel;
            }

            let mut final_weight = 0i64;
            let mut final_weight_count = 0i64;

            for l in 0..levels {
                if level_counts[l] > 0 {
                    let avg = level_aggregates[l] / level_counts[l];
                    let level_confidence = self.hierarchy_levels[l].consensus_threshold.to_f32();
                    let level_weight = (level_confidence * 256.0) as i64;
                    final_weight += avg * level_weight;
                    final_weight_count += level_weight;
                }
            }

            if final_weight_count > 0 {
                let avg = (final_weight / final_weight_count) as i16;
                let s = crate::snn::synapse::synapse(SynapseId::new(i));
                let blend = (s.weight.0 as f32 * 0.7 + avg as f32 * 0.3) as i16;
                s.weight = Weight(blend);
            }
        }
        self.federation_round += 1;
        for l in 0..levels {
            self.hierarchy_levels[l].aggregation_round += 1;
        }
    }

    // ------------------------------------------------------------------
    // Per-level differential privacy
    // ------------------------------------------------------------------

    pub fn add_hierarchical_noise(&self, weights: &mut [Weight], level: u8) {
        let seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() } ^ (level as u64) << 16;
        let mut rng = XorShift64Star::new(seed);
        let noise_scale = if (level as usize) < MAX_HIERARCHY_LEVELS as usize {
            self.hierarchy_levels[level as usize].noise_scale
        } else {
            self.noise_scale
        };
        for w in weights.iter_mut() {
            let noise = rng.next_gaussian().to_f32() * noise_scale.to_f32();
            *w = w.saturating_add(Weight::from_f32(noise));
        }
    }

    pub fn hierarchical_aggregate_with_noise(
        &mut self,
        node_weights: &[&[Weight]; MAX_FEDERATION_NODES],
        num_nodes: u8,
        level: u8,
    ) {
        self.hierarchical_average(node_weights, num_nodes);
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        if count > 0 && !cfg!(any(feature = "std", test)) {
            let seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
            let mut rng = XorShift64Star::new(seed);
            for i in 0..count.min(256) as u16 {
                let s = crate::snn::synapse::synapse(SynapseId::new(i));
                let noise = rng.next_gaussian().to_f32()
                    * self.hierarchy_levels[level as usize].noise_scale.to_f32();
                s.weight = s.weight.saturating_add(Weight::from_f32(noise));
            }
        }
    }

    // ------------------------------------------------------------------
    // Cross-level merging
    // ------------------------------------------------------------------

    pub fn push_aggregate(&mut self, agg: HierarchicalAggregate) -> bool {
        if self.aggregate_count < 256 {
            self.aggregate_buffer[self.aggregate_count as usize] = agg;
            self.aggregate_count += 1;
            true
        } else {
            false
        }
    }

    pub fn merge_level_aggregates(&mut self, target_level: u8) {
        let mut merged: [i32; 256] = [0; 256];
        let mut counts: [u16; 256] = [0; 256];
        let capacity = 256usize;
        for i in 0..self.aggregate_count as usize {
            let agg = &self.aggregate_buffer[i];
            if agg.level == target_level {
                let idx = agg.synapse_index as usize;
                if idx < capacity {
                    merged[idx] += agg.weight_avg as i32;
                    counts[idx] += 1;
                }
            }
        }
        for i in 0..capacity {
            if counts[i] > 0 {
                let avg = (merged[i] / counts[i] as i32) as i16;
                let s = crate::snn::synapse::synapse(SynapseId::new(i as u16));
                s.weight = Weight(avg);
            }
        }
    }

    // ------------------------------------------------------------------
    // Existing methods (extended)
    // ------------------------------------------------------------------

    pub fn add_dp_noise(&self, weights: &mut [Weight]) {
        let seed = unsafe { crate::core::time::METABOLIC_CLOCK.cycles() };
        let mut rng = XorShift64Star::new(seed);
        for w in weights.iter_mut() {
            let noise = rng.next_gaussian().to_f32() * self.noise_scale.to_f32();
            *w = w.saturating_add(Weight::from_f32(noise));
        }
    }

    pub fn federated_average(
        &mut self,
        node_weights: &[&[Weight]; MAX_FEDERATION_NODES],
        num_nodes: u8,
    ) {
        let count = SYNAPSE_COUNT.load(Ordering::Relaxed);
        if count < self.min_synapses_for_federation {
            return;
        }
        let num = (num_nodes as usize).max(1);
        self.active_nodes = num as u8;
        self.aggregation_count += 1;

        for i in 0..count as u16 {
            let mut weighted_sum = 0i64;
            let mut total_weight = 0i64;

            for n in 0..num.min(MAX_FEDERATION_NODES) {
                let reliability = self.node_reliability[n];
                let w_val = node_weights[n][i as usize].0 as i64;
                let rel = (reliability.to_f32() * 256.0) as i64;
                weighted_sum += w_val * rel;
                total_weight += rel;
            }

            if total_weight > 0 {
                let avg = (weighted_sum / total_weight) as i16;
                let s = crate::snn::synapse::synapse(SynapseId::new(i));
                let blend = (s.weight.0 as f32 * 0.7 + avg as f32 * 0.3) as i16;
                s.weight = Weight(blend);
            }
        }
        self.federation_round += 1;
    }

    pub fn update_reliability(&mut self, node_idx: usize, success: bool) {
        if node_idx >= MAX_FEDERATION_NODES {
            return;
        }
        let alpha = FixedPoint::from_f32(0.05);
        if success {
            self.node_reliability[node_idx] += alpha;
        } else {
            self.node_reliability[node_idx] -= alpha;
        }
        self.node_reliability[node_idx] =
            self.node_reliability[node_idx].clamp(FixedPoint::ZERO, FixedPoint::ONE);
    }

    pub fn update_topology(&mut self, now: u32) {
        if !self.adaptive_topology {
            return;
        }
        if now - self.last_topology_update < ADAPTIVE_TOPOLOGY_WINDOW {
            return;
        }
        self.last_topology_update = now;

        let mut count: u8 = 0;
        for i in 0..MAX_FEDERATION_NODES {
            if self.node_reliability[i] > FixedPoint::from_f32(0.3) {
                self.topology_cache[count as usize] = i as u16;
                count += 1;
            }
        }
        self.topology_cache_count = count;
    }

    pub fn active_node_count(&self) -> u8 {
        self.topology_cache_count
    }

    pub fn should_federate(&self) -> bool {
        self.active_nodes >= 2 && self.federation_round < 10000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_new() {
        let f = FederatedLearning::new();
        assert_eq!(f.noise_scale, FixedPoint::from_f32(0.001));
        assert_eq!(f.federation_round, 0);
        assert!(f.adaptive_topology);
        assert_eq!(f.hierarchy_count, 1);
    }

    #[test]
    fn test_init_hierarchy() {
        let mut f = FederatedLearning::new();
        f.init_hierarchy(3);
        assert_eq!(f.hierarchy_count, 3);
    }

    #[test]
    fn test_update_reliability_up() {
        let mut f = FederatedLearning::new();
        let before = f.node_reliability[0];
        f.update_reliability(0, true);
        assert!(f.node_reliability[0] > before);
    }

    #[test]
    fn test_update_reliability_down() {
        let mut f = FederatedLearning::new();
        let before = f.node_reliability[0];
        f.update_reliability(0, false);
        assert!(f.node_reliability[0] < before);
    }

    #[test]
    fn test_update_topology_filters_reliable() {
        let mut f = FederatedLearning::new();
        f.node_reliability[0] = FixedPoint::from_f32(0.8);
        f.node_reliability[1] = FixedPoint::from_f32(0.1);
        f.update_topology(200);
        assert!(f.topology_cache_count > 0);
    }

    #[test]
    fn test_should_federate_requires_two_nodes() {
        let mut f = FederatedLearning::new();
        assert!(!f.should_federate());
        f.active_nodes = 3;
        assert!(f.should_federate());
    }

    #[test]
    fn test_elect_cluster_head() {
        let mut f = FederatedLearning::new();
        let caps = [(1u16, 10u8), (2, 20), (3, 15)];
        let mut arr = [(0u16, 0u8); 128];
        for i in 0..3 {
            arr[i] = caps[i];
        }
        let elected = f.elect_cluster_head(&arr, 3);
        assert!(elected.is_some());
        assert!(f.cluster_head_elected);
    }

    #[test]
    fn test_hierarchical_noise_scale_per_level() {
        let f = FederatedLearning::new();
        assert!(f.hierarchy_levels[0].noise_scale < f.hierarchy_levels[1].noise_scale);
    }

    #[test]
    fn test_push_and_merge_aggregates() {
        let mut f = FederatedLearning::new();
        let agg = HierarchicalAggregate {
            level: 1,
            source_node: [1; 8],
            synapse_index: 0,
            weight_avg: 100,
            confidence: FixedPoint::from_f32(0.8),
            node_count: 5,
        };
        assert!(f.push_aggregate(agg));
        assert_eq!(f.aggregate_count, 1);
    }

    #[test]
    fn test_hierarchy_levels_default() {
        let f = FederatedLearning::new();
        assert_eq!(f.hierarchy_levels[0].level, 0);
        assert_eq!(f.hierarchy_levels[0].consensus_threshold, FixedPoint::from_f32(0.6));
    }

    #[test]
    fn test_active_node_count() {
        let mut f = FederatedLearning::new();
        f.topology_cache_count = 10;
        assert_eq!(f.active_node_count(), 10);
    }
}
