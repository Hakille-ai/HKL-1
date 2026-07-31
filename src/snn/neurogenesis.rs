//! Structural plasticity through neurogenesis. Creates and prunes synapses dynamically.
use crate::core::math::{FixedPoint, Weight, XorShift64Star};
use crate::core::memory::{MAX_NEURONS, NEURON_COUNT, NeuronFlags, NeuronId, SynapseId};

/// Structural plasticity through neurogenesis (Section 12)
/// Recycles silent synapses and creates new connections

pub struct NeurogenesisEngine {
    /// Pool of available synapse slots (linked list of free IDs)
    free_pool_head: SynapseId,
    free_count: u16,
    /// Adjacency matrix tracking pre/post connections
    pub adj_pre_count: [u16; MAX_NEURONS],
    pub adj_post_count: [u16; MAX_NEURONS],
    /// RNG for exploratory connections
    rng: XorShift64Star,
    /// Parameters
    pub min_weight_threshold: Weight,
    pub inactivity_cycles_before_prune: u16,
    pub max_connections_per_neuron: u16,
    /// Senescence parameters (Section 12 - Synaptic Aging)
    pub max_age: u32, // Max age in cycles before forced pruning
    /// Statistics
    pub total_pruned: u32,
    pub total_created: u32,
    pub total_senesced: u32,
}

impl NeurogenesisEngine {
    pub const fn new(seed: u64) -> Self {
        Self {
            free_pool_head: SynapseId::new(0),
            free_count: 0,
            adj_pre_count: [0; MAX_NEURONS],
            adj_post_count: [0; MAX_NEURONS],
            rng: XorShift64Star::new(seed),
            min_weight_threshold: Weight::from_f32(0.01),
            inactivity_cycles_before_prune: 100,
            max_connections_per_neuron: 256,
            max_age: 1_000_000, // ~1000s at 1ms step
            total_pruned: 0,
            total_created: 0,
            total_senesced: 0,
        }
    }

    /// Apply synaptic senescence
    fn apply_senescence(&mut self) -> usize {
        let pruned = crate::snn::synapse::apply_senescence(self.max_age);
        self.total_senesced += pruned as u32;
        pruned
    }

    /// Initialize free pool from synapse array
    pub fn init_free_pool(&mut self) {
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        self.free_pool_head = SynapseId::new(count as u16);
        self.free_count = (crate::MAX_SYNAPSES - count as usize) as u16;

        // Initialize adjacency counts
        let total = count as u16;
        for i in 0..total {
            let s = crate::snn::synapse::synapse_ref(SynapseId::new(i));
            self.adj_pre_count[s.pre.index()] += 1;
            self.adj_post_count[s.post.index()] += 1;
        }
    }

    /// Prune silent synapses (Section 12)
    pub fn prune_silent_synapses(&mut self) -> usize {
        let mut pruned = 0;
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);

        for i in 0..count as u16 {
            let id = SynapseId::new(i);
            let s = crate::snn::synapse::synapse(id);
            if !s.plasticity_enabled {
                continue;
            }

            // Check weight threshold
            let below_threshold = s.weight.0.abs() < self.min_weight_threshold.0;
            let inactive = s.silence_count > self.inactivity_cycles_before_prune;

            if below_threshold && inactive {
                // Mark for recycling
                let pre = s.pre;
                let post = s.post;
                s.pre = NeuronId::INVALID;
                s.post = NeuronId::INVALID;
                s.weight = Weight::ZERO;
                s.plasticity_enabled = false;

                // Update adjacency counts
                if pre != NeuronId::INVALID {
                    self.adj_pre_count[pre.index()] =
                        self.adj_pre_count[pre.index()].saturating_sub(1);
                }
                if post != NeuronId::INVALID {
                    self.adj_post_count[post.index()] =
                        self.adj_post_count[post.index()].saturating_sub(1);
                }

                // Add to free pool for recycling
                self.free_pool_head = id;
                self.free_count += 1;

                pruned += 1;
            }
        }

        self.total_pruned += pruned as u32;
        pruned
    }

    pub fn neurogenesis(&mut self, max_new: u16) -> usize {
        let mut created: usize = 0;
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);

        for pre_id in 0..count as u16 {
            if created >= max_new as usize {
                break;
            }
            if self.adj_pre_count[pre_id as usize] >= self.max_connections_per_neuron {
                continue;
            }

            let pre = NeuronId::new(pre_id);
            let pre_state = crate::core::memory::neuron_state_ref(pre);

            // Only create from active neurons
            if pre_state.membrane_potential <= FixedPoint::ZERO && pre_state.last_spike_time < 1000
            {
                continue;
            }

            // Find potential post-synaptic partner in nearby layers
            for post_id in 0..count as u16 {
                if created >= max_new as usize {
                    break;
                }
                if post_id == pre_id {
                    continue;
                }
                if self.adj_post_count[post_id as usize] >= self.max_connections_per_neuron {
                    continue;
                }

                let post = NeuronId::new(post_id);
                let post_state = crate::core::memory::neuron_state_ref(post);

                // Connection probability based on layer compatibility
                let prob = self.connection_probability(pre_state.layer, post_state.layer);
                if self.rng.next_f32() >= prob {
                    continue;
                }

                // Check if synapse already exists
                if self.synapse_exists(pre, post) {
                    continue;
                }

                // Create new synapse
                let weight = Weight::from_f32(self.rng.next_gaussian().to_f32() * 0.05);
                let delay = if pre_state.layer == post_state.layer {
                    1
                } else {
                    2
                };

                if let Some(id) = crate::snn::synapse::create_synapse(pre, post, weight, delay) {
                    let s = crate::snn::synapse::synapse(id);
                    s.plasticity_enabled = true;
                    self.adj_pre_count[pre.index()] += 1;
                    self.adj_post_count[post.index()] += 1;
                    created += 1;
                }
            }
        }

        self.total_created += created as u32;
        created
    }

    /// Check if synapse already exists between pre and post
    fn synapse_exists(&self, pre: NeuronId, post: NeuronId) -> bool {
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        for i in 0..count as u16 {
            let s = crate::snn::synapse::synapse_ref(SynapseId::new(i));
            if s.pre == pre && s.post == post {
                return true;
            }
        }
        false
    }

    /// Connection probability based on layer compatibility
    fn connection_probability(&self, pre_layer: u8, post_layer: u8) -> f32 {
        match (pre_layer, post_layer) {
            (0, 1) => 0.15, // Sensory -> Inhibitory
            (0, 2) => 0.10, // Sensory -> Adaptive
            (1, 2) => 0.10, // Inhibitory -> Adaptive
            (2, 3) => 0.05, // Adaptive -> Predictor
            (2, 4) => 0.03, // Adaptive -> Motor
            (3, 3) => 0.05, // Recurrent in predictor
            (7, _) => 0.10, // Curiosity -> everywhere
            (_, _) => 0.01, // Default low probability
        }
    }

    pub fn recycle_silent_neurons(&mut self, max_count: u16) -> usize {
        let count = NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let mut recycled: usize = 0;

        for i in 0..count as u16 {
            if recycled >= max_count as usize {
                break;
            }
            let id = NeuronId::new(i);
            let state = crate::core::memory::neuron_state(id);

            if state.flags.has(NeuronFlags::SILENCED) {
                // Reactivate with random initial values
                state.membrane_potential = FixedPoint::ZERO;
                state.threshold = FixedPoint::from_f32(self.rng.next_f32() * 0.5 + 0.75);
                state.flags.clear(NeuronFlags::SILENCED);
                recycled += 1;
            }
        }
        recycled
    }

    /// Run a full maintenance cycle
    pub fn maintenance_cycle(&mut self) -> (usize, usize, usize) {
        let senesced = self.apply_senescence();
        let pruned = self.prune_silent_synapses();
        let created = self.neurogenesis(64);
        (pruned, created, senesced)
    }
}

/// Global neurogenesis instance
pub static mut NEUROGENESIS: NeurogenesisEngine = NeurogenesisEngine::new(0xDEADBEEF);

/// Convenience function
pub fn recycle_silent_neurons(max_count: u16) -> usize {
    unsafe { NEUROGENESIS.recycle_silent_neurons(max_count) }
}

/// Push a freed synapse ID onto the global free pool.
/// Called by synapse.rs when a synapse is pruned.
pub fn free_pool_push(id: SynapseId) {
    unsafe {
        NEUROGENESIS.free_pool_head = id;
        NEUROGENESIS.free_count += 1;
    }
}

/// Pop a synapse ID from the global free pool.
/// Returns Some(id) if available, None if pool is empty.
/// Called by synapse.rs::create_synapse before incrementing SYNAPSE_COUNT.
pub fn free_pool_pop() -> Option<SynapseId> {
    unsafe {
        if NEUROGENESIS.free_count > 0 {
            NEUROGENESIS.free_count -= 1;
            let id = NEUROGENESIS.free_pool_head;
            NEUROGENESIS.free_pool_head = SynapseId::new(id.index() as u16 + 1);
            Some(id)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neurogenesis_engine_new() {
        let ng = NeurogenesisEngine::new(42);
        assert_eq!(ng.free_count, 0);
    }

    #[test]
    fn neurogenesis_adj_pre_count_starts_zero() {
        let ng = NeurogenesisEngine::new(42);
        assert_eq!(ng.adj_pre_count[0], 0);
        assert_eq!(ng.adj_pre_count[100], 0);
    }

    #[test]
    fn neurogenesis_max_connections_default() {
        let ng = NeurogenesisEngine::new(42);
        assert_eq!(ng.max_connections_per_neuron, 256);
    }

    #[test]
    fn neurogenesis_prune_silent_returns_zero() {
        let mut ng = NeurogenesisEngine::new(42);
        let pruned = ng.prune_silent_synapses();
        assert_eq!(pruned, 0);
    }

    #[test]
    fn neurogenesis_recycle_silent_returns_zero_when_idle() {
        let mut ng = NeurogenesisEngine::new(42);
        let recycled = ng.recycle_silent_neurons(10);
        assert_eq!(recycled, 0);
    }

    #[test]
    fn neurogenesis_maintenance_cycle_returns_zeros() {
        let mut ng = NeurogenesisEngine::new(42);
        let (pruned, created, senesced) = ng.maintenance_cycle();
        assert_eq!(pruned, 0);
        assert_eq!(created, 0);
        assert_eq!(senesced, 0);
    }

    #[test]
    fn neurogenesis_senescence_default_params() {
        let ng = NeurogenesisEngine::new(42);
        assert_eq!(ng.max_age, 1_000_000);
        assert_eq!(ng.total_senesced, 0);
    }

    #[test]
    fn neurogenesis_apply_senescence_increments_counter() {
        let mut ng = NeurogenesisEngine::new(42);
        ng.max_age = 0;

        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let id = crate::core::memory::SynapseId::new(count as u16);
        let s = crate::snn::synapse::synapse(id);
        *s = crate::snn::synapse::Synapse::new(
            crate::core::memory::NeuronId::new(0),
            crate::core::memory::NeuronId::new(1),
            crate::core::math::Weight::from_f32(0.5),
            1,
        );
        crate::snn::synapse::SYNAPSE_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);

        s.age = 1;
        let pruned = ng.apply_senescence();
        assert_eq!(pruned, 1, "age=1 >= max_age=0 should prune");
        assert_eq!(ng.total_senesced, 1);

        crate::snn::synapse::SYNAPSE_COUNT.store(count, core::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn neurogenesis_apply_senescence_no_prune_below_max() {
        let mut ng = NeurogenesisEngine::new(42);
        let count = crate::snn::synapse::SYNAPSE_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        let id = crate::core::memory::SynapseId::new(count as u16);
        let s = crate::snn::synapse::synapse(id);
        *s = crate::snn::synapse::Synapse::new(
            crate::core::memory::NeuronId::new(0),
            crate::core::memory::NeuronId::new(1),
            crate::core::math::Weight::from_f32(0.3),
            1,
        );
        crate::snn::synapse::SYNAPSE_COUNT.fetch_add(1, core::sync::atomic::Ordering::SeqCst);

        s.age = 500_000;
        let pruned = ng.apply_senescence();
        assert_eq!(pruned, 0);
        assert_eq!(ng.total_senesced, 0);

        assert!(s.weight.to_f32() < 0.3, "Weight should decay with age");

        crate::snn::synapse::SYNAPSE_COUNT.store(count, core::sync::atomic::Ordering::SeqCst);
    }
}
