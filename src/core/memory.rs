//! Neuron and synapse storage with a static pool allocator, CSR weight matrix,
//! homeostatic scaling, and neurogenesis support. All memory is pre-allocated
//! at compile time with zero dynamic allocation.

#[allow(unused_imports)]
use crate::core::atomic::FetchAtomic;
use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Maximum neurons and synapses (compile-time constants from lib.rs)
pub const MAX_NEURONS: usize = crate::MAX_NEURONS;
pub const MAX_SYNAPSES: usize = crate::MAX_SYNAPSES;
pub const NEUROGENESIS_POOL_SIZE: usize = 1024;

/// Neuron ID type
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
#[repr(transparent)]
pub struct NeuronId(pub u16);

impl NeuronId {
    pub const INVALID: Self = Self(u16::MAX);
    #[inline(always)]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Synapse ID type
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
#[repr(transparent)]
pub struct SynapseId(pub u16);

impl SynapseId {
    pub const INVALID: Self = Self(u16::MAX);
    #[inline(always)]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Weight type (re-export)
pub type Weight = crate::core::math::Weight;

// ============================================================================
// STATIC POOL ALLOCATOR - Zero allocation, deterministic
// ============================================================================

pub struct StaticPool<T, const N: usize> {
    storage: [core::mem::MaybeUninit<T>; N],
    free_list: [u16; N],
    free_head: AtomicUsize,
    allocated: AtomicUsize,
}

impl<T, const N: usize> StaticPool<T, N> {
    pub const fn new() -> Self {
        Self {
            storage: [const { core::mem::MaybeUninit::uninit() }; N],
            free_list: [0; N],
            free_head: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
        }
    }

    /// Initialize the free list (must be called once at startup)
    pub fn init(&self) {
        for i in 0..N {
            // SAFETY: We're writing to the free_list array
            unsafe {
                let ptr = &self.free_list as *const _ as *mut u16;
                *ptr.add(i) = i as u16;
            }
        }
        self.free_head.store(N, Ordering::Relaxed);
        self.allocated.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn allocate(&self) -> Option<NonNull<T>> {
        let head = self.free_head.load(Ordering::Acquire);
        if head == 0 {
            return None;
        }

        let new_head = head - 1;
        let idx = unsafe { *self.free_list.get_unchecked(new_head) } as usize;

        if self
            .free_head
            .compare_exchange(head, new_head, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            let base = &self.storage as *const [MaybeUninit<T>; N] as *mut T;
            let ptr = unsafe { NonNull::new_unchecked(base.add(idx)) };
            Some(ptr)
        } else {
            self.allocate()
        }
    }

    #[inline(always)]
    pub fn deallocate(&self, ptr: NonNull<T>) {
        let base = self.storage.as_ptr() as usize;
        let idx = (ptr.as_ptr() as usize - base) / core::mem::size_of::<T>();
        let head = self.free_head.load(Ordering::Acquire);
        let list_ptr = &self.free_list as *const [u16; N] as *mut u16;
        unsafe {
            *list_ptr.add(head) = idx as u16;
        }
        self.free_head.store(head + 1, Ordering::Release);
        self.allocated.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn allocated_count(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Get index of a pointer within the pool
    #[inline(always)]
    pub fn index_of(&self, ptr: NonNull<T>) -> usize {
        (ptr.as_ptr() as usize - self.storage.as_ptr() as usize) / core::mem::size_of::<T>()
    }
}

// ============================================================================
// NEUROGENESIS POOL - Recycles synapses for structural plasticity
// ============================================================================

pub struct NeurogenesisPool {
    // Free synapse slots
    free_synapses: StaticPool<SynapseSlot, MAX_SYNAPSES>,
    // Adjacency lists for each neuron (pre and post)
    pre_synaptic: [SynapseList; MAX_NEURONS],
    post_synaptic: [SynapseList; MAX_NEURONS],
}

#[derive(Clone, Copy)]
pub struct SynapseSlot {
    pub weight: Weight,
    pub delay: u8,
    pub plasticity_enabled: bool,
    pub pre: NeuronId,
    pub post: NeuronId,
    pub next_pre: SynapseId,  // Linked list for pre-synaptic
    pub next_post: SynapseId, // Linked list for post-synaptic
}

pub struct SynapseList {
    head: AtomicUsize, // SynapseId as usize
    count: AtomicUsize,
}

impl SynapseList {
    pub const fn new() -> Self {
        Self {
            head: AtomicUsize::new(SynapseId::INVALID.index()),
            count: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn push(&self, slot: &mut SynapseSlot, id: SynapseId) {
        let old_head = self.head.swap(id.index(), Ordering::AcqRel);
        slot.next_pre = SynapseId::new(old_head as u16);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn pop(&self, slot: &SynapseSlot) -> Option<usize> {
        let mut head = self.head.load(Ordering::Acquire);
        while head != SynapseId::INVALID.index() {
            let next = slot.next_pre.index();
            if self
                .head
                .compare_exchange(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.count.fetch_sub(1, Ordering::Relaxed);
                return Some(head);
            }
            head = self.head.load(Ordering::Acquire);
        }
        None
    }

    #[inline(always)]
    pub fn iter(&self) -> SynapseIter {
        SynapseIter {
            current: self.head.load(Ordering::Relaxed),
        }
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

pub struct SynapseIter {
    current: usize,
}

impl Iterator for SynapseIter {
    type Item = SynapseId;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == SynapseId::INVALID.index() {
            return None;
        }
        let id = SynapseId::new(self.current as u16);
        // Note: would need pool reference to get next - simplified here
        self.current = SynapseId::INVALID.index();
        Some(id)
    }
}

impl NeurogenesisPool {
    pub const fn new() -> Self {
        Self {
            free_synapses: StaticPool::new(),
            pre_synaptic: [const { SynapseList::new() }; MAX_NEURONS],
            post_synaptic: [const { SynapseList::new() }; MAX_NEURONS],
        }
    }

    pub fn init(&mut self) {
        self.free_synapses.init();
        for _i in 0..MAX_SYNAPSES {
            if let Some(mut ptr) = self.free_synapses.allocate() {
                unsafe {
                    *ptr.as_mut() = SynapseSlot {
                        weight: Weight::ZERO,
                        delay: 0,
                        plasticity_enabled: true,
                        pre: NeuronId::INVALID,
                        post: NeuronId::INVALID,
                        next_pre: SynapseId::INVALID,
                        next_post: SynapseId::INVALID,
                    };
                }
                self.free_synapses.deallocate(ptr);
            }
        }
    }

    /// Create new synapse (structural plasticity - neurogenesis)
    pub fn create_synapse(
        &mut self,
        pre: NeuronId,
        post: NeuronId,
        weight: Weight,
        delay: u8,
    ) -> Option<SynapseId> {
        let mut ptr = self.free_synapses.allocate()?;
        let idx = self.free_synapses.index_of(ptr);
        let id = SynapseId::new(idx as u16);

        unsafe {
            *ptr.as_mut() = SynapseSlot {
                weight,
                delay,
                plasticity_enabled: true,
                pre,
                post,
                next_pre: SynapseId::INVALID,
                next_post: SynapseId::INVALID,
            };
        }

        unsafe {
            let slot = &mut *ptr.as_ptr();
            self.pre_synaptic[pre.index()].push(slot, id);
            self.post_synaptic[post.index()].push(slot, id);
        }

        Some(id)
    }

    pub fn destroy_synapse(&mut self, id: SynapseId) {
        let slot_ptr = self.free_synapses.storage[id.index()].as_mut_ptr();
        unsafe {
            (*slot_ptr).pre = NeuronId::INVALID;
            (*slot_ptr).post = NeuronId::INVALID;
            (*slot_ptr).weight = Weight::ZERO;
        }
        let nn = unsafe { NonNull::new_unchecked(slot_ptr) };
        self.free_synapses.deallocate(nn);
    }

    #[inline(always)]
    pub fn get_slot(&self, id: SynapseId) -> &SynapseSlot {
        unsafe { &*self.free_synapses.storage[id.index()].as_ptr() }
    }

    #[inline(always)]
    pub fn get_slot_mut(&mut self, id: SynapseId) -> &mut SynapseSlot {
        unsafe { &mut *self.free_synapses.storage[id.index()].as_mut_ptr() }
    }

    #[inline(always)]
    pub fn pre_synaptic(&self, neuron: NeuronId) -> &SynapseList {
        &self.pre_synaptic[neuron.index()]
    }

    #[inline(always)]
    pub fn post_synaptic(&self, neuron: NeuronId) -> &SynapseList {
        &self.post_synaptic[neuron.index()]
    }

    pub fn prune_below_threshold(&mut self, threshold: Weight, _inactivity_cycles: u32) -> usize {
        let mut pruned = 0;
        // Simplified: iterate all synapses
        for i in 0..MAX_SYNAPSES {
            let id = SynapseId::new(i as u16);
            let slot = self.get_slot(id);
            if slot.pre != NeuronId::INVALID && slot.weight < threshold {
                // Check inactivity (would need activity tracking)
                self.destroy_synapse(id);
                pruned += 1;
            }
        }
        pruned
    }
}

// ============================================================================
// NEURON STATE STORAGE (contiguous arrays for cache efficiency)
// ============================================================================

#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct NeuronState {
    pub membrane_potential: FixedPoint, // V_m
    pub threshold: FixedPoint,          // V_th
    pub leak: FixedPoint,               // Leak rate
    pub refractory_remaining: u16,      // Refractory period counter
    pub last_spike_time: u32,           // For STDP
    pub bias_current: FixedPoint,       // I_bias
    pub layer: u8,                      // Layer index (0-7)
    pub neuron_type: NeuronType,
    pub flags: NeuronFlags,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum NeuronType {
    #[default]
    LIF = 0, // Standard Leaky Integrate-and-Fire
    ALIF = 1,       // Adaptive LIF (spike-frequency adaptation)
    BURST = 2,      // Bursting neuron
    INHIBITORY = 3, // Inhibitory interneuron
    PACER = 4,      // Metabolic pacemaker (1Hz)
    REFLEX = 5,     // Hard-coded reflex arc
}

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct NeuronFlags(pub u8);

impl NeuronFlags {
    pub const REFRACTORY: u8 = 1 << 0;
    pub const PLASTICITY_DISABLED: u8 = 1 << 1; // Hard-coded reflex
    pub const NEUROGENESIS_CANDIDATE: u8 = 1 << 2;
    pub const SILENCED: u8 = 1 << 3; // Pruned
    pub const PREDICTOR_MODE: u8 = 1 << 4; // Part of predictor network

    #[inline(always)]
    pub fn has(&self, flag: u8) -> bool {
        (self.0 & flag) != 0
    }
    #[inline(always)]
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }
    #[inline(always)]
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

const UNINIT_NEURON: core::mem::MaybeUninit<NeuronState> = core::mem::MaybeUninit::uninit();
pub static mut NEURON_ARRAY: [core::mem::MaybeUninit<NeuronState>; MAX_NEURONS] =
    [UNINIT_NEURON; MAX_NEURONS];
pub static NEURON_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn neuron_state(id: NeuronId) -> &'static mut NeuronState {
    unsafe { &mut *NEURON_ARRAY[id.index()].as_mut_ptr() }
}

pub fn neuron_state_ref(id: NeuronId) -> &'static NeuronState {
    unsafe { &*NEURON_ARRAY[id.index()].as_ptr() }
}

pub fn allocate_neuron(neuron_type: NeuronType, layer: u8) -> Option<NeuronId> {
    let count = NEURON_COUNT.fetch_add(1, Ordering::AcqRel);
    if count >= MAX_NEURONS {
        return None;
    }
    let id = NeuronId::new(count as u16);
    let state = neuron_state(id);
    state.membrane_potential = FixedPoint::ZERO;
    state.threshold = FixedPoint::from_f32(1.0);
    state.leak = FixedPoint::from_f32(0.9);
    state.refractory_remaining = 0;
    state.last_spike_time = 0;
    state.bias_current = FixedPoint::ZERO;
    state.layer = layer;
    state.neuron_type = neuron_type;
    state.flags = NeuronFlags::default();
    Some(id)
}

// ============================================================================
// SYNAPSE WEIGHT MATRIX (Compressed Sparse Row - CSR)
// ============================================================================

pub struct SynapseMatrix {
    pub weights: &'static mut [Weight],
    pub col_indices: &'static mut [SynapseId],
    pub row_ptr: &'static mut [u16], // length = num_neurons + 1
    pub num_neurons: usize,
    pub nnz: usize,
}

impl SynapseMatrix {
    #[inline(always)]
    pub fn row_range(&self, neuron: NeuronId) -> (usize, usize) {
        let r = neuron.index();
        let start = self.row_ptr[r] as usize;
        let end = self.row_ptr[r + 1] as usize;
        (start, end)
    }

    #[inline(always)]
    pub fn row_weights(&self, neuron: NeuronId) -> &[Weight] {
        let (s, e) = self.row_range(neuron);
        &self.weights[s..e]
    }

    #[inline(always)]
    pub fn row_weights_mut(&mut self, neuron: NeuronId) -> &mut [Weight] {
        let (s, e) = self.row_range(neuron);
        &mut self.weights[s..e]
    }

    #[inline(always)]
    pub fn row_targets(&self, neuron: NeuronId) -> &[SynapseId] {
        let (s, e) = self.row_range(neuron);
        &self.col_indices[s..e]
    }

    // Spike propagation: for each target, accumulate weight
    #[inline]
    pub fn propagate_spike(&self, source: NeuronId, target_potentials: &mut [FixedPoint]) {
        let (s, e) = self.row_range(source);
        for i in s..e {
            let target = self.col_indices[i].index();
            target_potentials[target] += self.weights[i].to_fixed();
        }
    }
}

// ============================================================================
// HOMEOSTATIC SCALING - Global synaptic scaling
// ============================================================================

pub struct HomeostaticScaler {
    pub target_rate: FixedPoint,     // Target firing rate (Hz)
    pub scaling_factor: FixedPoint,  // Global multiplier
    pub adaptation_rate: FixedPoint, // How fast to adapt
    pub measurement_window: u32,     // Time window for rate estimation
}

impl HomeostaticScaler {
    pub const fn new() -> Self {
        Self {
            target_rate: FixedPoint::from_f32(10.0), // 10 Hz target
            scaling_factor: FixedPoint::ONE,
            adaptation_rate: FixedPoint::from_f32(0.001),
            measurement_window: 10000, // 10s at 1kHz
        }
    }

    pub fn update(&mut self, avg_rate: FixedPoint) {
        let error = self.target_rate - avg_rate;
        let delta = error * self.adaptation_rate;
        self.scaling_factor += delta;
        // Clamp scaling factor
        self.scaling_factor = self
            .scaling_factor
            .clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(10.0));
    }

    #[inline(always)]
    pub fn apply(&self, weight: Weight) -> Weight {
        Weight::from_f32(weight.to_f32() * self.scaling_factor.to_f32())
    }
}

// ============================================================================
// STATIC STORAGE
// ============================================================================

#[cfg(not(feature = "std"))]
pub mod statics {
    use crate::core::math::Weight;
    use crate::core::memory::SynapseId;

    pub static mut SYNAPSE_WEIGHTS: [Weight; crate::MAX_SYNAPSES] =
        [Weight(0); crate::MAX_SYNAPSES];
    pub static mut SYNAPSE_INDICES: [SynapseId; crate::MAX_SYNAPSES] =
        [SynapseId(0); crate::MAX_SYNAPSES];
    pub static mut SYNAPSE_INDPTR: [u16; crate::MAX_NEURONS + 1] = [0; crate::MAX_NEURONS + 1];
}

pub fn init_memory() {
    use core::sync::atomic::Ordering;
    NEURON_COUNT.store(0, Ordering::Relaxed);
}

// ============================================================================
// ADAPTIVE MEMORY ENGINE & DYNAMIC POOLS (alloc / std)
// ============================================================================

/// Adaptive memory manager scaling memory capacity dynamically
pub struct AdaptiveMemoryEngine {
    pub active_capacity_neurons: AtomicUsize,
    pub active_capacity_synapses: AtomicUsize,
    pub dynamic_enabled: bool,
}

impl AdaptiveMemoryEngine {
    pub const fn new() -> Self {
        Self {
            active_capacity_neurons: AtomicUsize::new(crate::MAX_NEURONS),
            active_capacity_synapses: AtomicUsize::new(crate::MAX_SYNAPSES),
            dynamic_enabled: cfg!(any(feature = "alloc", feature = "std")),
        }
    }

    pub fn set_capacity(&self, neurons: usize, synapses: usize) {
        self.active_capacity_neurons
            .store(neurons, Ordering::Release);
        self.active_capacity_synapses
            .store(synapses, Ordering::Release);
    }

    pub fn current_capacity(&self) -> (usize, usize) {
        (
            self.active_capacity_neurons.load(Ordering::Acquire),
            self.active_capacity_synapses.load(Ordering::Acquire),
        )
    }
}

pub static ADAPTIVE_MEMORY: AdaptiveMemoryEngine = AdaptiveMemoryEngine::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_memory_engine() {
        let (n, s) = ADAPTIVE_MEMORY.current_capacity();
        assert!(n >= crate::MAX_NEURONS);
        assert!(s >= crate::MAX_SYNAPSES);

        ADAPTIVE_MEMORY.set_capacity(65536, 1048576);
        let (n2, s2) = ADAPTIVE_MEMORY.current_capacity();
        assert_eq!(n2, 65536);
        assert_eq!(s2, 1048576);

        ADAPTIVE_MEMORY.set_capacity(crate::MAX_NEURONS, crate::MAX_SYNAPSES);
    }

    #[test]
    fn neuron_id_new() {
        let id = NeuronId::new(42);
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn neuron_id_invalid() {
        assert_eq!(NeuronId::INVALID.index(), u16::MAX as usize);
    }

    #[test]
    fn synapse_id_new() {
        let id = SynapseId::new(999);
        assert_eq!(id.index(), 999);
    }

    #[test]
    fn neuron_flags_default() {
        let f = NeuronFlags::default();
        assert_eq!(f.0, 0);
    }

    #[test]
    fn neuron_flags_set_get() {
        let mut f = NeuronFlags::default();
        assert!(!f.has(NeuronFlags::REFRACTORY));
        f.set(NeuronFlags::REFRACTORY);
        assert!(f.has(NeuronFlags::REFRACTORY));
        f.clear(NeuronFlags::REFRACTORY);
        assert!(!f.has(NeuronFlags::REFRACTORY));

        f.set(NeuronFlags::PLASTICITY_DISABLED | NeuronFlags::SILENCED);
        assert!(f.has(NeuronFlags::PLASTICITY_DISABLED));
        assert!(f.has(NeuronFlags::SILENCED));
        assert!(!f.has(NeuronFlags::REFRACTORY));
    }

    #[test]
    fn neuron_state_defaults() {
        let ns = NeuronState::default();
        assert_eq!(ns.membrane_potential, FixedPoint::ZERO);
    }

    #[test]
    fn synapse_slot_defaults() {
        let slot = SynapseSlot {
            weight: Weight::ZERO,
            delay: 0,
            plasticity_enabled: true,
            pre: NeuronId::INVALID,
            post: NeuronId::INVALID,
            next_pre: SynapseId::INVALID,
            next_post: SynapseId::INVALID,
        };
        assert!(slot.weight == Weight::ZERO);
        assert_eq!(slot.delay, 0);
        assert!(slot.plasticity_enabled);
        assert!(slot.pre == NeuronId::INVALID);
    }

    #[test]
    fn static_pool_alloc_dealloc() {
        let pool: StaticPool<i32, 8> = StaticPool::new();
        pool.init();

        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.allocated_count(), 0);

        let mut ptr = pool.allocate().expect("allocate should succeed");
        unsafe {
            *ptr.as_mut() = 42;
        }
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(unsafe { *ptr.as_ref() }, 42);

        pool.deallocate(ptr);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn static_pool_alloc_all() {
        let pool: StaticPool<i32, 8> = StaticPool::new();
        pool.init();

        let mut ptrs: [Option<core::ptr::NonNull<i32>>; 8] = [None; 8];
        for i in 0..8 {
            let mut ptr = pool.allocate().expect("allocate should succeed");
            unsafe {
                *ptr.as_mut() = i as i32;
            }
            ptrs[i] = Some(ptr);
        }
        assert_eq!(pool.allocated_count(), 8);
        assert!(pool.allocate().is_none());

        for ptr in ptrs.iter().flatten() {
            pool.deallocate(*ptr);
        }
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn synapse_list_push_pop() {
        let list = SynapseList::new();
        assert_eq!(list.count(), 0);

        let mut slot = SynapseSlot {
            weight: Weight::from_f32(0.5),
            delay: 1,
            plasticity_enabled: true,
            pre: NeuronId::INVALID,
            post: NeuronId::INVALID,
            next_pre: SynapseId::INVALID,
            next_post: SynapseId::INVALID,
        };

        list.push(&mut slot, SynapseId::new(42));
        assert_eq!(list.count(), 1);

        let popped = list.pop(&slot);
        assert_eq!(popped, Some(42));
        assert_eq!(list.count(), 0);

        assert!(list.pop(&slot).is_none());
    }

    #[test]
    fn allocate_neuron_capacity() {
        NEURON_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        for i in 0..MAX_NEURONS {
            let id = allocate_neuron(NeuronType::LIF, 0);
            assert!(id.is_some(), "allocation {} should succeed", i);
            assert_eq!(id.unwrap().index(), i);
        }

        assert!(allocate_neuron(NeuronType::LIF, 0).is_none());

        NEURON_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
    }
}
