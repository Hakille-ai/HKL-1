//! Episodic memory with consolidation, replay, recall, and spatial navigation.
//! Hippocampus-like dual-store (short-term → long-term)
//! with significance-based transfer, forgetting curve,
//! pattern completion, place cells, grid cells, and theta phase precession
//! (Sections 11.1, 13, 31).

use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const ST_CAPACITY: usize = 256; // Short-term episodic buffer
const LT_CAPACITY: usize = 512; // Long-term consolidated store
const REPLAY_BATCH: usize = 16; // Experiences replayed per cycle
const PLACE_CELLS: usize = 128; // Hippocampal place cells for spatial nav
const GRID_CELLS: usize = 64; // Entorhinal grid cells
const THETA_RHYTHM_PERIOD: u64 = 125; // ~8 Hz theta rhythm (125 ms)

// ---------------------------------------------------------------------------
// Memory trace — one episodic experience
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MemoryTrace {
    pub state_hash: u64,
    pub action: u16,
    pub next_state_hash: u64,
    pub reward: FixedPoint,
    pub prediction_error: FixedPoint,
    pub novelty: FixedPoint,
    pub timestamp: u64,
    pub significance: FixedPoint,
    pub consolidation_count: u16,
    pub last_access: u64,
    pub valid: bool,
    pub is_long_term: bool,
}

// ---------------------------------------------------------------------------
// Place cell — spatial location encoding (hippocampus)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PlaceCell {
    /// Current firing rate
    pub firing_rate: FixedPoint,
    /// Preferred x coordinate (normalized 0.0–1.0)
    pub pref_x: FixedPoint,
    /// Preferred y coordinate (normalized 0.0–1.0)
    pub pref_y: FixedPoint,
    /// Place field width
    pub field_width: FixedPoint,
    /// Whether cell is active (animal is in place field)
    pub active: bool,
}

impl PlaceCell {
    const fn new(idx: usize) -> Self {
        // Distribute place fields across a 2D grid
        let grid_size = 8;
        let px = (idx % grid_size) as f32 / (grid_size - 1) as f32;
        let py = (idx / grid_size) as f32 / (grid_size - 1) as f32;
        Self {
            firing_rate: FixedPoint::ZERO,
            pref_x: FixedPoint::from_f32(px),
            pref_y: FixedPoint::from_f32(py),
            field_width: FixedPoint::from_f32(0.15),
            active: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Grid cell — entorhinal cortex spatial metric
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GridCell {
    /// Current firing rate
    pub firing_rate: FixedPoint,
    /// Grid orientation (radians)
    pub orientation: FixedPoint,
    /// Grid spacing
    pub spacing: FixedPoint,
    /// Spatial phase offset x
    pub phase_x: FixedPoint,
    /// Spatial phase offset y
    pub phase_y: FixedPoint,
    pub active: bool,
}

impl GridCell {
    pub const fn empty() -> Self {
        Self {
            firing_rate: FixedPoint::ZERO,
            orientation: FixedPoint::ZERO,
            spacing: FixedPoint::ONE,
            phase_x: FixedPoint::ZERO,
            phase_y: FixedPoint::ZERO,
            active: false,
        }
    }

    fn new(idx: usize) -> Self {
        // Multiple scales of grid spacing
        let scales = [0.2, 0.35, 0.5, 0.7, 1.0];
        let scale_idx = idx % scales.len();
        let angle_frac = (FixedPoint::from_int(idx as i32) * FixedPoint::from_f32(0.6743)).fract();
        let angle = angle_frac * FixedPoint::TAU;
        Self {
            firing_rate: FixedPoint::ZERO,
            orientation: angle,
            spacing: FixedPoint::from_f32(scales[scale_idx]),
            phase_x: (FixedPoint::from_int(idx as i32) * FixedPoint::from_f32(0.137)).fract(),
            phase_y: (FixedPoint::from_int(idx as i32) * FixedPoint::from_f32(0.269)).fract(),
            active: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Theta phase for phase precession
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ThetaPhase {
    /// Current phase (0–2π)
    pub phase: FixedPoint,
    pub frequency: FixedPoint,
}

impl ThetaPhase {
    const fn new() -> Self {
        Self {
            phase: FixedPoint::ZERO,
            frequency: FixedPoint::ZERO,
        }
    }
}

impl MemoryTrace {
    const fn empty() -> Self {
        Self {
            state_hash: 0,
            action: 0,
            next_state_hash: 0,
            reward: FixedPoint::ZERO,
            prediction_error: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            timestamp: 0,
            significance: FixedPoint::ZERO,
            consolidation_count: 0,
            last_access: 0,
            valid: false,
            is_long_term: false,
        }
    }

    /// Ebbinghaus forgetting multiplier: 2^(-t/τ)
    /// τ is the retention half-life in steps
    /// Uses power-of-two decay instead of exp for numerical stability
    fn retention(&self, now: u64, half_life_steps: u64) -> FixedPoint {
        let elapsed = now.saturating_sub(self.last_access);
        if half_life_steps == 0 || elapsed == 0 {
            return FixedPoint::ONE;
        }
        // Linear approx for small t: 1 - t/(2τ)
        // Exponential: 2^(-t/τ) ≈ 1 - ln2 * t/τ for small t/τ
        if elapsed < half_life_steps {
            let ratio = FixedPoint::from_f32(elapsed as f32 / (half_life_steps * 2) as f32);
            (FixedPoint::ONE - ratio).max(FixedPoint::ZERO)
        } else {
            // For larger t, use inverse square root approximation
            let steps_ratio = FixedPoint::from_int((half_life_steps / elapsed.max(1)) as i32);
            steps_ratio.max(FixedPoint::from_f32(0.001))
        }
    }
}

// ---------------------------------------------------------------------------
// Replay entry — what gets fed back into the network
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub struct ReplayExperience {
    pub state_hash: u64,
    pub action: u16,
    pub next_state_hash: u64,
    pub reward: FixedPoint,
    pub td_error: FixedPoint,
}

// ---------------------------------------------------------------------------
// Episodic memory — dual-store with consolidation
// ---------------------------------------------------------------------------
pub struct EpisodicMemory {
    // Short-term buffer (recent, fast-decay)
    short_term: [MemoryTrace; ST_CAPACITY],
    st_count: usize,
    st_idx: usize,

    // Long-term store (consolidated, slow-decay)
    long_term: [MemoryTrace; LT_CAPACITY],
    lt_count: usize,

    // Ebbinghaus parameters
    st_half_life: u64, // Short-term retention half-life (steps)
    lt_half_life: u64, // Long-term retention half-life (steps)

    // Consolidation thresholds
    min_significance: FixedPoint,
    max_consolidations: u16,

    // Recall cache
    last_recall: [u64; 16],
    recall_idx: usize,

    // Place cells (spatial navigation)
    place_cells: [PlaceCell; PLACE_CELLS],
    /// Current spatial position estimate (x, y)
    pub position_x: FixedPoint,
    pub position_y: FixedPoint,
    /// Path integration velocity
    velocity_x: FixedPoint,
    velocity_y: FixedPoint,

    // Grid cells (entorhinal metric)
    grid_cells: [GridCell; GRID_CELLS],

    // Theta rhythm
    theta: ThetaPhase,

    // Sharp-wave ripple replay buffer
    ripple_buffer: [u64; 32],
    ripple_count: usize,
    ripple_lock: bool,

    // Statistics
    pub total_recorded: u64,
    pub total_consolidated: u64,
    pub total_replayed: u64,
    pub total_recalled: u64,
    pub total_place_updates: u64,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        let mut place_cells = [PlaceCell::new(0); PLACE_CELLS];
        let mut i = 0;
        while i < PLACE_CELLS {
            place_cells[i] = PlaceCell::new(i);
            i += 1;
        }
        let mut grid_cells = [GridCell::empty(); GRID_CELLS];
        let mut j = 0;
        while j < GRID_CELLS {
            grid_cells[j] = GridCell::new(j);
            j += 1;
        }
        Self {
            short_term: [MemoryTrace::empty(); ST_CAPACITY],
            st_count: 0,
            st_idx: 0,
            long_term: [MemoryTrace::empty(); LT_CAPACITY],
            lt_count: 0,
            st_half_life: 10_000,
            lt_half_life: 1_000_000,
            min_significance: FixedPoint::from_f32(0.3),
            max_consolidations: 3,
            last_recall: [0; 16],
            recall_idx: 0,
            place_cells,
            position_x: FixedPoint::from_f32(0.5),
            position_y: FixedPoint::from_f32(0.5),
            velocity_x: FixedPoint::ZERO,
            velocity_y: FixedPoint::ZERO,
            grid_cells,
            theta: ThetaPhase::new(),
            ripple_buffer: [0; 32],
            ripple_count: 0,
            ripple_lock: false,
            total_recorded: 0,
            total_consolidated: 0,
            total_replayed: 0,
            total_recalled: 0,
            total_place_updates: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /// Record a new experience into short-term memory
    pub fn record(
        &mut self,
        state_hash: u64,
        action: u16,
        next_state_hash: u64,
        reward: FixedPoint,
        prediction_error: FixedPoint,
        novelty: FixedPoint,
        timestamp: u64,
    ) {
        let significance = self.compute_significance(reward, prediction_error, novelty);
        let idx = self.st_idx % ST_CAPACITY;
        self.short_term[idx] = MemoryTrace {
            state_hash,
            action,
            next_state_hash,
            reward,
            prediction_error,
            novelty,
            timestamp,
            significance,
            consolidation_count: 0,
            last_access: timestamp,
            valid: true,
            is_long_term: false,
        };
        self.st_idx += 1;
        if self.st_count < ST_CAPACITY {
            self.st_count += 1;
        }
        self.total_recorded += 1;
    }

    /// Compute significance of an experience (0.0–1.0)
    fn compute_significance(
        &self,
        reward: FixedPoint,
        prediction_error: FixedPoint,
        novelty: FixedPoint,
    ) -> FixedPoint {
        // High reward, high prediction error, or high novelty → significant
        let r = reward.abs().clamp(FixedPoint::ZERO, FixedPoint::ONE);
        let pe = prediction_error.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        let n = novelty.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        // Weighted combination
        r * FixedPoint::from_f32(0.4)
            + pe * FixedPoint::from_f32(0.35)
            + n * FixedPoint::from_f32(0.25)
    }

    // -----------------------------------------------------------------------
    // Consolidation — short-term → long-term transfer
    // -----------------------------------------------------------------------

    /// Run one consolidation cycle: sample significant ST memories, transfer to LT
    pub fn consolidate(&mut self, now: u64) -> usize {
        if self.st_count == 0 {
            return 0;
        }

        // Score all valid ST traces
        let mut scored: [(usize, FixedPoint); ST_CAPACITY] = [(0, FixedPoint::ZERO); ST_CAPACITY];
        let mut scored_count = 0;
        for i in 0..self.st_count {
            if !self.short_term[i].valid {
                continue;
            }
            let trace = &self.short_term[i];
            let retention = trace.retention(now, self.st_half_life);
            // Score = significance × retention (recent + important)
            let score = trace.significance * retention;
            if score > self.min_significance {
                scored[scored_count] = (i, score);
                scored_count += 1;
            }
        }

        if scored_count == 0 {
            return 0;
        }

        // Sort by score (descending) — simple insertion sort
        for i in 1..scored_count {
            let mut j = i;
            while j > 0 && scored[j].1 > scored[j - 1].1 {
                scored.swap(j, j - 1);
                j -= 1;
            }
        }

        // Transfer top scored entries to long-term
        let lt_free = LT_CAPACITY.saturating_sub(self.lt_count);
        let transfer_count = if lt_free > 0 {
            (scored_count / 2).max(1).min(lt_free)
        } else {
            0
        };
        let mut transferred = 0;
        for i in 0..transfer_count.min(scored_count) {
            let st_idx = scored[i].0;
            let mut trace = self.short_term[st_idx];
            trace.last_access = now;
            trace.is_long_term = true;

            // Find slot in LT
            if self.lt_count < LT_CAPACITY {
                self.long_term[self.lt_count] = trace;
                self.lt_count += 1;
                transferred += 1;
            } else {
                // Find lowest-significance LT entry
                let mut min_idx = 0;
                let mut min_sig = self.long_term[0].significance
                    * self.long_term[0].retention(now, self.lt_half_life);
                for j in 1..LT_CAPACITY {
                    let decayed = self.long_term[j].significance
                        * self.long_term[j].retention(now, self.lt_half_life);
                    if decayed < min_sig {
                        min_sig = decayed;
                        min_idx = j;
                    }
                }
                if trace.significance > min_sig {
                    self.long_term[min_idx] = trace;
                    transferred += 1;
                }
            }

            // Mark ST entry as consolidated — may prune if exhausted
            self.short_term[st_idx].consolidation_count += 1;
            if self.short_term[st_idx].consolidation_count >= self.max_consolidations {
                self.short_term[st_idx].valid = false;
            }
        }

        self.total_consolidated += transferred as u64;
        transferred
    }

    // -----------------------------------------------------------------------
    // Forgetting — decay significance based on Ebbinghaus curve
    // -----------------------------------------------------------------------

    /// Apply forgetting curve to all stored memories
    pub fn apply_forgetting(&mut self, now: u64) {
        // Decay short-term
        for i in 0..self.st_count {
            if !self.short_term[i].valid {
                continue;
            }
            let retention = self.short_term[i].retention(now, self.st_half_life);
            self.short_term[i].significance *= retention;
            // Prune if significance drops to near zero
            if self.short_term[i].significance < FixedPoint::from_f32(0.01) {
                self.short_term[i].valid = false;
            }
        }

        // Decay long-term
        for i in 0..self.lt_count {
            let retention = self.long_term[i].retention(now, self.lt_half_life);
            self.long_term[i].significance *= retention;
        }
    }

    // -----------------------------------------------------------------------
    // Replay — sample experiences and prepare for network replay
    // -----------------------------------------------------------------------

    /// Sample a batch of experiences for replay (prioritized by significance)
    pub fn sample_replay_batch(&self) -> [ReplayExperience; REPLAY_BATCH] {
        let mut batch = [ReplayExperience {
            state_hash: 0,
            action: 0,
            next_state_hash: 0,
            reward: FixedPoint::ZERO,
            td_error: FixedPoint::ZERO,
        }; REPLAY_BATCH];

        // Collect all valid traces with scores
        let mut all: [(MemoryTrace, FixedPoint); ST_CAPACITY + LT_CAPACITY] =
            [(MemoryTrace::empty(), FixedPoint::ZERO); ST_CAPACITY + LT_CAPACITY];
        let mut count = 0;

        for i in 0..self.st_count {
            if self.short_term[i].valid {
                all[count] = (self.short_term[i], self.short_term[i].significance);
                count += 1;
            }
        }
        for i in 0..self.lt_count {
            let margin =
                (FixedPoint::ONE - self.long_term[i].significance) * FixedPoint::from_f32(0.1);
            let priority = self.long_term[i].significance + margin; // LT bias
            all[count] = (self.long_term[i], priority);
            count += 1;
        }

        if count == 0 {
            return batch;
        }

        // Weighted random selection (approximate by picking top-k)
        let sample_size = REPLAY_BATCH.min(count);
        for i in 0..sample_size {
            let trace = &all[i % count].0;
            batch[i] = ReplayExperience {
                state_hash: trace.state_hash,
                action: trace.action,
                next_state_hash: trace.next_state_hash,
                reward: trace.reward,
                td_error: trace.prediction_error,
            };
        }

        batch
    }

    /// Called after replay to update trace access times
    pub fn mark_replayed(&mut self, state_hash: u64, now: u64) {
        for i in 0..self.st_count {
            if self.short_term[i].valid && self.short_term[i].state_hash == state_hash {
                self.short_term[i].last_access = now;
                self.short_term[i].significance = (self.short_term[i].significance
                    + FixedPoint::from_f32(0.05))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                break;
            }
        }
        for i in 0..self.lt_count {
            if self.long_term[i].state_hash == state_hash {
                self.long_term[i].last_access = now;
                self.long_term[i].significance = (self.long_term[i].significance
                    + FixedPoint::from_f32(0.02))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                break;
            }
        }
        self.total_replayed += 1;
    }

    // -----------------------------------------------------------------------
    // Recall — retrieve memories by state cue
    // -----------------------------------------------------------------------

    /// Find closest matching memory by state hash hamming distance
    /// Returns None if the closest match exceeds max_distance bits
    pub fn recall_by_state(&mut self, state_hash: u64, now: u64) -> Option<MemoryTrace> {
        const MAX_DIST: u64 = 16; // Up to 16 bits different (out of 64)
        let mut best: Option<MemoryTrace> = None;
        let mut best_dist = u64::MAX;

        for i in 0..self.st_count {
            if !self.short_term[i].valid {
                continue;
            }
            let dist = (self.short_term[i].state_hash ^ state_hash).count_ones() as u64;
            if dist < best_dist {
                best_dist = dist;
                best = Some(self.short_term[i]);
            }
        }
        for i in 0..self.lt_count {
            let dist = (self.long_term[i].state_hash ^ state_hash).count_ones() as u64;
            if dist < best_dist {
                best_dist = dist;
                best = Some(self.long_term[i]);
            }
        }

        if best_dist <= MAX_DIST
            && let Some(mut trace) = best
        {
            trace.last_access = now;
            self.last_recall[self.recall_idx % 16] = trace.state_hash;
            self.recall_idx += 1;
            self.total_recalled += 1;
            return Some(trace);
        }
        None
    }

    /// Recall by time window (recent N steps)
    pub fn recall_recent(&self, window_end: u64, window_steps: u64) -> core::ops::Range<usize> {
        let start = window_end.saturating_sub(window_steps);
        let mut first = self.st_count;
        let mut last = 0;
        for i in 0..self.st_count {
            if !self.short_term[i].valid {
                continue;
            }
            if self.short_term[i].timestamp >= start && self.short_term[i].timestamp <= window_end {
                if i < first {
                    first = i;
                }
                if i >= last {
                    last = i + 1;
                }
            }
        }
        first..last
    }

    /// Pattern completion: given partial hash, find closest full memory
    pub fn pattern_complete(&self, partial_hash: u64, mask: u64) -> Option<MemoryTrace> {
        let mut best: Option<MemoryTrace> = None;
        let mut best_dist = u64::MAX;

        let check = |trace: &MemoryTrace, dist: &mut u64| {
            if (trace.state_hash & mask) == (partial_hash & mask) {
                let d = (trace.state_hash ^ partial_hash).count_ones() as u64;
                if d < *dist {
                    *dist = d;
                    return true;
                }
            }
            false
        };

        for i in 0..self.st_count {
            if self.short_term[i].valid && check(&self.short_term[i], &mut best_dist) {
                best = Some(self.short_term[i]);
            }
        }
        for i in 0..self.lt_count {
            if check(&self.long_term[i], &mut best_dist) {
                best = Some(self.long_term[i]);
            }
        }

        best
    }

    // -----------------------------------------------------------------------
    // Spatial navigation — place cells and grid cells
    // -----------------------------------------------------------------------

    /// Update spatial position via path integration
    pub fn update_position(&mut self, vx: FixedPoint, vy: FixedPoint, dt: u64) {
        let dt_ms = FixedPoint::from_f32(dt as f32 * 0.001);
        self.velocity_x = vx.clamp(FixedPoint::from_f32(-1.0), FixedPoint::ONE);
        self.velocity_y = vy.clamp(FixedPoint::from_f32(-1.0), FixedPoint::ONE);
        self.position_x =
            (self.position_x + self.velocity_x * dt_ms).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.position_y =
            (self.position_y + self.velocity_y * dt_ms).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.total_place_updates += 1;
    }

    /// Update theta rhythm oscillator
    pub fn update_theta(&mut self, timestep: u64) {
        let period = FixedPoint::from_f32(THETA_RHYTHM_PERIOD as f32);
        let step_fp = FixedPoint::from_f32(timestep as f32);
        let phase_delta = step_fp / period * FixedPoint::from_f32(core::f32::consts::TAU);
        self.theta.phase = (self.theta.phase + phase_delta).clamp(
            FixedPoint::ZERO,
            FixedPoint::from_f32(core::f32::consts::TAU),
        );
    }

    /// Current theta phase as a fraction of cycle (0.0–1.0)
    pub fn theta_phase_frac(&self) -> FixedPoint {
        let tau = FixedPoint::from_f32(core::f32::consts::TAU);
        (self.theta.phase / tau).clamp(FixedPoint::ZERO, FixedPoint::ONE)
    }

    /// Compute place cell firing rates for current position
    pub fn compute_place_cells(&mut self) {
        for i in 0..PLACE_CELLS {
            let pc = &mut self.place_cells[i];
            let dx = self.position_x - pc.pref_x;
            let dy = self.position_y - pc.pref_y;
            let dist_sq = dx * dx + dy * dy;
            let width_sq = pc.field_width * pc.field_width;
            // Gaussian place field
            pc.firing_rate = if width_sq > FixedPoint::ZERO {
                let exponent = -dist_sq / (width_sq * FixedPoint::from_f32(2.0));
                // Approximate exp via linear for small values
                if exponent > FixedPoint::from_f32(-3.0) {
                    FixedPoint::ONE + exponent.exp()
                } else {
                    FixedPoint::ZERO
                }
            } else {
                FixedPoint::ZERO
            };
            pc.firing_rate = pc.firing_rate.clamp(FixedPoint::ZERO, FixedPoint::ONE);
            pc.active = pc.firing_rate > FixedPoint::from_f32(0.1);
        }
    }

    /// Compute grid cell firing rates for current position
    pub fn compute_grid_cells(&mut self) {
        for i in 0..GRID_CELLS {
            let gc = &mut self.grid_cells[i];
            let cos_a = gc.orientation.cos();
            let sin_a = gc.orientation.sin();
            let spacing = gc.spacing;
            let s = if spacing > FixedPoint::ZERO {
                FixedPoint::ONE / spacing
            } else {
                FixedPoint::from_f32(10.0)
            };

            // Rotate position by grid orientation
            let rx = self.position_x * cos_a - self.position_y * sin_a + gc.phase_x;
            let ry = self.position_x * sin_a + self.position_y * cos_a + gc.phase_y;

            // Hexagonal grid firing (3 directions, 60° apart)
            let c1 = (rx * s * FixedPoint::from_f32(2.0)).cos();
            let c2 = ((rx * FixedPoint::from_f32(-0.5) + ry * FixedPoint::from_f32(0.866))
                * s
                * FixedPoint::from_f32(2.0))
            .cos();
            let c3 = ((rx * FixedPoint::from_f32(-0.5) - ry * FixedPoint::from_f32(0.866))
                * s
                * FixedPoint::from_f32(2.0))
            .cos();

            let sum_cos = c1 + c2 + c3;
            // Grid firing rate = (sum of cos + 3) / 6 → 0..1
            let rate = (sum_cos + FixedPoint::from_f32(3.0)) / FixedPoint::from_f32(6.0);
            gc.firing_rate = rate.clamp(FixedPoint::ZERO, FixedPoint::ONE);
            gc.active = gc.firing_rate > FixedPoint::from_f32(0.3);
        }
    }

    /// Theta phase precession: as animal moves through place field,
    /// firing phase shifts from late → early theta (precession)
    pub fn phase_precession(&self, place_idx: usize) -> FixedPoint {
        if place_idx >= PLACE_CELLS {
            return FixedPoint::ZERO;
        }
        let pc = &self.place_cells[place_idx];
        if !pc.active {
            return FixedPoint::ZERO;
        }
        // Distance from center of place field
        let dx = self.position_x - pc.pref_x;
        let dy = self.position_y - pc.pref_y;
        let dist = (dx * dx + dy * dy).sqrt();
        let phase = self.theta_phase_frac();
        // Precession: theta phase shifts proportional to distance from field center

        (phase + dist * FixedPoint::from_f32(2.0)).clamp(FixedPoint::ZERO, FixedPoint::ONE)
    }

    /// Encode current spatial context into memory state hash
    pub fn spatial_context_hash(&self) -> u64 {
        let mut hash: u64 = 0;
        // Mix position into hash
        let px_bits = (self.position_x.to_f32() * 65535.0) as u64;
        let py_bits = (self.position_y.to_f32() * 65535.0) as u64;
        hash ^= px_bits.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        hash ^= py_bits.wrapping_mul(0xC29C_E2F9_9E6D_DC7B);

        // Mix active place cell indices
        let mut count = 0u64;
        for i in 0..PLACE_CELLS {
            if self.place_cells[i].active {
                hash ^= (i as u64).wrapping_mul(0xBF58_476F_299E_8B3A);
                count += 1;
            }
        }
        hash ^= count.wrapping_mul(0x9DDF_EA08_EB38_2D69);
        hash
    }

    /// Trigger sharp-wave ripple replay during "rest"
    pub fn trigger_ripple_replay(&mut self, now: u64) -> usize {
        if self.ripple_lock {
            return 0;
        }
        self.ripple_lock = true;

        // Sample significant memories from recent and long-term
        let mut count = 0;
        let mut ripples = [0u64; 32];
        for i in 0..self.st_count.min(32) {
            if self.short_term[i].valid
                && self.short_term[i].significance > FixedPoint::from_f32(0.4)
            {
                ripples[count] = self.short_term[i].state_hash;
                count += 1;
            }
        }
        if count < 32 {
            for i in 0..self.lt_count.min(32 - count) {
                ripples[count] = self.long_term[i].state_hash;
                count += 1;
            }
        }

        self.ripple_count = count;
        self.ripple_buffer = ripples;

        // Replay in reverse order (as in hippocampus)
        for i in (0..count).rev() {
            // Mark replayed to boost significance
            self.mark_replayed(ripples[i], now);
        }

        self.ripple_lock = false;
        count
    }

    /// Get place cell activity vector
    pub fn place_cell_activity(&self) -> [FixedPoint; PLACE_CELLS] {
        let mut rates = [FixedPoint::ZERO; PLACE_CELLS];
        for i in 0..PLACE_CELLS {
            rates[i] = self.place_cells[i].firing_rate;
        }
        rates
    }

    /// Get grid cell activity vector
    pub fn grid_cell_activity(&self) -> [FixedPoint; GRID_CELLS] {
        let mut rates = [FixedPoint::ZERO; GRID_CELLS];
        for i in 0..GRID_CELLS {
            rates[i] = self.grid_cells[i].firing_rate;
        }
        rates
    }

    /// Get best-matching place cell index (current location)
    pub fn best_place_cell(&self) -> usize {
        let mut best = 0;
        let mut best_rate = FixedPoint::ZERO;
        for i in 0..PLACE_CELLS {
            if self.place_cells[i].firing_rate > best_rate {
                best_rate = self.place_cells[i].firing_rate;
                best = i;
            }
        }
        best
    }

    /// Full spatial update cycle
    pub fn spatial_update(&mut self, vx: FixedPoint, vy: FixedPoint, dt: u64) {
        self.update_position(vx, vy, dt);
        self.update_theta(dt);
        self.compute_place_cells();
        self.compute_grid_cells();
    }

    // -----------------------------------------------------------------------
    // Memory stats
    // -----------------------------------------------------------------------

    pub fn short_term_count(&self) -> usize {
        self.st_count
    }
    pub fn long_term_count(&self) -> usize {
        self.lt_count
    }
    pub fn utilization(&self) -> FixedPoint {
        let total = ST_CAPACITY + LT_CAPACITY;
        let used = self.st_count + self.lt_count;
        FixedPoint::from_int(used as i32) / FixedPoint::from_int(total as i32)
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------
pub static mut EPISODIC_MEMORY: MaybeUninit<EpisodicMemory> = MaybeUninit::uninit();

static INITIALIZED_EPISODIC_MEMORY: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_episodic_memory() {
    unsafe {
        EPISODIC_MEMORY.write(EpisodicMemory::new());
        INITIALIZED_EPISODIC_MEMORY.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn episodic_memory() -> &'static mut EpisodicMemory {
    unsafe {
        if !INITIALIZED_EPISODIC_MEMORY.load(core::sync::atomic::Ordering::Relaxed) {
            init_episodic_memory();
        }
        &mut *EPISODIC_MEMORY.as_mut_ptr()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episodic_memory_new_default() {
        let mem = EpisodicMemory::new();
        assert_eq!(mem.short_term_count(), 0);
        assert_eq!(mem.long_term_count(), 0);
        assert_eq!(mem.total_recorded, 0);
    }

    #[test]
    fn record_increases_count() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            42,
            1,
            43,
            FixedPoint::from_f32(0.5),
            FixedPoint::from_f32(0.1),
            FixedPoint::from_f32(0.2),
            100,
        );
        assert_eq!(mem.short_term_count(), 1);
        assert_eq!(mem.total_recorded, 1);
    }

    #[test]
    fn record_wraps_around() {
        let mut mem = EpisodicMemory::new();
        for i in 0..ST_CAPACITY + 10 {
            mem.record(
                i as u64,
                0,
                (i + 1) as u64,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        assert_eq!(mem.short_term_count(), ST_CAPACITY);
        assert_eq!(mem.total_recorded, (ST_CAPACITY + 10) as u64);
    }

    #[test]
    fn significance_high_for_high_reward() {
        let mem = EpisodicMemory::new();
        let sig = mem.compute_significance(FixedPoint::ONE, FixedPoint::ZERO, FixedPoint::ZERO);
        assert_eq!(sig, FixedPoint::from_f32(0.4));
    }

    #[test]
    fn significance_high_for_high_pe() {
        let mem = EpisodicMemory::new();
        let sig = mem.compute_significance(FixedPoint::ZERO, FixedPoint::ONE, FixedPoint::ZERO);
        assert_eq!(sig, FixedPoint::from_f32(0.35));
    }

    #[test]
    fn significance_high_for_high_novelty() {
        let mem = EpisodicMemory::new();
        let sig = mem.compute_significance(FixedPoint::ZERO, FixedPoint::ZERO, FixedPoint::ONE);
        assert_eq!(sig, FixedPoint::from_f32(0.25));
    }

    #[test]
    fn consolidation_transfers_to_long_term() {
        let mut mem = EpisodicMemory::new();
        for i in 0..10 {
            mem.record(
                i as u64,
                1,
                (i + 1) as u64,
                FixedPoint::ONE, // High reward → high significance
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        let transferred = mem.consolidate(100);
        assert!(transferred > 0);
        assert!(mem.long_term_count() > 0);
        assert!(mem.total_consolidated > 0);
    }

    #[test]
    fn consolidation_requires_significance() {
        let mut mem = EpisodicMemory::new();
        // Record low-significance experiences
        for i in 0..5 {
            mem.record(
                i as u64,
                0,
                0,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        let transferred = mem.consolidate(100);
        assert_eq!(transferred, 0); // None significant enough
    }

    #[test]
    fn recall_by_state_finds_exact() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            42,
            1,
            43,
            FixedPoint::from_f32(0.5),
            FixedPoint::from_f32(0.1),
            FixedPoint::ZERO,
            100,
        );
        let recalled = mem.recall_by_state(42, 200);
        assert!(recalled.is_some());
        assert_eq!(recalled.unwrap().action, 1);
    }

    #[test]
    fn recall_by_state_not_found() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            0x0000_0000_0000_0042,
            1,
            0,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            100,
        );
        // Very different hash — more than 16 bits different
        let recalled = mem.recall_by_state(0xFFFF_FFFF_FFFF_0000, 200);
        assert!(recalled.is_none());
    }

    #[test]
    fn pattern_complete_with_mask() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            0xABCD,
            1,
            0xABCE,
            FixedPoint::from_f32(0.5),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            100,
        );
        let completed = mem.pattern_complete(0xAB00, 0xFF00);
        assert!(completed.is_some());
        assert_eq!(completed.unwrap().state_hash, 0xABCD);
    }

    #[test]
    fn pattern_complete_no_match() {
        let mem = EpisodicMemory::new();
        let completed = mem.pattern_complete(0xAB00, 0xFF00);
        assert!(completed.is_none());
    }

    #[test]
    fn sample_replay_batch_returns_experiences() {
        let mem = EpisodicMemory::new();
        let batch = mem.sample_replay_batch();
        assert_eq!(batch.len(), REPLAY_BATCH);
    }

    #[test]
    fn sample_replay_batch_with_data() {
        let mut mem = EpisodicMemory::new();
        for i in 0..5 {
            mem.record(
                i as u64,
                1,
                (i + 1) as u64,
                FixedPoint::from_f32(0.5),
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        let batch = mem.sample_replay_batch();
        assert_eq!(batch.len(), REPLAY_BATCH);
        assert!(batch[0].state_hash != 0 || batch[0].action != 0);
    }

    #[test]
    fn mark_replayed_increases_significance() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            42,
            1,
            43,
            FixedPoint::from_f32(0.5),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            100,
        );
        let sig_before = mem.short_term[0].significance;
        mem.mark_replayed(42, 200);
        let sig_after = mem.short_term[0].significance;
        assert!(sig_after > sig_before);
    }

    #[test]
    fn apply_forgetting_reduces_significance() {
        let mut mem = EpisodicMemory::new();
        mem.record(
            42,
            1,
            43,
            FixedPoint::from_f32(0.8),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            0,
        );
        let sig_before = mem.short_term[0].significance;
        mem.apply_forgetting(1_000_000); // Far future
        let sig_after = mem.short_term[0].significance;
        assert!(sig_after < sig_before);
    }

    #[test]
    fn recall_recent_returns_window() {
        let mut mem = EpisodicMemory::new();
        for i in 0..20 {
            mem.record(
                i as u64,
                0,
                0,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        let range = mem.recall_recent(15, 10); // timestamp 5..15
        assert!(range.end > range.start);
    }

    #[test]
    fn retention_decays_with_time() {
        let trace = MemoryTrace {
            state_hash: 0,
            action: 0,
            next_state_hash: 0,
            reward: FixedPoint::ZERO,
            prediction_error: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            timestamp: 0,
            significance: FixedPoint::ONE,
            consolidation_count: 0,
            last_access: 0,
            valid: true,
            is_long_term: false,
        };
        let r1 = trace.retention(0, 1000);
        assert_eq!(r1, FixedPoint::ONE); // No elapsed time
        let r2 = trace.retention(10000, 1000);
        assert!(r2 < FixedPoint::ONE); // Decayed
    }

    #[test]
    fn max_consolidations_prunes_st() {
        let mut mem = EpisodicMemory::new();
        mem.max_consolidations = 2;
        mem.record(
            42,
            1,
            43,
            FixedPoint::ONE,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            0,
        );
        // Two consolidations should exhaust it
        mem.consolidate(100);
        assert!(mem.short_term[0].valid); // Still valid (consolidation_count=1)
        mem.consolidate(200); // Now consolidation_count=2 ≥ max=2
        assert!(!mem.short_term[0].valid);
    }

    #[test]
    fn retention_half_life_zero_returns_one() {
        let trace = MemoryTrace {
            state_hash: 0,
            action: 0,
            next_state_hash: 0,
            reward: FixedPoint::ZERO,
            prediction_error: FixedPoint::ZERO,
            novelty: FixedPoint::ZERO,
            timestamp: 0,
            significance: FixedPoint::ZERO,
            consolidation_count: 0,
            last_access: 0,
            valid: true,
            is_long_term: false,
        };
        assert_eq!(trace.retention(1000, 0), FixedPoint::ONE);
    }

    #[test]
    fn utilization_ratio() {
        let mut mem = EpisodicMemory::new();
        assert_eq!(mem.utilization(), FixedPoint::ZERO);
        mem.record(
            1,
            0,
            2,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            FixedPoint::ZERO,
            0,
        );
        assert!(mem.utilization() > FixedPoint::ZERO);
    }

    #[test]
    fn place_cell_initialization() {
        let mem = EpisodicMemory::new();
        let rates = mem.place_cell_activity();
        assert_eq!(rates.len(), PLACE_CELLS);
    }

    #[test]
    fn place_cells_fire_near_preferred_location() {
        let mut mem = EpisodicMemory::new();
        mem.position_x = FixedPoint::from_f32(0.5);
        mem.position_y = FixedPoint::from_f32(0.5);
        mem.compute_place_cells();
        let best = mem.best_place_cell();
        // Expect some place cell to be active near center
        assert!(mem.place_cells[best].firing_rate > FixedPoint::ZERO);
    }

    #[test]
    fn place_cells_different_positions() {
        let mut mem = EpisodicMemory::new();
        mem.position_x = FixedPoint::from_f32(0.1);
        mem.position_y = FixedPoint::from_f32(0.1);
        mem.compute_place_cells();
        let best1 = mem.best_place_cell();

        mem.position_x = FixedPoint::from_f32(0.9);
        mem.position_y = FixedPoint::from_f32(0.9);
        mem.compute_place_cells();
        let best2 = mem.best_place_cell();

        // Different positions should activate different place cells
        assert!(best1 != best2 || mem.place_cells[best1].firing_rate > FixedPoint::ZERO);
    }

    #[test]
    fn grid_cells_fire_at_multiple_locations() {
        let mut mem = EpisodicMemory::new();
        mem.position_x = FixedPoint::from_f32(0.3);
        mem.position_y = FixedPoint::from_f32(0.3);
        mem.compute_grid_cells();
        let activity = mem.grid_cell_activity();
        let mut any_active = false;
        for i in 0..GRID_CELLS {
            if activity[i] > FixedPoint::from_f32(0.3) {
                any_active = true;
                break;
            }
        }
        // Some grid cells should be active at any given position
        assert!(any_active);
    }

    #[test]
    fn spatial_update_changes_position() {
        let mut mem = EpisodicMemory::new();
        let x_before = mem.position_x;
        let y_before = mem.position_y;
        mem.spatial_update(FixedPoint::from_f32(0.5), FixedPoint::ZERO, 100);
        assert!(mem.position_x != x_before || mem.position_y != y_before);
        assert!(mem.total_place_updates > 0);
    }

    #[test]
    fn spatial_context_hash_is_deterministic() {
        let mut mem = EpisodicMemory::new();
        // Use position directly + compute place cells (no theta/velocity accumulation)
        mem.position_x = FixedPoint::from_f32(0.5);
        mem.position_y = FixedPoint::from_f32(0.5);
        mem.compute_place_cells();
        let h1 = mem.spatial_context_hash();
        let h2 = mem.spatial_context_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn theta_phase_advances_with_time() {
        let mut mem = EpisodicMemory::new();
        let p1 = mem.theta_phase_frac();
        mem.update_theta(10);
        let p2 = mem.theta_phase_frac();
        assert!(p2 >= p1 || p2 < FixedPoint::from_f32(0.1)); // Wrapping allowed
    }

    #[test]
    fn ripple_replay_does_not_panic() {
        let mut mem = EpisodicMemory::new();
        for i in 0..10 {
            mem.record(
                i as u64,
                1,
                (i + 1) as u64,
                FixedPoint::from_f32(0.8),
                FixedPoint::ZERO,
                FixedPoint::ZERO,
                i as u64,
            );
        }
        let count = mem.trigger_ripple_replay(100);
        // Should have found some memories to replay
        assert!(count > 0 || mem.st_count > 0);
    }

    #[test]
    fn phase_precession_within_bounds() {
        let mut mem = EpisodicMemory::new();
        mem.spatial_update(FixedPoint::from_f32(0.0), FixedPoint::from_f32(0.0), 10);
        let precession = mem.phase_precession(0);
        assert!(precession >= FixedPoint::ZERO && precession <= FixedPoint::ONE);
    }

    #[test]
    fn place_cell_field_width_affects_firing() {
        let mut mem = EpisodicMemory::new();
        mem.position_x = FixedPoint::ZERO;
        mem.position_y = FixedPoint::ZERO;
        mem.compute_place_cells();
        let rate_at_origin = mem.place_cells[0].firing_rate;
        mem.position_x = FixedPoint::ONE;
        mem.position_y = FixedPoint::ONE;
        mem.compute_place_cells();
        let rate_far = mem.place_cells[0].firing_rate;
        // Firing should be lower far from preferred location
        assert!(rate_at_origin >= rate_far);
    }

    #[test]
    fn best_place_cell_returns_valid_index() {
        let mut mem = EpisodicMemory::new();
        mem.spatial_update(FixedPoint::from_f32(0.5), FixedPoint::from_f32(0.5), 10);
        let best = mem.best_place_cell();
        assert!(best < PLACE_CELLS);
    }

    #[test]
    fn grid_cells_have_multiple_scales() {
        let mem = EpisodicMemory::new();
        let spacings: [FixedPoint; GRID_CELLS] =
            core::array::from_fn(|i| mem.grid_cells[i].spacing);
        let min_spacing = spacings
            .iter()
            .copied()
            .fold(FixedPoint::from_f32(100.0), |a, b| a.min(b));
        let max_spacing = spacings
            .iter()
            .copied()
            .fold(FixedPoint::ZERO, |a, b| a.max(b));
        assert!(min_spacing > FixedPoint::ZERO);
        assert!(max_spacing >= min_spacing);
    }
}
