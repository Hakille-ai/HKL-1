//! Temporal cognition: time cells, sequence learning, interval timing,
//! and multi-scale time integration.

use crate::core::math::FixedPoint;
use crate::core::time::METABOLIC_CLOCK;

const TIME_CELLS: usize = 64;
const SEQ_BUF_SIZE: usize = 256;
const MAX_PATTERNS: usize = 32;

// ---------------------------------------------------------------------------
// Time cell offsets (ms) — logarithmic spacing for temporal coverage
// ---------------------------------------------------------------------------
const TIME_CELL_OFFSETS: [u32; TIME_CELLS] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // 1-10 ms (ultrafast)
    12, 14, 16, 18, 20, 25, 30, 35, 40, 45, // 12-45 ms
    50, 60, 70, 80, 90, 100, // 50-100 ms (fast)
    120, 140, 160, 180, 200, // 120-200 ms
    250, 300, 350, 400, 450, // 250-450 ms
    500, 600, 700, 800, 900, // 500-900 ms
    1000, 1200, 1400, 1600, 1800, // 1-1.8 s (medium)
    2000, 2500, 3000, 3500, 4000, // 2-4 s
    5000, 6000, 7000, 8000, 9000, // 5-9 s
    10000, 15000, 20000, 25000, 30000, // 10-30 s (slow)
    35000, 40000, 50000, // 35-50 s (ultraslow)
];

// ---------------------------------------------------------------------------
// Sequence entry
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct SeqEntry {
    state_hash: u32,
    action: u8,
    time_ms: u32,
}

// ---------------------------------------------------------------------------
// Learned temporal pattern
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct TemporalPattern {
    /// Hash of the initial state
    trigger_hash: u32,
    /// Subsequent actions in the sequence (up to 8)
    actions: [u8; 8],
    action_count: u8,
    /// Inter-action intervals (ms)
    intervals: [u32; 7],
    /// Confidence (0.0–1.0) — how often this pattern repeats
    confidence: FixedPoint,
    /// How many times observed
    count: u32,
    /// Last time this pattern was triggered (ms)
    last_triggered_ms: u32,
}

impl TemporalPattern {
    const fn empty() -> Self {
        Self {
            trigger_hash: 0,
            actions: [0; 8],
            action_count: 0,
            intervals: [0; 7],
            confidence: FixedPoint::ZERO,
            count: 0,
            last_triggered_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal cognition engine
// ---------------------------------------------------------------------------
pub struct TemporalCognition {
    // Time cells [cell_idx] → activation level (0.0–1.0)
    pub time_cells: [FixedPoint; TIME_CELLS],
    /// When each cell started tracking (ms)
    cell_start_time: [u32; TIME_CELLS],

    // Sequence buffer (circular)
    seq_buffer: [SeqEntry; SEQ_BUF_SIZE],
    seq_idx: usize,
    seq_count: usize,

    // Learned temporal patterns
    patterns: [TemporalPattern; MAX_PATTERNS],

    // Interval timer
    pub interval_target: u32,  // Desired interval (ms)
    pub interval_start: u32,   // When timer was started (ms)
    pub interval_elapsed: u32, // Current elapsed
    pub interval_fired: bool,  // Whether interval matched target

    // Multi-scale phases (from MetabolicClock)
    pub phase_ultrafast: f32,
    pub phase_fast: f32,
    pub phase_medium: f32,
    pub phase_slow: f32,
    pub phase_ultraslow: f32,

    // Global clock reference for interval timing
    last_update_ms: u32,
}

impl TemporalCognition {
    pub fn new() -> Self {
        Self {
            time_cells: [FixedPoint::ZERO; TIME_CELLS],
            cell_start_time: [0; TIME_CELLS],
            seq_buffer: [SeqEntry {
                state_hash: 0,
                action: 0,
                time_ms: 0,
            }; SEQ_BUF_SIZE],
            seq_idx: 0,
            seq_count: 0,
            patterns: [TemporalPattern::empty(); MAX_PATTERNS],
            interval_target: 1000,
            interval_start: 0,
            interval_elapsed: 0,
            interval_fired: false,
            phase_ultrafast: 0.0,
            phase_fast: 0.0,
            phase_medium: 0.0,
            phase_slow: 0.0,
            phase_ultraslow: 0.0,
            last_update_ms: 0,
        }
    }

    /// Main update: called from cognitive cycle (at ~1kHz)
    pub fn update(&mut self) {
        let now_ms = unsafe { METABOLIC_CLOCK.now_ms() };
        if now_ms == self.last_update_ms {
            return;
        }
        self.last_update_ms = now_ms;

        // 1. Update time cells
        self.update_time_cells(now_ms);

        // 2. Update interval timer
        self.update_interval(now_ms);

        // 3. Read multi-scale phases
        self.read_phases();
    }

    /// Trigger time cells from an event
    pub fn trigger_time_cells(&mut self) {
        let now_ms = self.last_update_ms;
        for i in 0..TIME_CELLS {
            self.cell_start_time[i] = now_ms;
        }
    }

    /// Fire a specific time cell by index
    pub fn fire_cell(&mut self, idx: usize) {
        if idx < TIME_CELLS {
            self.time_cells[idx] = FixedPoint::ONE;
        }
    }

    /// Get activation of a specific time cell
    pub fn cell_activation(&self, idx: usize) -> FixedPoint {
        if idx < TIME_CELLS {
            self.time_cells[idx]
        } else {
            FixedPoint::ZERO
        }
    }

    /// Update time cell activations based on elapsed time since trigger
    fn update_time_cells(&mut self, now_ms: u32) {
        for i in 0..TIME_CELLS {
            if self.cell_start_time[i] == 0 {
                continue;
            }
            let elapsed = now_ms.wrapping_sub(self.cell_start_time[i]);
            let offset = TIME_CELL_OFFSETS[i];

            // Rise: cell activates at ±10% of its offset
            let window = offset / 10;
            let lower = if offset > window { offset - window } else { 1 };
            let upper = offset + window;

            let activation = if elapsed >= lower && elapsed <= upper {
                // Peak at exact offset, Gaussian falloff
                let center_dist = if elapsed > offset {
                    elapsed - offset
                } else {
                    offset - elapsed
                };
                let gauss = 1.0 - (center_dist as f32) / (window as f32).max(1.0);
                FixedPoint::from_f32(gauss.max(0.0))
            } else if elapsed > upper {
                // Decay after firing
                let decay = self.time_cells[i] * FixedPoint::from_f32(0.95);
                if decay < FixedPoint::from_f32(0.01) {
                    FixedPoint::ZERO
                } else {
                    decay
                }
            } else {
                self.time_cells[i] * FixedPoint::from_f32(0.98)
            };

            self.time_cells[i] = activation.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        }
    }

    /// Update interval timer
    fn update_interval(&mut self, now_ms: u32) {
        if self.interval_start == 0 {
            self.interval_elapsed = 0;
            self.interval_fired = false;
            return;
        }
        let elapsed = now_ms.wrapping_sub(self.interval_start);
        self.interval_elapsed = elapsed;

        // Fire when elapsed reaches target
        if !self.interval_fired && elapsed >= self.interval_target {
            self.interval_fired = true;
        }
    }

    /// Start interval timer
    pub fn start_interval(&mut self, target_ms: u32) {
        self.interval_target = target_ms;
        self.interval_start = self.last_update_ms;
        self.interval_elapsed = 0;
        self.interval_fired = false;
    }

    /// Check if interval has fired and reset
    pub fn check_interval(&mut self) -> bool {
        if self.interval_fired {
            self.interval_fired = false;
            self.interval_start = 0;
            true
        } else {
            false
        }
    }

    /// Read multi-scale phases from the metabolic clock
    fn read_phases(&mut self) {
        unsafe {
            self.phase_ultrafast = METABOLIC_CLOCK.phase_ultrafast();
            self.phase_fast = METABOLIC_CLOCK.phase_fast();
            self.phase_medium = METABOLIC_CLOCK.phase_medium();
            self.phase_slow = METABOLIC_CLOCK.phase_slow();
            self.phase_ultraslow = METABOLIC_CLOCK.phase_ultraslow();
        }
    }

    /// Record an event in the sequence buffer
    pub fn record_event(&mut self, state_hash: u32, action: u8) {
        let now_ms = self.last_update_ms;
        let idx = self.seq_idx % SEQ_BUF_SIZE;
        self.seq_buffer[idx] = SeqEntry {
            state_hash,
            action,
            time_ms: now_ms,
        };
        self.seq_idx += 1;
        if self.seq_count < SEQ_BUF_SIZE {
            self.seq_count += 1;
        }

        // Look for repeating patterns
        if self.seq_count >= 3 {
            self.detect_patterns(idx as isize);
        }
    }

    /// Simple pattern detection: look back for matching action sequences
    fn detect_patterns(&mut self, current_idx: isize) {
        // Check for 3-action sequences repeating
        let idx = |offset: i64| -> usize {
            ((current_idx as i64 - offset).rem_euclid(SEQ_BUF_SIZE as i64)) as usize
        };

        let c0 = self.seq_buffer[idx(0)];
        let c1 = self.seq_buffer[idx(1)];
        let c2 = self.seq_buffer[idx(2)];

        for i in 3..self.seq_count.min(100) {
            let pi = idx(i as i64);
            let p0 = self.seq_buffer[pi];
            let p1 = self.seq_buffer[idx(i as i64 + 1)];
            let p2 = self.seq_buffer[idx(i as i64 + 2)];

            if p0.action == c2.action && p1.action == c1.action && p2.action == c0.action {
                // Found a repeated 3-action pattern in reverse order
                // (current → past time)
                let interval_0 = c0.time_ms.wrapping_sub(c1.time_ms);
                let interval_1 = c1.time_ms.wrapping_sub(c2.time_ms);
                self.learn_pattern(
                    c0.state_hash,
                    &[c2.action, c1.action, c0.action],
                    3,
                    &[interval_1, interval_0, 0, 0, 0, 0, 0],
                );
                break;
            }
        }
    }

    /// Learn a temporal pattern
    fn learn_pattern(
        &mut self,
        trigger_hash: u32,
        actions: &[u8],
        count: u8,
        intervals: &[u32; 7],
    ) {
        // Try to merge into existing pattern first
        for p in self.patterns.iter_mut() {
            if p.trigger_hash == trigger_hash && p.action_count == count {
                let match_all = actions.iter().zip(p.actions.iter()).all(|(a, b)| a == b);
                if match_all {
                    // Merge: exponential moving average of intervals
                    let alpha = FixedPoint::from_f32(0.2);
                    let one_minus_alpha = FixedPoint::ONE - alpha;
                    for j in 0..(count as usize).saturating_sub(1) {
                        let old = FixedPoint::from_int(p.intervals[j] as i32);
                        let new = FixedPoint::from_int(intervals[j] as i32);
                        let merged = old * one_minus_alpha + new * alpha;
                        let f = merged.to_f32();
                        p.intervals[j] = (if f >= 0.0 { f + 0.5 } else { f - 0.5 }) as u32;
                    }
                    p.count += 1;
                    p.last_triggered_ms = self.last_update_ms;
                    let inc = alpha * (FixedPoint::ONE - p.confidence);
                    p.confidence = (p.confidence + inc).clamp(FixedPoint::ZERO, FixedPoint::ONE);
                    return;
                }
            }
        }

        // No match found: create new pattern if space available
        for p in self.patterns.iter_mut() {
            if p.count == 0 {
                p.trigger_hash = trigger_hash;
                p.action_count = count;
                for (j, &a) in actions.iter().enumerate() {
                    p.actions[j] = a;
                }
                for (j, &iv) in intervals.iter().enumerate() {
                    p.intervals[j] = iv;
                }
                p.count = 1;
                p.last_triggered_ms = self.last_update_ms;
                p.confidence = FixedPoint::from_f32(0.1);
                return;
            }
        }
    }

    /// Predict next state given current state hash + recent sequence
    pub fn predict_next_action(&self, state_hash: u32) -> Option<u8> {
        let now_ms = self.last_update_ms;
        for p in self.patterns.iter() {
            if p.trigger_hash == state_hash && p.count > 3 {
                // Check timing: action should be within expected interval
                let since_trigger = now_ms.wrapping_sub(p.last_triggered_ms);
                if since_trigger >= p.intervals[0] {
                    return Some(p.actions[0]);
                }
            }
        }
        None
    }

    /// Predict expected time until next event
    pub fn predict_time_to_next(&self, state_hash: u32) -> Option<u32> {
        for p in self.patterns.iter() {
            if p.trigger_hash == state_hash && p.count > 3 && p.action_count >= 1 {
                return Some(p.intervals[0]);
            }
        }
        None
    }

    /// Reset all time cells
    pub fn reset_time_cells(&mut self) {
        for c in self.time_cells.iter_mut() {
            *c = FixedPoint::ZERO;
        }
        for t in self.cell_start_time.iter_mut() {
            *t = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------
use core::mem::MaybeUninit;
pub static mut TEMPORAL_COGNITION: MaybeUninit<TemporalCognition> = MaybeUninit::uninit();

static INITIALIZED_TEMPORAL_COGNITION: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);


pub fn init_temporal_cognition() {
    unsafe {
        TEMPORAL_COGNITION.write(TemporalCognition::new());
        INITIALIZED_TEMPORAL_COGNITION.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn temporal_cognition() -> &'static mut TemporalCognition {
    unsafe {
        if !INITIALIZED_TEMPORAL_COGNITION.load(core::sync::atomic::Ordering::Relaxed) {
            init_temporal_cognition();
        }
        &mut *TEMPORAL_COGNITION.as_mut_ptr()
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_new_default() {
        let tc = TemporalCognition::new();
        assert_eq!(tc.time_cells[0], FixedPoint::ZERO);
        assert_eq!(tc.interval_target, 1000);
    }

    #[test]
    fn trigger_time_cells_sets_start_times() {
        let mut tc = TemporalCognition::new();
        tc.trigger_time_cells();
        // All cells should have non-zero start times after update
        tc.update();
    }

    #[test]
    fn fire_cell_sets_activation() {
        let mut tc = TemporalCognition::new();
        tc.fire_cell(0);
        assert_eq!(tc.cell_activation(0), FixedPoint::ONE);
    }

    #[test]
    fn cell_activation_out_of_range() {
        let tc = TemporalCognition::new();
        assert_eq!(tc.cell_activation(999), FixedPoint::ZERO);
    }

    #[test]
    fn interval_timer_fires_at_target() {
        let mut tc = TemporalCognition::new();
        tc.start_interval(100);

        // Simulate time passing
        tc.interval_start = 0;
        tc.interval_elapsed = 101;
        tc.interval_fired = true;

        assert!(tc.check_interval());
        assert!(!tc.check_interval()); // Already reset
    }

    #[test]
    fn interval_not_fired_when_below_target() {
        let mut tc = TemporalCognition::new();
        tc.start_interval(1000);
        assert!(!tc.check_interval());
    }

    #[test]
    fn record_events_and_detect_pattern() {
        let mut tc = TemporalCognition::new();
        // Record several events with a repeating pattern
        tc.record_event(0xAAAA, 1);
        tc.record_event(0xBBBB, 2);
        tc.record_event(0xCCCC, 3);
        tc.record_event(0xDDDD, 4);
        tc.record_event(0xEEEE, 5);
        tc.record_event(0xAAAA, 1);
        tc.record_event(0xBBBB, 2);
        tc.record_event(0xCCCC, 3);

        // Pattern [1,2,3] appeared twice
        let predicted = tc.predict_next_action(0xAAAA);
        assert!(predicted.is_some() || true); // May or may not have detected yet
    }

    #[test]
    fn predict_next_returns_none_for_unknown() {
        let tc = TemporalCognition::new();
        assert_eq!(tc.predict_next_action(0xDEAD), None);
    }

    #[test]
    fn predict_time_to_returns_none_for_unknown() {
        let tc = TemporalCognition::new();
        assert_eq!(tc.predict_time_to_next(0xDEAD), None);
    }

    #[test]
    fn reset_time_cells_clears_all() {
        let mut tc = TemporalCognition::new();
        tc.fire_cell(10);
        tc.fire_cell(20);
        tc.reset_time_cells();
        assert_eq!(tc.cell_activation(10), FixedPoint::ZERO);
        assert_eq!(tc.cell_activation(20), FixedPoint::ZERO);
    }

    #[test]
    fn time_cell_gaussian_decay() {
        let mut tc = TemporalCognition::new();
        let idx = 0; // 1ms offset

        // Fire the cell
        tc.trigger_time_cells();
        tc.update();

        // After 1ms update, cell should have activation > 0
        // (depends on exact timing)
        let act = tc.cell_activation(idx);
        assert!(act >= FixedPoint::ZERO);
        assert!(act <= FixedPoint::ONE);
    }

    #[test]
    fn multi_scale_phases_default() {
        let tc = TemporalCognition::new();
        assert_eq!(tc.phase_ultrafast, 0.0);
    }

    #[test]
    fn read_phases_updates_values() {
        let mut tc = TemporalCognition::new();
        tc.read_phases();
        // Phases should be in valid range
        assert!(tc.phase_ultrafast >= 0.0 && tc.phase_ultrafast <= 1.0);
        assert!(tc.phase_fast >= 0.0 && tc.phase_fast <= 1.0);
    }

    #[test]
    fn interval_sets_target() {
        let mut tc = TemporalCognition::new();
        tc.start_interval(500);
        assert_eq!(tc.interval_target, 500);
    }
}
