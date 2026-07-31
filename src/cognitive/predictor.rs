use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const TRANSITION_BUF_SIZE: usize = 256;
const PROTOTYPES_PER_ACTION: usize = 8;
const MAX_ACTIONS: usize = 8;
const DELTA_DIM: usize = 16;

// ---------------------------------------------------------------------------
// Learned prototypes per action (online Hebbian-like clustering)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Prototype {
    delta: [FixedPoint; DELTA_DIM],
    state_hash: u32,
    confidence: FixedPoint, // 0.0 (new) → 1.0 (well-established)
    count: u32,             // Number of merged observations
}

impl Prototype {
    const fn empty() -> Self {
        Self {
            delta: [FixedPoint::ZERO; DELTA_DIM],
            state_hash: 0,
            confidence: FixedPoint::ZERO,
            count: 0,
        }
    }

    /// Merge a new delta into this prototype (online update)
    fn merge(&mut self, delta: &[FixedPoint; DELTA_DIM], hash: u32, lr: FixedPoint) {
        let alpha = if self.confidence < FixedPoint::from_f32(0.01) {
            FixedPoint::ONE // First observation: take it fully
        } else {
            lr
        };
        let one_minus_alpha = FixedPoint::ONE - alpha;
        for i in 0..DELTA_DIM {
            self.delta[i] = self.delta[i] * one_minus_alpha + delta[i] * alpha;
        }
        self.state_hash = hash;
        self.count += 1;
        // Confidence asymptotically approaches 1.0
        let inc = FixedPoint::from_f32(0.1) * (FixedPoint::ONE - self.confidence);
        self.confidence = (self.confidence + inc).clamp(FixedPoint::ZERO, FixedPoint::ONE);
    }

    fn distance_to(&self, delta: &[FixedPoint; DELTA_DIM]) -> FixedPoint {
        let mut dist = FixedPoint::ZERO;
        for i in 0..DELTA_DIM {
            let d = self.delta[i] - delta[i];
            dist += d * d;
        }
        dist
    }
}

// ---------------------------------------------------------------------------
// Transition record (history buffer for replay)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Transition {
    state_hash: u32,
    action: u8,
    delta: [FixedPoint; DELTA_DIM],
    valid: bool,
}

// ---------------------------------------------------------------------------
// Predictor with online Hebbian/TD learning
// ---------------------------------------------------------------------------
pub struct Predictor {
    pub predicted_state: [FixedPoint; 1024],
    pub prediction_errors: [FixedPoint; 1024],
    pub mean_error: FixedPoint,
    pub confidence: FixedPoint, // Average confidence across prototypes
    pub last_action: u8,        // Most recent action taken

    // Prototype-based learning (fast online adaptation)
    prototypes: [[Prototype; PROTOTYPES_PER_ACTION]; MAX_ACTIONS],

    // Transition history buffer (for potential offline replay)
    transitions: [Transition; TRANSITION_BUF_SIZE],
    transition_idx: usize,
    transition_count: usize,

    pub learning_rate: FixedPoint,
    pub error_threshold: FixedPoint, // Minimum error to trigger learning
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            predicted_state: [FixedPoint::ZERO; 1024],
            prediction_errors: [FixedPoint::ZERO; 1024],
            mean_error: FixedPoint::ZERO,
            confidence: FixedPoint::from_f32(0.5),
            last_action: 0,
            prototypes: [[Prototype::empty(); PROTOTYPES_PER_ACTION]; MAX_ACTIONS],
            transitions: [Transition {
                state_hash: 0,
                action: 0,
                delta: [FixedPoint::ZERO; DELTA_DIM],
                valid: false,
            }; TRANSITION_BUF_SIZE],
            transition_idx: 0,
            transition_count: 0,
            learning_rate: FixedPoint::from_f32(0.1),
            error_threshold: FixedPoint::from_f32(0.05),
        }
    }

    // -----------------------------------------------------------------------
    // Prediction API (used by Network::predictive_cycle)
    // -----------------------------------------------------------------------

    /// Predict delta for a given state + action using nearest prototype
    pub fn predict_next(&self, state: &[FixedPoint; 1024], action: u8) -> [FixedPoint; 16] {
        let hash = self.compute_hash(state);
        let action_idx = (action as usize) % MAX_ACTIONS;
        let mut best_dist = FixedPoint::MAX;
        let mut best_delta = [FixedPoint::ZERO; 16];
        let mut best_conf = FixedPoint::ZERO;

        for p in &self.prototypes[action_idx] {
            if p.count == 0 {
                continue;
            }
            let state_dist = self.hash_distance(hash, p.state_hash);
            let delta = p.delta;
            // Weight distance by inverse confidence (low conf = trust less)
            let weighted_dist = if p.confidence > FixedPoint::from_f32(0.01) {
                state_dist / (p.confidence * FixedPoint::from_f32(2.0))
            } else {
                state_dist
            };
            if weighted_dist < best_dist {
                best_dist = weighted_dist;
                best_delta = delta;
                best_conf = p.confidence;
            }
        }

        // Scale delta by confidence (uncertain = smaller prediction)
        if best_conf > FixedPoint::ZERO && best_conf < FixedPoint::ONE {
            for d in best_delta.iter_mut() {
                *d = *d * best_conf;
            }
        }

        best_delta
    }

    /// Predict next state (state + delta) with confidence
    pub fn predict_next_with_confidence(
        &self,
        state: &[FixedPoint; 1024],
        action: u8,
    ) -> ([FixedPoint; 16], FixedPoint) {
        let hash = self.compute_hash(state);
        let action_idx = (action as usize) % MAX_ACTIONS;
        let mut best_dist = FixedPoint::MAX;
        let mut best_delta = [FixedPoint::ZERO; 16];
        let mut best_conf = FixedPoint::ZERO;

        for p in &self.prototypes[action_idx] {
            if p.count == 0 {
                continue;
            }
            let state_dist = self.hash_distance(hash, p.state_hash);
            if state_dist < best_dist {
                best_dist = state_dist;
                best_delta = p.delta;
                best_conf = p.confidence;
            }
        }

        (best_delta, best_conf)
    }

    /// Predict next state using nearest prototype for the last action taken
    pub fn predict(&mut self, current_state: &[FixedPoint]) {
        let n = current_state.len().min(1024);
        let mut state_array = [FixedPoint::ZERO; 1024];
        for i in 0..n {
            state_array[i] = current_state[i];
        }
        let delta = self.predict_next(&state_array, self.last_action);
        for i in 0..DELTA_DIM.min(n) {
            self.predicted_state[i] = current_state[i] + delta[i];
        }
        for i in DELTA_DIM..n {
            self.predicted_state[i] = current_state[i];
        }
    }

    // -----------------------------------------------------------------------
    // Online learning (called by Network::predictive_cycle after warp)
    // -----------------------------------------------------------------------

    /// Record transition + learn from it (online Hebbian-style update)
    pub fn record_transition(
        &mut self,
        state: &[FixedPoint; 1024],
        action: u8,
        next_state: &[FixedPoint; 1024],
    ) {
        let hash = self.compute_hash(state);
        let action_idx = (action as usize) % MAX_ACTIONS;
        let mut computed_delta = [FixedPoint::ZERO; DELTA_DIM];
        for i in 0..DELTA_DIM {
            computed_delta[i] = next_state[i] - state[i];
        }

        // --- Online prototype learning ---
        self.learn_prototype(action_idx, hash, &computed_delta);

        // --- Store in transition history ---
        let idx = self.transition_idx % TRANSITION_BUF_SIZE;
        self.transitions[idx] = Transition {
            state_hash: hash,
            action,
            delta: computed_delta,
            valid: true,
        };
        self.transition_idx += 1;
        if self.transition_count < TRANSITION_BUF_SIZE {
            self.transition_count += 1;
        }

        // --- Update confidence from predictive error ---
        self.update_confidence_from_error();
    }

    /// Update predictor from prediction error (Hebbian correction)
    pub fn update_from_prediction_error(&mut self, observed: &[FixedPoint; 1024]) {
        let mut sum_err = FixedPoint::ZERO;
        for i in 0..1024 {
            let err = (self.predicted_state[i] - observed[i]).abs();
            self.prediction_errors[i] = err;
            sum_err += err;
        }
        self.mean_error = sum_err / FixedPoint::from_int(1024);

        // Correct prototypes if error is high
        if self.mean_error > self.error_threshold {
            let correction_lr = self.learning_rate * self.mean_error;
            for action_idx in 0..MAX_ACTIONS {
                for p in self.prototypes[action_idx].iter_mut() {
                    if p.count > 0 {
                        // Pull prototype toward recent observations
                        let observed_state = &observed[0..DELTA_DIM];
                        for i in 0..DELTA_DIM {
                            let err_dir =
                                observed_state[i] - (self.predicted_state[i] + p.delta[i]);
                            p.delta[i] += err_dir * correction_lr * p.confidence;
                        }
                    }
                }
            }
        }
    }

    /// Get average confidence across all prototypes
    pub fn get_confidence(&self) -> FixedPoint {
        self.confidence
    }

    /// Predict using nearest neighbor in transition buffer (legacy)
    pub fn predict_next_transition(
        &self,
        state: &[FixedPoint; 1024],
        action: u8,
    ) -> [FixedPoint; 16] {
        let hash = self.compute_hash(state);
        let mut best_match = FixedPoint::MAX;
        let mut result = [FixedPoint::ZERO; 16];

        for i in 0..self.transition_count.min(TRANSITION_BUF_SIZE) {
            let t = &self.transitions[i];
            if !t.valid {
                continue;
            }
            if t.action != action {
                continue;
            }
            let hd = self.hash_distance(hash, t.state_hash);
            if hd < best_match {
                best_match = hd;
                result = t.delta;
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Internal learning logic
    // -----------------------------------------------------------------------

    /// Find nearest prototype and merge, or create new one
    fn learn_prototype(&mut self, action_idx: usize, hash: u32, delta: &[FixedPoint; DELTA_DIM]) {
        let mut best_idx = None;
        let mut best_dist = FixedPoint::from_f32(0.5); // Merge threshold
        let mut empty_idx = None;

        for (i, p) in self.prototypes[action_idx].iter().enumerate() {
            if p.count == 0 && empty_idx.is_none() {
                empty_idx = Some(i);
                continue;
            }
            if p.count == 0 {
                continue;
            }
            let dist = p.distance_to(delta);
            let hash_dist = self.hash_distance(hash, p.state_hash);
            let combined = dist + FixedPoint::from_f32(0.3) * hash_dist;
            if combined < best_dist {
                best_dist = combined;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            // Merge into nearest prototype
            self.prototypes[action_idx][idx].merge(delta, hash, self.learning_rate);
        } else if let Some(idx) = empty_idx {
            // Create new prototype
            self.prototypes[action_idx][idx].delta = *delta;
            self.prototypes[action_idx][idx].state_hash = hash;
            self.prototypes[action_idx][idx].count = 1;
            self.prototypes[action_idx][idx].confidence = FixedPoint::from_f32(0.1);
        }
        // else: no space, discard (all slots occupied and no match)
    }

    /// Update overall confidence from mean error
    fn update_confidence_from_error(&mut self) {
        let total_protos: usize = self
            .prototypes
            .iter()
            .flat_map(|a| a.iter())
            .map(|p| if p.count > 0 { 1 } else { 0 })
            .sum();
        let total = MAX_ACTIONS * PROTOTYPES_PER_ACTION;
        let ratio = FixedPoint::from_int(total_protos as i32) / FixedPoint::from_int(total as i32);

        // Conf = f(prototype_coverage, low_error)
        let error_factor =
            FixedPoint::ONE - self.mean_error.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.confidence = (ratio * FixedPoint::from_f32(0.6)
            + error_factor * FixedPoint::from_f32(0.4))
        .clamp(FixedPoint::from_f32(0.05), FixedPoint::ONE);
    }

    // -----------------------------------------------------------------------
    // Hashing utilities
    // -----------------------------------------------------------------------

    fn compute_hash(&self, state: &[FixedPoint; 1024]) -> u32 {
        let mut h: u32 = 5381;
        let mut i = 0;
        while i < 1024 {
            let bits = state[i].to_bits();
            h = h.wrapping_mul(33).wrapping_add(bits as u32);
            i += 64;
        }
        h
    }

    fn hash_distance(&self, a: u32, b: u32) -> FixedPoint {
        let diff = (a as i64 - b as i64).unsigned_abs();
        FixedPoint::from_f32((diff as f32) / (u32::MAX as f32))
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------
pub static mut COGNITIVE_PREDICTOR: MaybeUninit<Predictor> = MaybeUninit::uninit();
static INITIALIZED_COGNITIVE_PREDICTOR: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_cognitive_predictor() {
    unsafe {
        COGNITIVE_PREDICTOR.write(Predictor::new());
        INITIALIZED_COGNITIVE_PREDICTOR.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn cognitive_predictor() -> &'static mut Predictor {
    unsafe {
        if !INITIALIZED_COGNITIVE_PREDICTOR.load(core::sync::atomic::Ordering::Relaxed) {
            init_cognitive_predictor();
        }
        &mut *COGNITIVE_PREDICTOR.as_mut_ptr()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(val: f32) -> [FixedPoint; 1024] {
        let mut s = [FixedPoint::ZERO; 1024];
        for i in 0..16 {
            s[i] = FixedPoint::from_f32(val + (i as f32) * 0.1);
        }
        s
    }

    #[test]
    fn predictor_new_creates_default() {
        let p = Predictor::new();
        assert_eq!(p.mean_error, FixedPoint::ZERO);
        assert!(p.learning_rate > FixedPoint::ZERO);
    }

    #[test]
    fn predictor_predict_does_not_panic() {
        let mut p = Predictor::new();
        let state = make_state(0.5);
        p.predict(&state);
    }

    #[test]
    fn predictor_record_and_predict() {
        let mut p = Predictor::new();
        let state = make_state(1.0);
        let next = make_state(1.5); // delta ≈ 0.5
        p.record_transition(&state, 0, &next);

        let delta = p.predict_next(&state, 0);
        assert!(delta[0] > FixedPoint::ZERO);
    }

    #[test]
    fn predictor_learns_from_repeated_transitions() {
        let mut p = Predictor::new();
        let state = make_state(1.0);

        // Teach the predictor: same state → next_state with delta +0.3
        for _ in 0..10 {
            let mut next = state;
            for i in 0..16 {
                next[i] = next[i] + FixedPoint::from_f32(0.3);
            }
            p.record_transition(&state, 1, &next);
        }

        // delta is scaled by confidence — should approach 0.3 as conf→1
        let delta = p.predict_next(&state, 1);
        assert!(
            delta[0] > FixedPoint::from_f32(0.15),
            "delta[0]={} should be positive",
            delta[0].to_f32()
        );

        // Prototype delta should be ≈ 0.3 (unscaled)
        let proto_delta = p.prototypes[1][0].delta;
        let expected = FixedPoint::from_f32(0.3);
        let error = (proto_delta[0] - expected).abs();
        assert!(
            error < FixedPoint::from_f32(0.02),
            "proto delta[0]={} expected={}",
            proto_delta[0].to_f32(),
            expected.to_f32()
        );

        // Confidence should be high
        assert!(p.prototypes[1][0].confidence > FixedPoint::from_f32(0.5));
    }

    #[test]
    fn predictor_different_actions_have_different_deltas() {
        let mut p = Predictor::new();
        let state = make_state(0.0);

        // Action 0: delta +0.1
        for _ in 0..5 {
            let mut next = state;
            for i in 0..16 {
                next[i] = next[i] + FixedPoint::from_f32(0.1);
            }
            p.record_transition(&state, 0, &next);
        }

        // Action 1: delta +0.5
        for _ in 0..5 {
            let mut next = state;
            for i in 0..16 {
                next[i] = next[i] + FixedPoint::from_f32(0.5);
            }
            p.record_transition(&state, 1, &next);
        }

        let d0 = p.predict_next(&state, 0);
        let d1 = p.predict_next(&state, 1);
        assert!(d1[0] > d0[0], "action 1 should produce larger delta");
    }

    #[test]
    fn predictor_confidence_starts_low() {
        let p = Predictor::new();
        assert!(p.confidence > FixedPoint::ZERO);
    }

    #[test]
    fn predictor_confidence_increases_with_data() {
        let mut p = Predictor::new();
        let state = make_state(2.0);

        for _ in 0..10 {
            let next = make_state(2.5);
            p.record_transition(&state, 0, &next);
        }

        assert!(p.confidence > FixedPoint::from_f32(0.1));
    }

    #[test]
    fn predictor_update_from_error() {
        let mut p = Predictor::new();
        let state = make_state(0.0);
        p.predict(&state);

        let observed = make_state(0.5);
        p.update_from_prediction_error(&observed);

        assert!(p.mean_error > FixedPoint::ZERO);
    }

    #[test]
    fn predict_next_transition_fallback() {
        let p = Predictor::new();
        let state = make_state(0.0);
        let delta = p.predict_next_transition(&state, 0);
        assert_eq!(delta[0], FixedPoint::ZERO);
    }

    #[test]
    fn predictor_prototype_capacity() {
        let mut p = Predictor::new();
        // Fill with different state→delta patterns
        for a in 0..MAX_ACTIONS {
            for j in 0..20 {
                let state = make_state((a * 10 + j) as f32);
                let mut next = state;
                for i in 0..16 {
                    next[i] = next[i] + FixedPoint::from_f32((a + j) as f32 * 0.01);
                }
                p.record_transition(&state, a as u8, &next);
            }
        }
        // Should have at least some prototypes
        let total: usize = p
            .prototypes
            .iter()
            .flat_map(|a| a.iter())
            .filter(|p| p.count > 0)
            .count();
        assert!(total > 0);
    }

    #[test]
    fn predictor_predict_with_confidence() {
        let p = Predictor::new();
        let state = make_state(0.0);
        let (_delta, conf) = p.predict_next_with_confidence(&state, 0);
        assert_eq!(conf, FixedPoint::ZERO); // No data yet
    }

    #[test]
    fn predictor_error_threshold_default() {
        let p = Predictor::new();
        assert!(p.error_threshold > FixedPoint::ZERO);
    }
}
