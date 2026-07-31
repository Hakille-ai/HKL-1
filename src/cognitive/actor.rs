use crate::cognitive::predictor::Predictor;
use crate::core::math::FixedPoint;
use crate::core::memory::NeuronId;
use crate::snn::network::{PredictorNetwork, SimulationResult};

pub const MAX_HYPOTHESES: usize = 8;
const VALUE_BUCKETS: usize = 64;
const GAMMA: f32 = 0.95;
const ALPHA: f32 = 0.1;

pub struct ActorCritic {
    pub motors: [motor::MotorOutput; 256],
    pub hypotheses: [Hypothesis; MAX_HYPOTHESES],
    pub selected_action: Option<u8>,
    pub action_confidence: FixedPoint,
    pub cycle_active: bool,
    pub cycle_result: SimulationResult,
    pub td_error: FixedPoint,
    pub reward: FixedPoint,
    pub state_value: FixedPoint,
    pub last_state_hash: u32,
    pub last_action: u8,
    value_table: [FixedPoint; VALUE_BUCKETS],
    pub episode_count: u32,
    pub epsilon: FixedPoint,
}

#[derive(Clone, Copy)]
pub struct Hypothesis {
    pub action: u8,
    pub predicted_outcome: SimulationResult,
    pub confidence: FixedPoint,
    pub simulated: bool,
}

impl Hypothesis {
    pub const fn empty() -> Self {
        Self {
            action: 0,
            predicted_outcome: SimulationResult::Neutral,
            confidence: FixedPoint::ZERO,
            simulated: false,
        }
    }
}

impl ActorCritic {
    pub fn new() -> Self {
        Self {
            motors: [motor::MotorOutput::default(); 256],
            hypotheses: [Hypothesis::empty(); MAX_HYPOTHESES],
            selected_action: None,
            action_confidence: FixedPoint::ZERO,
            cycle_active: false,
            cycle_result: SimulationResult::Neutral,
            td_error: FixedPoint::ZERO,
            reward: FixedPoint::ZERO,
            state_value: FixedPoint::ZERO,
            last_state_hash: 0,
            last_action: 0,
            value_table: [FixedPoint::ZERO; VALUE_BUCKETS],
            episode_count: 0,
            epsilon: FixedPoint::from_f32(0.3),
        }
    }

    pub fn step(&mut self) {
        for i in 0..256 {
            let id = NeuronId::new(i as u16);
            let state = crate::core::memory::neuron_state_ref(id);
            if state.layer == 4 {
                self.motors[i] = motor::MotorOutput {
                    value: state.membrane_potential,
                    timestamp: unsafe { crate::core::time::METABOLIC_CLOCK.ticks_1khz() },
                };
            }
        }
    }

    fn hash_state(&self, state: &[FixedPoint; 1024]) -> u32 {
        let mut h: u32 = 5381;
        let mut i = 0;
        while i < 1024 {
            let bits = state[i].to_bits();
            h = h.wrapping_mul(33).wrapping_add(bits as u32);
            i += 64;
        }
        h
    }

    fn value_index(&self, hash: u32) -> usize {
        (hash as usize) % VALUE_BUCKETS
    }

    pub fn compute_reward(
        &self,
        pred_error: FixedPoint,
        novelty: FixedPoint,
        energy: FixedPoint,
    ) -> FixedPoint {
        let mut r = FixedPoint::ZERO;
        r = r - pred_error * FixedPoint::from_f32(0.5);
        r = r + novelty * FixedPoint::from_f32(0.3);
        r = r - (FixedPoint::ONE - energy) * FixedPoint::from_f32(0.2);
        r
    }

    pub fn compute_td_error(&mut self, state: &[FixedPoint; 1024], reward: FixedPoint) {
        let hash = self.hash_state(state);
        let idx = self.value_index(hash);
        let current_v = self.value_table[idx];
        let next_v = current_v * FixedPoint::from_f32(GAMMA);
        self.td_error = reward + next_v - current_v;
        self.state_value = current_v;

        let delta = self.td_error * FixedPoint::from_f32(ALPHA);
        self.value_table[idx] = current_v + delta;

        self.last_state_hash = hash;
        self.reward = reward;
    }

    pub fn update_value_from_next(&mut self, next_state: &[FixedPoint; 1024]) {
        let next_hash = self.hash_state(next_state);
        let next_idx = self.value_index(next_hash);
        let next_v = self.value_table[next_idx];
        let current_idx = self.value_index(self.last_state_hash);
        let current_v = self.value_table[current_idx];
        let target = self.reward + FixedPoint::from_f32(GAMMA) * next_v;
        let delta = (target - current_v) * FixedPoint::from_f32(ALPHA);
        self.value_table[current_idx] = current_v + delta;
        self.td_error = target - current_v;
        self.state_value = current_v;
    }

    pub fn generate_hypotheses(
        &mut self,
        base_action: u8,
        rng: &mut crate::core::math::XorShift64Star,
    ) {
        let mut n: usize = 0;
        if base_action < 255 {
            self.hypotheses[n] = Hypothesis {
                action: base_action,
                predicted_outcome: SimulationResult::Neutral,
                confidence: FixedPoint::from_f32(0.5),
                simulated: false,
            };
            n += 1;
        }
        if n < MAX_HYPOTHESES {
            self.hypotheses[n] = Hypothesis {
                action: base_action.wrapping_add(1),
                predicted_outcome: SimulationResult::Neutral,
                confidence: FixedPoint::from_f32(0.3),
                simulated: false,
            };
            n += 1;
        }
        if n < MAX_HYPOTHESES && base_action > 0 {
            self.hypotheses[n] = Hypothesis {
                action: base_action.wrapping_sub(1),
                predicted_outcome: SimulationResult::Neutral,
                confidence: FixedPoint::from_f32(0.3),
                simulated: false,
            };
            n += 1;
        }
        // ε-greedy exploration: add random actions
        let use_random = rng.next_f32() < self.epsilon.to_f32();
        if use_random {
            n = 1;
            let rand_action = (rng.next_u32() & 0xFF) as u8;
            self.hypotheses[0] = Hypothesis {
                action: rand_action,
                predicted_outcome: SimulationResult::Exploratory,
                confidence: FixedPoint::from_f32(0.2),
                simulated: false,
            };
        }
        while n < MAX_HYPOTHESES {
            let rand_action = (rng.next_u32() & 0xFF) as u8;
            self.hypotheses[n] = Hypothesis {
                action: rand_action,
                predicted_outcome: SimulationResult::Neutral,
                confidence: FixedPoint::from_f32(0.1),
                simulated: false,
            };
            n += 1;
        }
        // Decay epsilon
        self.epsilon *= FixedPoint::from_f32(0.999);
        self.epsilon = self.epsilon.max(FixedPoint::from_f32(0.01));
    }

    pub fn select_action(&mut self, state: &[FixedPoint; 1024]) -> Option<u8> {
        let hash = self.hash_state(state);
        let idx = self.value_index(hash);
        let base_value = self.value_table[idx];

        let mut best_idx: Option<usize> = None;
        let mut best_score = FixedPoint::ZERO;

        for (i, h) in self.hypotheses.iter().enumerate() {
            let action_val = FixedPoint::from_int(h.action as i32);
            let q_val = base_value + action_val * FixedPoint::from_f32(0.001);
            let score = q_val * h.confidence;
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            self.selected_action = Some(self.hypotheses[idx].action);
            self.action_confidence = self.hypotheses[idx].confidence;
            self.last_action = self.hypotheses[idx].action;
        }
        best_idx.map(|i| self.hypotheses[i].action)
    }

    pub fn select_best_hypothesis(&mut self) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_conf = FixedPoint::ZERO;
        for (i, h) in self.hypotheses.iter().enumerate() {
            if h.confidence > best_conf {
                best_conf = h.confidence;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            self.selected_action = Some(self.hypotheses[idx].action);
            self.action_confidence = self.hypotheses[idx].confidence;
            self.last_action = self.hypotheses[idx].action;
        }
        best_idx
    }

    pub fn test_hypothesis(
        &self,
        hypothesis: &Hypothesis,
        predictor: &Predictor,
        state: &[FixedPoint; 1024],
    ) -> FixedPoint {
        let predicted_delta = predictor.predict_next(state, hypothesis.action);
        let mut total_delta = FixedPoint::ZERO;
        for d in predicted_delta.iter().take(16) {
            total_delta += d.abs();
        }
        total_delta
    }

    pub fn validate_outcome(&mut self, predictor_net: &PredictorNetwork) {
        self.cycle_result = predictor_net.evaluate_outcome();
    }
}

use core::mem::MaybeUninit;
pub static mut COGNITIVE_ACTOR: MaybeUninit<ActorCritic> = MaybeUninit::uninit();

static INITIALIZED_COGNITIVE_ACTOR: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_cognitive_actor() {
    unsafe {
        COGNITIVE_ACTOR.write(ActorCritic::new());
        INITIALIZED_COGNITIVE_ACTOR.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn cognitive_actor() -> &'static mut ActorCritic {
    unsafe {
        if !INITIALIZED_COGNITIVE_ACTOR.load(core::sync::atomic::Ordering::Relaxed) {
            init_cognitive_actor();
        }
        &mut *COGNITIVE_ACTOR.as_mut_ptr()
    }
}

pub mod motor {
    use crate::core::math::FixedPoint;

    #[derive(Clone, Copy, Default)]
    pub struct MotorOutput {
        pub value: FixedPoint,
        pub timestamp: u32,
    }
}
