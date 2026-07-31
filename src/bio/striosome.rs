use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const STRIOSOME_COUNT: usize = 16;
const MATRIX_COMPARTMENTS: usize = 16;
const DA_THRESHOLD: FixedPoint = FixedPoint::from_bits(13107); // ~0.2
const LEARNING_RATE: FixedPoint = FixedPoint::from_bits(6554); // ~0.1
const MODULE_NEURONS: usize = 32;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Striosome {
    pub id: u8,
    pub dopamine_sensitivity: FixedPoint,
    pub learning_gate: FixedPoint,
    pub action_bias: FixedPoint,
    pub activity: FixedPoint,
    pub plastic_weights: [FixedPoint; MODULE_NEURONS],
    pub active: bool,
}

impl Striosome {
    pub const fn new(id: u8) -> Self {
        Self {
            id,
            dopamine_sensitivity: FixedPoint::from_f32(0.5),
            learning_gate: FixedPoint::ZERO,
            action_bias: FixedPoint::ZERO,
            activity: FixedPoint::ZERO,
            plastic_weights: [FixedPoint::from_f32(0.5); MODULE_NEURONS],
            active: false,
        }
    }

    pub fn step(&mut self, dopamine: FixedPoint, input: &[FixedPoint]) {
        self.learning_gate = if dopamine > DA_THRESHOLD {
            (dopamine - DA_THRESHOLD) * self.dopamine_sensitivity
        } else {
            FixedPoint::ZERO
        };
        if self.learning_gate > FixedPoint::ONE {
            self.learning_gate = FixedPoint::ONE;
        }
        let mut act = FixedPoint::ZERO;
        for (w, &inp) in self.plastic_weights.iter().zip(input.iter()) {
            act += *w * inp;
        }
        self.activity = (act / FixedPoint::from_int(MODULE_NEURONS as i32))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.active = self.activity > FixedPoint::from_bits(6554);
    }

    pub fn learn(&mut self, td_error: FixedPoint, input: &[FixedPoint]) {
        if self.learning_gate > FixedPoint::ZERO {
            for (w, &inp) in self.plastic_weights.iter_mut().zip(input.iter()) {
                let delta = LEARNING_RATE * td_error * inp * self.learning_gate;
                *w = (*w + delta).clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
        }
    }

    pub fn modulate_weights(&self, weights: &mut [FixedPoint], offset: usize) {
        for (i, w) in weights.iter_mut().enumerate() {
            if i >= offset && (i - offset) < MODULE_NEURONS {
                *w *= FixedPoint::ONE + self.action_bias * FixedPoint::from_bits(6554);
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MatrixCompartment {
    pub id: u8,
    pub striosome_affinity: FixedPoint,
    pub inhibition_strength: FixedPoint,
    pub activity: FixedPoint,
    pub lateral_inhibition: FixedPoint,
    pub active: bool,
}

impl MatrixCompartment {
    pub const fn new(id: u8) -> Self {
        Self {
            id,
            striosome_affinity: FixedPoint::from_f32(0.3),
            inhibition_strength: FixedPoint::from_f32(0.2),
            activity: FixedPoint::ZERO,
            lateral_inhibition: FixedPoint::ZERO,
            active: false,
        }
    }

    pub fn step(&mut self, striosome_activity: FixedPoint, input_activity: FixedPoint) {
        let excitation = input_activity * (FixedPoint::ONE - self.striosome_affinity);
        let inhibition = striosome_activity * self.inhibition_strength;
        self.activity =
            (excitation - inhibition + FixedPoint::ONE).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.active = self.activity > FixedPoint::from_bits(6554);
        self.lateral_inhibition = self.activity * FixedPoint::from_bits(9830);
    }

    pub fn compete(&self, other_activity: FixedPoint) -> FixedPoint {
        let diff = self.activity - other_activity;
        if diff > FixedPoint::ZERO {
            diff * FixedPoint::from_f32(0.8)
        } else {
            FixedPoint::ZERO
        }
    }
}

#[repr(C)]
pub struct StriosomeMatrixSystem {
    pub striosomes: [Striosome; STRIOSOME_COUNT],
    pub matrix: [MatrixCompartment; MATRIX_COMPARTMENTS],
    pub selected_action: u8,
    pub competition_winner: u8,
}

impl StriosomeMatrixSystem {
    pub const fn new() -> Self {
        let mut s = [Striosome::new(0); STRIOSOME_COUNT];
        let mut i = 0;
        while i < STRIOSOME_COUNT {
            s[i] = Striosome::new(i as u8);
            i += 1;
        }
        let mut m = [MatrixCompartment::new(0); MATRIX_COMPARTMENTS];
        let mut j = 0;
        while j < MATRIX_COMPARTMENTS {
            m[j] = MatrixCompartment::new(j as u8);
            j += 1;
        }
        Self {
            striosomes: s,
            matrix: m,
            selected_action: 0,
            competition_winner: 0,
        }
    }

    pub fn step_all(&mut self, dopamine: FixedPoint, sensory_input: &[FixedPoint]) {
        for (i, striosome) in self.striosomes.iter_mut().enumerate() {
            let start = (i * MODULE_NEURONS) % sensory_input.len();
            let end = core::cmp::min(start + MODULE_NEURONS, sensory_input.len());
            if end > start {
                striosome.step(dopamine, &sensory_input[start..end]);
            } else {
                striosome.step(dopamine, &[]);
            }
        }
        for (i, comp) in self.matrix.iter_mut().enumerate() {
            let s_idx = i % STRIOSOME_COUNT;
            comp.step(self.striosomes[s_idx].activity, FixedPoint::from_f32(0.3));
        }
        self.competition();
    }

    pub fn learn_all(&mut self, td_error: FixedPoint, sensory_input: &[FixedPoint]) {
        for (i, striosome) in self.striosomes.iter_mut().enumerate() {
            let start = (i * MODULE_NEURONS) % sensory_input.len();
            let end = core::cmp::min(start + MODULE_NEURONS, sensory_input.len());
            if end > start {
                striosome.learn(td_error, &sensory_input[start..end]);
            }
        }
    }

    pub fn competition(&mut self) {
        let mut max_activity = FixedPoint::ZERO;
        let mut winner = 0u8;
        for (i, striosome) in self.striosomes.iter().enumerate() {
            if striosome.activity > max_activity {
                max_activity = striosome.activity;
                winner = i as u8;
            }
        }
        self.competition_winner = winner;
        for comp in self.matrix.iter_mut() {
            if comp.active {
                let compete_val = comp.compete(max_activity);
                if compete_val > FixedPoint::ZERO {
                    max_activity -= compete_val;
                }
            }
        }
        if max_activity > FixedPoint::from_bits(6554) {
            self.selected_action = winner;
        }
    }

    pub fn winning_striosome(&self) -> Option<&Striosome> {
        if self.competition_winner < STRIOSOME_COUNT as u8 {
            Some(&self.striosomes[self.competition_winner as usize])
        } else {
            None
        }
    }

    pub fn dopamine_gate_open(&self) -> bool {
        self.striosomes
            .iter()
            .any(|s| s.learning_gate > FixedPoint::ZERO)
    }

    pub fn matrix_activation_level(&self) -> FixedPoint {
        let mut sum = FixedPoint::ZERO;
        for comp in &self.matrix {
            sum += comp.activity;
        }
        sum / FixedPoint::from_int(MATRIX_COMPARTMENTS as i32)
    }
}

pub static mut STRIOSOME_MATRIX: MaybeUninit<StriosomeMatrixSystem> = MaybeUninit::uninit();

pub fn init_striosome_matrix() {
    unsafe {
        STRIOSOME_MATRIX = MaybeUninit::new(StriosomeMatrixSystem::new());
    }
}

pub fn striosome_matrix() -> &'static mut StriosomeMatrixSystem {
    unsafe { STRIOSOME_MATRIX.as_mut_ptr().as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_striosome_new() {
        let s = Striosome::new(0);
        assert_eq!(s.id, 0);
        assert_eq!(s.activity, FixedPoint::ZERO);
        assert!(!s.active);
    }

    #[test]
    fn test_striosome_step_high_da() {
        let mut s = Striosome::new(0);
        let input = [FixedPoint::from_f32(0.8); MODULE_NEURONS];
        let da = FixedPoint::from_f32(0.5);
        s.step(da, &input);
        assert!(s.activity > FixedPoint::ZERO);
        assert!(s.learning_gate > FixedPoint::ZERO);
    }

    #[test]
    fn test_striosome_step_low_da() {
        let mut s = Striosome::new(0);
        let input = [FixedPoint::from_f32(0.8); MODULE_NEURONS];
        s.step(FixedPoint::ZERO, &input);
        assert_eq!(s.learning_gate, FixedPoint::ZERO);
    }

    #[test]
    fn test_striosome_learn() {
        let mut s = Striosome::new(0);
        s.learning_gate = FixedPoint::ONE;
        let input = [FixedPoint::from_f32(1.0); MODULE_NEURONS];
        let old_w = s.plastic_weights[0];
        s.learn(FixedPoint::from_f32(0.5), &input);
        assert!(s.plastic_weights[0] != old_w);
    }

    #[test]
    fn test_matrix_compartment_new() {
        let m = MatrixCompartment::new(0);
        assert_eq!(m.activity, FixedPoint::ZERO);
    }

    #[test]
    fn test_matrix_compartment_step() {
        let mut m = MatrixCompartment::new(0);
        m.step(FixedPoint::from_f32(0.5), FixedPoint::from_f32(0.8));
        assert!(m.activity > FixedPoint::ZERO);
    }

    #[test]
    fn test_matrix_competition() {
        let mut m1 = MatrixCompartment::new(0);
        let mut m2 = MatrixCompartment::new(1);
        m1.activity = FixedPoint::from_f32(0.8);
        m2.activity = FixedPoint::from_f32(0.3);
        let result = m1.compete(m2.activity);
        assert!(result > FixedPoint::ZERO);
    }

    #[test]
    fn test_system_new() {
        let sys = StriosomeMatrixSystem::new();
        assert_eq!(sys.striosomes.len(), STRIOSOME_COUNT);
        assert_eq!(sys.matrix.len(), MATRIX_COMPARTMENTS);
    }

    #[test]
    fn test_system_step_all() {
        let mut sys = StriosomeMatrixSystem::new();
        let input = [FixedPoint::from_f32(0.6); 64];
        sys.step_all(FixedPoint::from_f32(0.5), &input);
        assert!(sys.competition_winner < STRIOSOME_COUNT as u8);
    }

    #[test]
    fn test_system_learn_all() {
        let mut sys = StriosomeMatrixSystem::new();
        sys.striosomes[0].learning_gate = FixedPoint::ONE;
        let input = [FixedPoint::from_f32(0.8); 64];
        let old_w = sys.striosomes[0].plastic_weights[0];
        sys.learn_all(FixedPoint::from_f32(0.3), &input);
        assert!(sys.striosomes[0].plastic_weights[0] != old_w);
    }

    #[test]
    fn test_winning_striosome() {
        let mut sys = StriosomeMatrixSystem::new();
        sys.striosomes[2].activity = FixedPoint::from_f32(0.9);
        sys.competition_winner = 2;
        let w = sys.winning_striosome();
        assert!(w.is_some());
        assert_eq!(w.unwrap().id, 2);
    }

    #[test]
    fn test_dopamine_gate() {
        let mut sys = StriosomeMatrixSystem::new();
        assert!(!sys.dopamine_gate_open());
        sys.striosomes[0].learning_gate = FixedPoint::from_f32(0.1);
        assert!(sys.dopamine_gate_open());
    }

    #[test]
    fn test_matrix_activation_level() {
        let mut sys = StriosomeMatrixSystem::new();
        for comp in sys.matrix.iter_mut() {
            comp.activity = FixedPoint::from_f32(0.5);
        }
        assert_eq!(sys.matrix_activation_level(), FixedPoint::from_f32(0.5));
    }

    #[test]
    fn test_init() {
        init_striosome_matrix();
        let sys = striosome_matrix();
        assert_eq!(sys.striosomes.len(), STRIOSOME_COUNT);
    }
}
