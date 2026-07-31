use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const RELAY_NUCLEI: usize = 4;
const TRN_NEURONS: usize = 16;
const BURST_THRESHOLD: FixedPoint = FixedPoint::from_bits(19661); // ~0.3
const TONIC_DRIVE: FixedPoint = FixedPoint::from_bits(6554); // ~0.1

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum FiringMode {
    Burst,
    Tonic,
    Silent,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ThalamicRelay {
    pub id: u8,
    pub mode: FiringMode,
    pub membrane_potential: FixedPoint,
    pub burst_calcium: FixedPoint,
    pub sensory_gate: FixedPoint,
    pub attention_gain: FixedPoint,
    pub output_rate: FixedPoint,
    pub active: bool,
}

impl ThalamicRelay {
    pub const fn new(id: u8) -> Self {
        Self {
            id,
            mode: FiringMode::Silent,
            membrane_potential: FixedPoint::ZERO,
            burst_calcium: FixedPoint::ZERO,
            sensory_gate: FixedPoint::ONE,
            attention_gain: FixedPoint::from_f32(0.5),
            output_rate: FixedPoint::ZERO,
            active: false,
        }
    }

    pub fn step(&mut self, sensory_input: FixedPoint, attention_signal: FixedPoint, tick: u64) {
        self.attention_gain +=
            (attention_signal - self.attention_gain) * FixedPoint::from_bits(3277);
        self.attention_gain = self.attention_gain.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        let gated_input = sensory_input
            * self.sensory_gate
            * (FixedPoint::from_f32(0.5) + self.attention_gain * FixedPoint::from_f32(0.5));
        let leak = FixedPoint::from_bits(6554);
        self.burst_calcium = if tick % 100 < 5 {
            self.burst_calcium + FixedPoint::from_bits(6554)
        } else {
            self.burst_calcium * (FixedPoint::ONE - FixedPoint::from_bits(3277))
        };
        if self.burst_calcium > FixedPoint::ONE {
            self.burst_calcium = FixedPoint::ONE;
        }
        if self.mode == FiringMode::Burst {
            let burst_boost = self.burst_calcium * FixedPoint::from_f32(2.0);
            self.membrane_potential += gated_input * burst_boost - leak * self.membrane_potential;
        } else {
            self.membrane_potential += gated_input * TONIC_DRIVE - leak * self.membrane_potential;
        }
        self.membrane_potential = self
            .membrane_potential
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        if self.membrane_potential > BURST_THRESHOLD {
            self.output_rate = self.membrane_potential;
            self.active = true;
        } else {
            self.output_rate = FixedPoint::ZERO;
            self.active = false;
        }
        if self.burst_calcium > FixedPoint::from_bits(19661) && sensory_input > FixedPoint::ZERO {
            self.mode = FiringMode::Burst;
        } else if self.attention_gain > FixedPoint::from_bits(13107) {
            self.mode = FiringMode::Tonic;
        } else {
            self.mode = FiringMode::Silent;
        }
    }

    pub fn set_gate(&mut self, open: bool) {
        self.sensory_gate = if open {
            FixedPoint::ONE
        } else {
            FixedPoint::ZERO
        };
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ThalamicReticularNucleus {
    pub neurons: [FixedPoint; TRN_NEURONS],
    pub global_inhibition: FixedPoint,
    pub attention_focus: FixedPoint,
    pub winner_idx: u8,
}

impl ThalamicReticularNucleus {
    pub const fn new() -> Self {
        Self {
            neurons: [FixedPoint::ZERO; TRN_NEURONS],
            global_inhibition: FixedPoint::ZERO,
            attention_focus: FixedPoint::ZERO,
            winner_idx: 0,
        }
    }

    pub fn step(&mut self, relay_activities: &[FixedPoint], cortical_feedback: FixedPoint) {
        let mut max_act = FixedPoint::ZERO;
        let mut winner = 0u8;
        for (i, &act) in relay_activities.iter().enumerate() {
            self.neurons[i] = act;
            if act > max_act {
                max_act = act;
                winner = i as u8;
            }
        }
        self.winner_idx = winner;
        self.attention_focus = max_act;
        let mut inhibition = FixedPoint::ZERO;
        for (i, &act) in relay_activities.iter().enumerate() {
            if i != winner as usize {
                inhibition += act;
            }
        }
        self.global_inhibition = (inhibition / FixedPoint::from_int((RELAY_NUCLEI - 1) as i32))
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        let feedback = cortical_feedback * FixedPoint::from_bits(6554);
        self.global_inhibition =
            (self.global_inhibition + feedback).clamp(FixedPoint::ZERO, FixedPoint::ONE);
    }

    pub fn gate_for_relay(&self, relay_idx: u8) -> FixedPoint {
        if relay_idx == self.winner_idx {
            FixedPoint::ONE - self.global_inhibition * FixedPoint::from_bits(6554)
        } else {
            FixedPoint::ONE - self.global_inhibition
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Somatosensory,
    Motor,
}

#[repr(C)]
pub struct Thalamus {
    pub relays: [ThalamicRelay; RELAY_NUCLEI],
    pub trn: ThalamicReticularNucleus,
    pub selected_modality: SensoryModality,
    pub global_gate: FixedPoint,
}

impl Thalamus {
    pub const fn new() -> Self {
        Self {
            relays: [
                ThalamicRelay::new(0),
                ThalamicRelay::new(1),
                ThalamicRelay::new(2),
                ThalamicRelay::new(3),
            ],
            trn: ThalamicReticularNucleus::new(),
            selected_modality: SensoryModality::Visual,
            global_gate: FixedPoint::ONE,
        }
    }

    pub fn step(
        &mut self,
        sensory_inputs: &[FixedPoint; RELAY_NUCLEI],
        attention: FixedPoint,
        cortical_feedback: FixedPoint,
        tick: u64,
    ) {
        let mut relay_acts = [FixedPoint::ZERO; RELAY_NUCLEI];
        for (i, relay) in self.relays.iter_mut().enumerate() {
            let attn = if i == self.selected_modality as u8 as usize {
                attention
            } else {
                attention * FixedPoint::from_bits(6554)
            };
            relay.step(sensory_inputs[i], attn, tick);
            relay_acts[i] = relay.output_rate;
        }
        self.trn.step(&relay_acts, cortical_feedback);
        for (i, relay) in self.relays.iter_mut().enumerate() {
            let trn_gate = self.trn.gate_for_relay(i as u8);
            relay.sensory_gate = trn_gate;
        }
        let mut max_act = FixedPoint::ZERO;
        for (i, &act) in relay_acts.iter().enumerate() {
            if act > max_act {
                max_act = act;
                self.selected_modality = match i {
                    0 => SensoryModality::Visual,
                    1 => SensoryModality::Auditory,
                    2 => SensoryModality::Somatosensory,
                    _ => SensoryModality::Motor,
                };
            }
        }
        self.global_gate = self.trn.attention_focus;
    }

    pub fn select_modality(&mut self, modality: SensoryModality) {
        let modality_idx = modality as u8 as usize;
        self.selected_modality = modality;
        for (i, relay) in self.relays.iter_mut().enumerate() {
            relay.set_gate(i == modality_idx);
        }
    }

    pub fn gating_vector(&self) -> [FixedPoint; RELAY_NUCLEI] {
        let mut v = [FixedPoint::ZERO; RELAY_NUCLEI];
        for (i, relay) in self.relays.iter().enumerate() {
            v[i] = relay.sensory_gate;
        }
        v
    }

    pub fn active_relay_count(&self) -> usize {
        self.relays.iter().filter(|r| r.active).count()
    }

    pub fn dominant_firing_mode(&self) -> FiringMode {
        let mut modes = [0u32; 3];
        for relay in &self.relays {
            match relay.mode {
                FiringMode::Burst => modes[0] += 1,
                FiringMode::Tonic => modes[1] += 1,
                FiringMode::Silent => modes[2] += 1,
            }
        }
        if modes[0] > modes[1] && modes[0] > modes[2] {
            FiringMode::Burst
        } else if modes[1] > modes[2] {
            FiringMode::Tonic
        } else {
            FiringMode::Silent
        }
    }
}

pub static mut THALAMUS: MaybeUninit<Thalamus> = MaybeUninit::uninit();

pub fn init_thalamus() {
    unsafe {
        THALAMUS = MaybeUninit::new(Thalamus::new());
    }
}

pub fn thalamus() -> &'static mut Thalamus {
    unsafe { THALAMUS.as_mut_ptr().as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_new() {
        let r = ThalamicRelay::new(0);
        assert_eq!(r.mode, FiringMode::Silent);
        assert_eq!(r.sensory_gate, FixedPoint::ONE);
    }

    #[test]
    fn test_relay_step_with_input() {
        let mut r = ThalamicRelay::new(0);
        r.step(FixedPoint::from_f32(0.8), FixedPoint::from_f32(0.5), 0);
        assert!(r.membrane_potential > FixedPoint::ZERO || r.mode != FiringMode::Silent);
    }

    #[test]
    fn test_relay_set_gate() {
        let mut r = ThalamicRelay::new(0);
        r.set_gate(false);
        assert_eq!(r.sensory_gate, FixedPoint::ZERO);
        r.set_gate(true);
        assert_eq!(r.sensory_gate, FixedPoint::ONE);
    }

    #[test]
    fn test_trn_new() {
        let trn = ThalamicReticularNucleus::new();
        assert_eq!(trn.global_inhibition, FixedPoint::ZERO);
    }

    #[test]
    fn test_trn_step_selects_winner() {
        let mut trn = ThalamicReticularNucleus::new();
        let acts = [
            FixedPoint::from_f32(0.8),
            FixedPoint::from_f32(0.2),
            FixedPoint::ZERO,
            FixedPoint::ZERO,
        ];
        trn.step(&acts, FixedPoint::ZERO);
        assert_eq!(trn.winner_idx, 0);
        assert!(trn.attention_focus > FixedPoint::ZERO);
    }

    #[test]
    fn test_trn_gate_for_winner() {
        let mut trn = ThalamicReticularNucleus::new();
        trn.winner_idx = 0;
        trn.global_inhibition = FixedPoint::from_f32(0.5);
        let gate = trn.gate_for_relay(0);
        let gate_other = trn.gate_for_relay(1);
        assert!(gate > gate_other);
    }

    #[test]
    fn test_thalamus_new() {
        let t = Thalamus::new();
        assert_eq!(t.relays.len(), RELAY_NUCLEI);
    }

    #[test]
    fn test_thalamus_step() {
        let mut t = Thalamus::new();
        let inputs = [FixedPoint::from_f32(0.6); RELAY_NUCLEI];
        t.step(&inputs, FixedPoint::from_f32(0.5), FixedPoint::ZERO, 0);
        assert!(t.global_gate >= FixedPoint::ZERO);
    }

    #[test]
    fn test_select_modality() {
        let mut t = Thalamus::new();
        t.select_modality(SensoryModality::Auditory);
        assert_eq!(t.relays[0].sensory_gate, FixedPoint::ZERO);
        assert_eq!(t.relays[1].sensory_gate, FixedPoint::ONE);
    }

    #[test]
    fn test_gating_vector() {
        let mut t = Thalamus::new();
        t.relays[0].sensory_gate = FixedPoint::from_f32(0.2);
        t.relays[1].sensory_gate = FixedPoint::from_f32(0.8);
        let v = t.gating_vector();
        assert_eq!(v[0], FixedPoint::from_f32(0.2));
        assert_eq!(v[1], FixedPoint::from_f32(0.8));
    }

    #[test]
    fn test_active_relay_count() {
        let mut t = Thalamus::new();
        assert_eq!(t.active_relay_count(), 0);
        t.relays[0].active = true;
        t.relays[1].active = true;
        assert_eq!(t.active_relay_count(), 2);
    }

    #[test]
    fn test_dominant_firing_mode() {
        let mut t = Thalamus::new();
        t.relays[0].mode = FiringMode::Burst;
        t.relays[1].mode = FiringMode::Burst;
        t.relays[2].mode = FiringMode::Tonic;
        t.relays[3].mode = FiringMode::Silent;
        assert_eq!(t.dominant_firing_mode(), FiringMode::Burst);
    }

    #[test]
    fn test_init() {
        init_thalamus();
        let t = thalamus();
        t.relays[0].active = true;
        assert!(t.active_relay_count() > 0);
    }
}
