use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const CA1_NEURONS: usize = 128;
const CA3_NEURONS: usize = 128;
const DG_NEURONS: usize = 256;
const SWR_DURATION: u64 = 50;
const THETA_CYCLES: u64 = 125;
const PATTERN_COMPLETION_THRESHOLD: FixedPoint = FixedPoint::from_bits(13107); // ~0.2
const LTP_RATE: FixedPoint = FixedPoint::from_bits(6554); // ~0.1
const _LTD_RATE: FixedPoint = FixedPoint::from_bits(3277); // ~0.05

#[derive(Clone, Copy)]
#[repr(C)]
pub struct GranuleCell {
    pub activity: FixedPoint,
    pub pattern_separated: bool,
    pub winning: bool,
}

impl GranuleCell {
    pub const fn new() -> Self {
        Self {
            activity: FixedPoint::ZERO,
            pattern_separated: false,
            winning: false,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PyramidalCell {
    pub activity: FixedPoint,
    pub place_field_id: u16,
    pub burst_mode: bool,
    pub trace: FixedPoint,
    pub active: bool,
}

impl PyramidalCell {
    pub const fn new() -> Self {
        Self {
            activity: FixedPoint::ZERO,
            place_field_id: 0,
            burst_mode: false,
            trace: FixedPoint::ZERO,
            active: false,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SchafferCollateral {
    pub weight: FixedPoint,
    pub active: bool,
}

impl SchafferCollateral {
    pub const fn new() -> Self {
        Self {
            weight: FixedPoint::from_f32(0.1),
            active: false,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MossyFiber {
    pub weight: FixedPoint,
}

impl MossyFiber {
    pub const fn new() -> Self {
        Self {
            weight: FixedPoint::from_f32(0.05),
        }
    }
}

#[repr(C)]
pub struct Hippocampus {
    pub dg: [GranuleCell; DG_NEURONS],
    pub ca3: [PyramidalCell; CA3_NEURONS],
    pub ca1: [PyramidalCell; CA1_NEURONS],
    pub schaffer: [SchafferCollateral; CA3_NEURONS * CA1_NEURONS / 64],
    pub mossy: [MossyFiber; DG_NEURONS * CA3_NEURONS / 256],
    pub theta_phase: FixedPoint,
    pub swr_active: bool,
    pub swr_timer: u64,
    pub consolidation_trigger: bool,
    pub novelty_detected: bool,
    pub tick: u64,
}

impl Hippocampus {
    pub const fn new() -> Self {
        Self {
            dg: [GranuleCell::new(); DG_NEURONS],
            ca3: [PyramidalCell::new(); CA3_NEURONS],
            ca1: [PyramidalCell::new(); CA1_NEURONS],
            schaffer: [SchafferCollateral::new(); CA3_NEURONS * CA1_NEURONS / 64],
            mossy: [MossyFiber::new(); DG_NEURONS * CA3_NEURONS / 256],
            theta_phase: FixedPoint::ZERO,
            swr_active: false,
            swr_timer: 0,
            consolidation_trigger: false,
            novelty_detected: false,
            tick: 0,
        }
    }

    pub fn step(&mut self, sensory_input: &[FixedPoint], novelty: FixedPoint, reward: FixedPoint) {
        self.tick += 1;
        let theta_frac = FixedPoint::from_int((self.tick % THETA_CYCLES) as i32)
            / FixedPoint::from_int(THETA_CYCLES as i32);
        self.theta_phase = (theta_frac * FixedPoint::TAU).sin() * FixedPoint::HALF + FixedPoint::HALF;
        self.dg_processing(sensory_input);
        self.ca3_recurrent(novelty);
        self.ca1_output(reward);
        self.swr_check();
        self.novelty_detected = novelty > PATTERN_COMPLETION_THRESHOLD;
    }

    fn dg_processing(&mut self, input: &[FixedPoint]) {
        for cell in self.dg.iter_mut().enumerate() {
            let (i, cell) = (cell.0, cell.1);
            let inp = if i < input.len() { input[i] } else { FixedPoint::ZERO };
            let mut separated = inp;
            let hash_phase = ((i as u64 * 2654435761u64) % 1000) as f32 / 1000.0;
            separated *= FixedPoint::from_f32(1.0 - hash_phase * 0.3);
            cell.activity = separated.clamp(FixedPoint::ZERO, FixedPoint::ONE);
            cell.pattern_separated = hash_phase > 0.5;
            cell.winning = false;
        }
        const K_WTA: usize = 16;
        let mut activations: [(FixedPoint, usize); DG_NEURONS] = [(FixedPoint::ZERO, 0); DG_NEURONS];
        for (i, cell) in self.dg.iter().enumerate() {
            activations[i] = (cell.activity, i);
        }
        activations.sort_by(|a, b| b.0.cmp(&a.0));
        for &(_, idx) in activations.iter().take(K_WTA) {
            self.dg[idx].winning = true;
        }
        for cell in self.dg.iter_mut() {
            if !cell.winning {
                cell.activity = FixedPoint::ZERO;
            }
        }
    }

    fn ca3_recurrent(&mut self, novelty: FixedPoint) {
        for i in 0..CA3_NEURONS {
            let mut input_sum = FixedPoint::ZERO;
            let dg_idx = (i * (DG_NEURONS / CA3_NEURONS)) % DG_NEURONS;
            if self.dg[dg_idx].winning {
                let mf_idx = (i * (DG_NEURONS * CA3_NEURONS / 256) / CA3_NEURONS) % self.mossy.len();
                input_sum += self.mossy[mf_idx].weight * self.dg[dg_idx].activity;
            }
            for (j, cell) in self.ca3.iter().enumerate() {
                if j != i && cell.active {
                    let sc_idx = (i * CA3_NEURONS + j) % self.schaffer.len();
                    input_sum += self.schaffer[sc_idx].weight * cell.activity;
                }
            }
            let threshold = if novelty > PATTERN_COMPLETION_THRESHOLD {
                FixedPoint::from_bits(9830)
            } else {
                FixedPoint::from_bits(13107)
            };
            self.ca3[i].activity = if input_sum > threshold {
                (input_sum - threshold).clamp(FixedPoint::ZERO, FixedPoint::ONE)
            } else {
                FixedPoint::ZERO
            };
            self.ca3[i].active = self.ca3[i].activity > FixedPoint::from_bits(6554);
            if self.ca3[i].active {
                self.ca3[i].trace += FixedPoint::from_bits(6554);
            }
            self.ca3[i].trace *= FixedPoint::from_bits(16384);
        }
        if novelty > PATTERN_COMPLETION_THRESHOLD {
            for i in 0..self.ca3.len() {
                for j in 0..self.ca3.len() {
                    if i != j && self.ca3[i].active && self.ca3[j].active {
                        let sc_idx = (i * CA3_NEURONS + j) % self.schaffer.len();
                        self.schaffer[sc_idx].weight = (self.schaffer[sc_idx].weight + LTP_RATE * novelty)
                            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                    }
                }
            }
        }
    }

    fn ca1_output(&mut self, reward: FixedPoint) {
        for (i, cell) in self.ca1.iter_mut().enumerate() {
            let ca3_idx = i % CA3_NEURONS;
            let ca3_act = self.ca3[ca3_idx].activity;
            let sc_idx = (i * CA3_NEURONS + ca3_idx) % self.schaffer.len();
            let mut inp = self.schaffer[sc_idx].weight * ca3_act;
            if self.ca3[ca3_idx].active {
                inp += FixedPoint::from_bits(6554);
            }
            cell.activity = inp.clamp(FixedPoint::ZERO, FixedPoint::ONE);
            cell.active = cell.activity > FixedPoint::from_bits(6554);
            if cell.active && reward > FixedPoint::ZERO {
                let delta = LTP_RATE * reward * cell.activity;
                self.schaffer[sc_idx].weight = (self.schaffer[sc_idx].weight + delta)
                    .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
        }
    }

    fn swr_check(&mut self) {
        let ca3_active = self.ca3.iter().filter(|c| c.active).count();
        if ca3_active > CA3_NEURONS / 4 && self.theta_phase > FixedPoint::from_bits(16384) && !self.swr_active {
            self.swr_active = true;
            self.swr_timer = 0;
            self.consolidation_trigger = true;
        }
        if self.swr_active {
            self.swr_timer += 1;
            let swr_frac = FixedPoint::from_int((self.swr_timer % SWR_DURATION) as i32)
                / FixedPoint::from_int(SWR_DURATION as i32);
            let sharp_wave = (swr_frac * FixedPoint::TAU).sin() * FixedPoint::HALF + FixedPoint::HALF;
            for cell in self.ca3.iter_mut() {
                if cell.active && cell.trace > FixedPoint::from_bits(6554) {
                    cell.activity = (cell.activity + sharp_wave * FixedPoint::from_bits(3277))
                        .clamp(FixedPoint::ZERO, FixedPoint::ONE);
                }
            }
            if self.swr_timer >= SWR_DURATION {
                self.swr_active = false;
                self.swr_timer = 0;
            }
        } else {
            self.consolidation_trigger = false;
        }
    }

    pub fn recall_pattern(&self, cue: &[FixedPoint]) -> [FixedPoint; CA1_NEURONS] {
        let mut output = [FixedPoint::ZERO; CA1_NEURONS];
        let dg_acts: [FixedPoint; DG_NEURONS] = {
            let mut d = [FixedPoint::ZERO; DG_NEURONS];
            for (i, cell) in self.dg.iter().enumerate() {
                let inp = if i < cue.len() { cue[i] } else { FixedPoint::ZERO };
                d[i] = if cell.pattern_separated {
                    FixedPoint::ZERO
                } else {
                    inp * cell.activity
                };
            }
            d
        };
        for i in 0..CA3_NEURONS {
            let mut ca3_act = FixedPoint::ZERO;
            for (j, &da) in dg_acts.iter().enumerate() {
                if j % (DG_NEURONS / CA3_NEURONS) == i {
                    ca3_act += da;
                }
            }
            if ca3_act > PATTERN_COMPLETION_THRESHOLD {
                for (k, out) in output.iter_mut().enumerate().take(CA1_NEURONS) {
                    if k % CA3_NEURONS == i {
                        *out = ca3_act;
                    }
                }
            }
        }
        output
    }

    pub fn clear_swr(&mut self) {
        self.swr_active = false;
        self.swr_timer = 0;
        self.consolidation_trigger = false;
    }

    pub fn ca3_active_count(&self) -> usize {
        self.ca3.iter().filter(|c| c.active).count()
    }

    pub fn ca1_active_count(&self) -> usize {
        self.ca1.iter().filter(|c| c.active).count()
    }
}

pub static mut HIPPOCAMPUS: MaybeUninit<Hippocampus> = MaybeUninit::uninit();

pub fn init_hippocampus() {
    unsafe {
        HIPPOCAMPUS = MaybeUninit::new(Hippocampus::new());
    }
}

pub fn hippocampus() -> &'static mut Hippocampus {
    unsafe { HIPPOCAMPUS.as_mut_ptr().as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hippocampus_new() {
        let h = Hippocampus::new();
        assert_eq!(h.dg.len(), DG_NEURONS);
        assert_eq!(h.ca3.len(), CA3_NEURONS);
        assert_eq!(h.ca1.len(), CA1_NEURONS);
    }

    #[test]
    fn test_dg_processing_wta() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::from_f32(0.8); DG_NEURONS];
        h.dg_processing(&input);
        let winners = h.dg.iter().filter(|c| c.winning).count();
        assert!(winners > 0);
        assert!(winners <= DG_NEURONS);
    }

    #[test]
    fn test_ca3_recurrent_activity() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::from_f32(0.9); DG_NEURONS];
        h.ca3[0].trace = FixedPoint::ONE;
        h.dg_processing(&input);
        for cell in h.dg.iter_mut() {
            if cell.winning {
                cell.activity = FixedPoint::from_f32(0.95);
            }
        }
        h.mossy[0].weight = FixedPoint::from_f32(0.5);
        h.ca3_recurrent(FixedPoint::from_f32(0.1));
        assert!(h.ca3.iter().any(|c| c.activity > FixedPoint::ZERO));
    }

    #[test]
    fn test_ca1_output() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::from_f32(0.5); DG_NEURONS];
        h.dg_processing(&input);
        h.ca3_recurrent(FixedPoint::ZERO);
        h.ca1_output(FixedPoint::ZERO);
        assert!(h.ca1.iter().any(|c| c.activity >= FixedPoint::ZERO));
    }

    #[test]
    fn test_theta_phase() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::ZERO; DG_NEURONS];
        h.step(&input, FixedPoint::ZERO, FixedPoint::ZERO);
        assert!(h.theta_phase >= FixedPoint::ZERO);
        assert!(h.theta_phase <= FixedPoint::ONE);
    }

    #[test]
    fn test_swr_triggered_by_active_ca3() {
        let mut h = Hippocampus::new();
        for cell in h.ca3.iter_mut() {
            cell.active = true;
        }
        h.theta_phase = FixedPoint::from_f32(0.8);
        h.swr_check();
        assert!(h.swr_active);
    }

    #[test]
    fn test_swr_duration() {
        let mut h = Hippocampus::new();
        for cell in h.ca3.iter_mut() {
            cell.active = true;
        }
        h.theta_phase = FixedPoint::from_f32(0.8);
        h.swr_check();
        h.swr_active = true;
        h.swr_timer = SWR_DURATION - 1;
        h.swr_check();
        assert!(!h.swr_active);
    }

    #[test]
    fn test_recall_pattern() {
        let h = Hippocampus::new();
        let cue = [FixedPoint::from_f32(0.3); DG_NEURONS];
        let recalled = h.recall_pattern(&cue);
        assert_eq!(recalled.len(), CA1_NEURONS);
    }

    #[test]
    fn test_novelty_detection() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::ZERO; DG_NEURONS];
        h.step(&input, FixedPoint::from_f32(0.5), FixedPoint::ZERO);
        assert!(h.novelty_detected);
    }

    #[test]
    fn test_full_step() {
        let mut h = Hippocampus::new();
        let input = [FixedPoint::from_f32(0.6); DG_NEURONS];
        h.step(&input, FixedPoint::ZERO, FixedPoint::from_f32(0.3));
        assert!(h.tick > 0);
        assert!(h.theta_phase >= FixedPoint::ZERO);
    }

    #[test]
    fn test_clear_swr() {
        let mut h = Hippocampus::new();
        h.swr_active = true;
        h.consolidation_trigger = true;
        h.clear_swr();
        assert!(!h.swr_active);
        assert!(!h.consolidation_trigger);
    }

    #[test]
    fn test_ca3_ca1_active_counts() {
        let mut h = Hippocampus::new();
        h.ca3[0].active = true;
        h.ca1[0].active = true;
        h.ca1[1].active = true;
        assert_eq!(h.ca3_active_count(), 1);
        assert_eq!(h.ca1_active_count(), 2);
    }

    #[test]
    fn test_init() {
        init_hippocampus();
        let h = hippocampus();
        assert_eq!(h.dg.len(), DG_NEURONS);
    }
}
