use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const GRANULE_CELLS: usize = 1024;
const PURKINJE_CELLS: usize = 64;
const PARALLEL_FIBERS_PER_PC: usize = 16;
const CLIMBING_FIBERS: usize = 64;
const MOSSY_FIBERS: usize = 128;
const PF_LTP_RATE: FixedPoint = FixedPoint::from_bits(6554); // ~0.1
const PF_LTD_RATE: FixedPoint = FixedPoint::from_bits(13107); // ~0.2
const TIMING_PRECISION_US: u64 = 1000;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct GranuleCell {
    pub activity: FixedPoint,
    pub parallel_fiber_active: bool,
    pub encoding: FixedPoint,
    pub active: bool,
}

impl GranuleCell {
    pub const fn new() -> Self {
        Self {
            activity: FixedPoint::ZERO,
            parallel_fiber_active: false,
            encoding: FixedPoint::ZERO,
            active: false,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PurkinjeCell {
    pub id: u8,
    pub membrane_potential: FixedPoint,
    pub firing_rate: FixedPoint,
    pub simple_spikes: u32,
    pub complex_spikes: u32,
    pub climbing_fiber_error: FixedPoint,
    pub parallel_fiber_weights: [FixedPoint; PARALLEL_FIBERS_PER_PC],
    pub pff_activity: [FixedPoint; PARALLEL_FIBERS_PER_PC],
    pub active: bool,
    pub inhibition_output: FixedPoint,
}

impl PurkinjeCell {
    pub const fn new(id: u8) -> Self {
        Self {
            id,
            membrane_potential: FixedPoint::ZERO,
            firing_rate: FixedPoint::from_f32(0.5),
            simple_spikes: 0,
            complex_spikes: 0,
            climbing_fiber_error: FixedPoint::ZERO,
            parallel_fiber_weights: [FixedPoint::from_f32(0.3); PARALLEL_FIBERS_PER_PC],
            pff_activity: [FixedPoint::ZERO; PARALLEL_FIBERS_PER_PC],
            active: false,
            inhibition_output: FixedPoint::ZERO,
        }
    }

    pub fn step(&mut self, pf_inputs: &[FixedPoint], cf_input: FixedPoint, motor_error: FixedPoint) {
        self.climbing_fiber_error = cf_input;
        let mut total_pf = FixedPoint::ZERO;
        for (i, (&pf_weight, &pf_act)) in self.parallel_fiber_weights.iter().zip(pf_inputs.iter()).enumerate() {
            let contribution = pf_weight * pf_act;
            total_pf += contribution;
            self.pff_activity[i] = pf_act;
        }
        let avg_pf = total_pf / FixedPoint::from_int(PARALLEL_FIBERS_PER_PC as i32);
        let cf_inhibition = cf_input * FixedPoint::from_bits(13107);
        self.membrane_potential = (avg_pf - cf_inhibition)
            .clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.firing_rate = FixedPoint::ONE - self.membrane_potential;
        self.simple_spikes += if self.firing_rate > FixedPoint::from_bits(13107) { 1 } else { 0 };
        if cf_input > FixedPoint::from_bits(19661) {
            self.complex_spikes += 1;
            let ltd = PF_LTD_RATE * cf_input * motor_error;
            for w in self.parallel_fiber_weights.iter_mut() {
                *w = (*w - ltd).clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
        } else if motor_error > FixedPoint::ZERO && motor_error < FixedPoint::from_bits(6554) {
            let ltp = PF_LTP_RATE * (FixedPoint::ONE - motor_error);
            for w in self.parallel_fiber_weights.iter_mut() {
                *w = (*w + ltp).clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
        }
        self.active = self.firing_rate > FixedPoint::from_bits(6554);
        self.inhibition_output = self.firing_rate;
    }

    pub fn learn_timing(&mut self, expected_time: u64, actual_time: u64, cf_input: FixedPoint) {
        let timing_error = if expected_time > actual_time {
            FixedPoint::from_f32((expected_time - actual_time) as f32 / TIMING_PRECISION_US as f32)
        } else {
            FixedPoint::from_f32((actual_time - expected_time) as f32 / TIMING_PRECISION_US as f32)
        };
        let cf_adjusted = cf_input + timing_error * FixedPoint::from_bits(6554);
        let ltd = PF_LTD_RATE * cf_adjusted;
        for w in self.parallel_fiber_weights.iter_mut() {
            *w = (*w - ltd * FixedPoint::from_bits(6554)).clamp(FixedPoint::ZERO, FixedPoint::ONE);
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ClimbingFiber {
    pub error_signal: FixedPoint,
    pub active: bool,
}

impl ClimbingFiber {
    pub const fn new() -> Self {
        Self {
            error_signal: FixedPoint::ZERO,
            active: false,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MossyFiber {
    pub encoding: FixedPoint,
    pub active: bool,
}

impl MossyFiber {
    pub const fn new() -> Self {
        Self {
            encoding: FixedPoint::ZERO,
            active: false,
        }
    }
}

#[repr(C)]
pub struct Cerebellum {
    pub granule_cells: [GranuleCell; GRANULE_CELLS],
    pub purkinje_cells: [PurkinjeCell; PURKINJE_CELLS],
    pub climbing_fibers: [ClimbingFiber; CLIMBING_FIBERS],
    pub mossy_fibers: [MossyFiber; MOSSY_FIBERS],
    pub motor_output: FixedPoint,
    pub timing_precision: u64,
    pub tick: u64,
}

impl Cerebellum {
    pub const fn new() -> Self {
        let mut pcs = [PurkinjeCell::new(0); PURKINJE_CELLS];
        let mut i = 0;
        while i < PURKINJE_CELLS {
            pcs[i] = PurkinjeCell::new(i as u8);
            i += 1;
        }
        Self {
            granule_cells: [GranuleCell::new(); GRANULE_CELLS],
            purkinje_cells: pcs,
            climbing_fibers: [ClimbingFiber::new(); CLIMBING_FIBERS],
            mossy_fibers: [MossyFiber::new(); MOSSY_FIBERS],
            motor_output: FixedPoint::ZERO,
            timing_precision: TIMING_PRECISION_US,
            tick: 0,
        }
    }

    pub fn step(&mut self, sensory_input: &[FixedPoint], motor_command: FixedPoint, error: FixedPoint, expected_time: u64) {
        self.tick += 1;
        self.mossy_fiber_encoding(sensory_input, motor_command);
        self.granule_layer();
        self.parallel_fiber_to_purkinje(error, expected_time);
        self.motor_output = self.purkinje_output();
    }

    fn mossy_fiber_encoding(&mut self, sensory_input: &[FixedPoint], motor_command: FixedPoint) {
        for (i, mf) in self.mossy_fibers.iter_mut().enumerate() {
            let sensor_val = if i < sensory_input.len() {
                sensory_input[i]
            } else {
                FixedPoint::ZERO
            };
            let motor_comp = if i == 0 {
                motor_command
            } else {
                motor_command * FixedPoint::from_f32(1.0 / (i as f32 + 1.0))
            };
            mf.encoding = (sensor_val * FixedPoint::from_bits(16384) + motor_comp * FixedPoint::from_bits(6554))
                .clamp(FixedPoint::ZERO, FixedPoint::ONE);
            mf.active = mf.encoding > FixedPoint::from_bits(6554);
        }
    }

    fn granule_layer(&mut self) {
        const GC_PER_MF: usize = GRANULE_CELLS / MOSSY_FIBERS;
        for (i, gc) in self.granule_cells.iter_mut().enumerate() {
            let mf_idx = i / GC_PER_MF;
            if mf_idx < MOSSY_FIBERS {
                let mf = &self.mossy_fibers[mf_idx];
                let expansion = FixedPoint::from_f32(
                    (((i as u64 * 2654435761u64) % 1000) as f32 / 1000.0) * 0.5 + 0.5,
                );
                gc.activity = (mf.encoding * expansion).clamp(FixedPoint::ZERO, FixedPoint::ONE);
            }
            gc.parallel_fiber_active = gc.activity > FixedPoint::from_bits(9830);
            gc.active = gc.parallel_fiber_active;
        }
    }

    fn parallel_fiber_to_purkinje(&mut self, error: FixedPoint, expected_time: u64) {
        let pc_per_gc = GRANULE_CELLS / PURKINJE_CELLS;
        let cf_per_pc = CLIMBING_FIBERS / PURKINJE_CELLS;
        for (i, pc) in self.purkinje_cells.iter_mut().enumerate() {
            let start_gc = i * pc_per_gc;
            let mut pf_inputs = [FixedPoint::ZERO; PARALLEL_FIBERS_PER_PC];
            for (j, pf_inp) in pf_inputs.iter_mut().enumerate() {
                let gc_idx = start_gc + j;
                if gc_idx < GRANULE_CELLS {
                    *pf_inp = self.granule_cells[gc_idx].activity;
                }
            }
            let cf_idx = i * cf_per_pc;
            let cf_input = if cf_idx < CLIMBING_FIBERS {
                self.climbing_fibers[cf_idx].error_signal
            } else {
                FixedPoint::ZERO
            };
            pc.step(&pf_inputs, cf_input, error);
            if error > FixedPoint::from_bits(13107) && pc.climbing_fiber_error > FixedPoint::ZERO {
                pc.learn_timing(expected_time, self.tick, cf_input);
            }
        }
    }

    fn purkinje_output(&self) -> FixedPoint {
        let mut output = FixedPoint::ZERO;
        for pc in &self.purkinje_cells {
            output += pc.inhibition_output;
        }
        output / FixedPoint::from_int(PURKINJE_CELLS as i32)
    }

    pub fn set_error_signal(&mut self, error: FixedPoint, fiber_idx: usize) {
        if fiber_idx < CLIMBING_FIBERS {
            self.climbing_fibers[fiber_idx].error_signal = error;
            self.climbing_fibers[fiber_idx].active = error > FixedPoint::from_bits(6554);
        }
    }

    pub fn motor_precision(&self) -> FixedPoint {
        let mut total_inhibition = FixedPoint::ZERO;
        for pc in &self.purkinje_cells {
            total_inhibition += pc.inhibition_output;
        }
        total_inhibition / FixedPoint::from_int(PURKINJE_CELLS as i32)
    }

    pub fn active_purkinje_count(&self) -> usize {
        self.purkinje_cells.iter().filter(|pc| pc.active).count()
    }

    pub fn active_granule_count(&self) -> usize {
        self.granule_cells.iter().filter(|gc| gc.active).count()
    }
}

pub static mut CEREBELLUM: MaybeUninit<Cerebellum> = MaybeUninit::uninit();

pub fn init_cerebellum() {
    unsafe {
        CEREBELLUM = MaybeUninit::new(Cerebellum::new());
    }
}

pub fn cerebellum() -> &'static mut Cerebellum {
    unsafe { CEREBELLUM.as_mut_ptr().as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purkinje_new() {
        let pc = PurkinjeCell::new(0);
        assert_eq!(pc.id, 0);
        assert_eq!(pc.firing_rate, FixedPoint::from_f32(0.5));
        assert_eq!(pc.simple_spikes, 0);
    }

    #[test]
    fn test_purkinje_step_baseline() {
        let mut pc = PurkinjeCell::new(0);
        let pf_inputs = [FixedPoint::from_f32(0.3); PARALLEL_FIBERS_PER_PC];
        pc.step(&pf_inputs, FixedPoint::ZERO, FixedPoint::ZERO);
        assert!(pc.firing_rate <= FixedPoint::ONE);
    }

    #[test]
    fn test_purkinje_step_cf_inhibition() {
        let mut pc = PurkinjeCell::new(0);
        let pf_inputs = [FixedPoint::from_f32(0.5); PARALLEL_FIBERS_PER_PC];
        pc.step(&pf_inputs, FixedPoint::from_f32(0.8), FixedPoint::from_f32(0.5));
        assert!(pc.complex_spikes > 0);
    }

    #[test]
    fn test_purkinje_learn_timing() {
        let mut pc = PurkinjeCell::new(0);
        let old_w = pc.parallel_fiber_weights[0];
        pc.learn_timing(1000, 1200, FixedPoint::from_f32(0.5));
        assert!(pc.parallel_fiber_weights[0] < old_w);
    }

    #[test]
    fn test_purkinje_ltp_low_error() {
        let mut pc = PurkinjeCell::new(0);
        pc.parallel_fiber_weights[0] = FixedPoint::from_f32(0.2);
        let pf_inputs = [FixedPoint::from_f32(0.3); PARALLEL_FIBERS_PER_PC];
        pc.step(&pf_inputs, FixedPoint::ZERO, FixedPoint::from_f32(0.05));
        assert!(pc.parallel_fiber_weights[0] > FixedPoint::from_f32(0.2));
    }

    #[test]
    fn test_cerebellum_new() {
        let c = Cerebellum::new();
        assert_eq!(c.granule_cells.len(), GRANULE_CELLS);
        assert_eq!(c.purkinje_cells.len(), PURKINJE_CELLS);
    }

    #[test]
    fn test_mossy_fiber_encoding() {
        let mut c = Cerebellum::new();
        let input = [FixedPoint::from_f32(0.7); 10];
        c.mossy_fiber_encoding(&input, FixedPoint::from_f32(0.3));
        assert!(c.mossy_fibers[0].encoding > FixedPoint::ZERO);
    }

    #[test]
    fn test_granule_layer() {
        let mut c = Cerebellum::new();
        c.mossy_fibers[0].encoding = FixedPoint::from_f32(0.8);
        c.granule_layer();
        let active = c.granule_cells.iter().filter(|g| g.active).count();
        assert!(active > 0);
    }

    #[test]
    fn test_full_step() {
        let mut c = Cerebellum::new();
        let input = [FixedPoint::from_f32(0.5); 10];
        c.step(&input, FixedPoint::from_f32(0.3), FixedPoint::from_f32(0.1), 0);
        assert!(c.tick > 0);
        assert!(c.motor_output >= FixedPoint::ZERO);
    }

    #[test]
    fn test_set_error_signal() {
        let mut c = Cerebellum::new();
        c.set_error_signal(FixedPoint::from_f32(0.9), 0);
        assert!(c.climbing_fibers[0].active);
        assert_eq!(c.climbing_fibers[0].error_signal, FixedPoint::from_f32(0.9));
    }

    #[test]
    fn test_motor_precision() {
        let mut c = Cerebellum::new();
        for pc in c.purkinje_cells.iter_mut() {
            pc.inhibition_output = FixedPoint::from_f32(0.5);
        }
        let prec = c.motor_precision();
        assert!(prec > FixedPoint::ZERO);
    }

    #[test]
    fn test_active_counts() {
        let mut c = Cerebellum::new();
        c.purkinje_cells[0].active = true;
        c.granule_cells[0].active = true;
        c.granule_cells[1].active = true;
        assert_eq!(c.active_purkinje_count(), 1);
        assert_eq!(c.active_granule_count(), 2);
    }

    #[test]
    fn test_purkinje_cf_error_tracking() {
        let mut pc = PurkinjeCell::new(0);
        let pf_inputs = [FixedPoint::from_f32(0.4); PARALLEL_FIBERS_PER_PC];
        pc.step(&pf_inputs, FixedPoint::from_f32(0.6), FixedPoint::ZERO);
        assert_eq!(pc.climbing_fiber_error, FixedPoint::from_f32(0.6));
    }

    #[test]
    fn test_init() {
        init_cerebellum();
        let c = cerebellum();
        assert_eq!(c.purkinje_cells.len(), PURKINJE_CELLS);
    }
}
