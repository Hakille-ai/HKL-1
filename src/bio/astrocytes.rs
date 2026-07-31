use crate::core::math::FixedPoint;
use core::mem::MaybeUninit;

const ASTROCYTES: usize = 64;
const GRID_COLS: usize = 8;
const CALCIUM_ACTIVATION: FixedPoint = FixedPoint::from_bits(16384); // ~0.25
const GLUTAMATE_UPTAKE_RATE: FixedPoint = FixedPoint::from_bits(6554); // ~0.1
const WAVE_SPEED: FixedPoint = FixedPoint::from_bits(6554);
const GLIOTRANSMITTER_RELEASE: FixedPoint = FixedPoint::from_bits(9830); // ~0.15
const SLOW_OSCILLATION_PERIOD: u64 = 10000;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Astrocyte {
    pub calcium: FixedPoint,
    pub ip3: FixedPoint,
    pub glutamate_uptake: FixedPoint,
    pub gliotransmitter_level: FixedPoint,
    pub wave_phase: FixedPoint,
    pub modulation_strength: FixedPoint,
    pub active: bool,
}

impl Astrocyte {
    pub const fn new() -> Self {
        Self {
            calcium: FixedPoint::ZERO,
            ip3: FixedPoint::ZERO,
            glutamate_uptake: FixedPoint::ONE,
            gliotransmitter_level: FixedPoint::ZERO,
            wave_phase: FixedPoint::ZERO,
            modulation_strength: FixedPoint::ZERO,
            active: false,
        }
    }

    pub fn step(&mut self, synaptic_activity: FixedPoint, dt: FixedPoint) {
        self.ip3 += synaptic_activity * FixedPoint::from_bits(6554);
        if self.ip3 > FixedPoint::ONE {
            self.ip3 = FixedPoint::ONE;
        }
        let ca_influx = if self.ip3 > CALCIUM_ACTIVATION {
            (self.ip3 - CALCIUM_ACTIVATION) * FixedPoint::from_bits(13107)
        } else {
            FixedPoint::ZERO
        };
        self.calcium += (ca_influx - self.calcium * FixedPoint::from_bits(6554)) * dt;
        self.calcium = self.calcium.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.glutamate_uptake = FixedPoint::ONE - self.calcium * GLUTAMATE_UPTAKE_RATE;
        if self.calcium > FixedPoint::from_bits(19661) {
            self.gliotransmitter_level += GLIOTRANSMITTER_RELEASE * dt;
        } else {
            self.gliotransmitter_level *= FixedPoint::ONE - FixedPoint::from_bits(1311) * dt;
        }
        self.gliotransmitter_level = self.gliotransmitter_level.clamp(FixedPoint::ZERO, FixedPoint::ONE);
        self.wave_phase += WAVE_SPEED * dt;
        if self.wave_phase > FixedPoint::ONE {
            self.wave_phase -= FixedPoint::ONE;
        }
        self.modulation_strength = self.calcium * FixedPoint::from_bits(19661);
        self.active = self.calcium > FixedPoint::from_bits(6554);
    }

    pub fn modulate_weight(&self, weight: FixedPoint) -> FixedPoint {
        if self.active {
            weight * (FixedPoint::ONE + self.modulation_strength * FixedPoint::from_bits(6554))
        } else {
            weight * (FixedPoint::ONE - self.gliotransmitter_level * FixedPoint::from_bits(3277))
        }
    }

    pub fn slow_oscillation_phase(tick: u64) -> FixedPoint {
        let fraction = FixedPoint::from_int((tick % SLOW_OSCILLATION_PERIOD) as i32)
            / FixedPoint::from_int(SLOW_OSCILLATION_PERIOD as i32);
        (fraction * FixedPoint::TAU).sin() * FixedPoint::HALF + FixedPoint::HALF
    }
}

#[repr(C)]
pub struct AstrocyteNetwork {
    pub cells: [Astrocyte; ASTROCYTES],
}

impl AstrocyteNetwork {
    pub const fn new() -> Self {
        Self {
            cells: [Astrocyte::new(); ASTROCYTES],
        }
    }

    pub fn step_all(&mut self, synaptic_input: &[FixedPoint], tick: u64, dt: FixedPoint) {
        let slow_phase = Astrocyte::slow_oscillation_phase(tick);
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let activity = if i < synaptic_input.len() {
                synaptic_input[i]
            } else {
                FixedPoint::ZERO
            };
            let modulated = activity + slow_phase * FixedPoint::from_bits(3277);
            cell.step(modulated, dt);
        }
    }

    pub fn propagate_waves(&mut self) {
        let mut new_calcium = [FixedPoint::ZERO; ASTROCYTES];
        for (i, cell) in self.cells.iter().enumerate() {
            if i >= ASTROCYTES { break; }
            if cell.active {
                let row = i / GRID_COLS;
                let col = i % GRID_COLS;
                for dr in [0usize, 1, 2].iter() {
                    for dc in [0usize, 1, 2].iter() {
                        let nr = row + dr;
                        let nc = col + dc;
                        if nr < GRID_COLS && nc < GRID_COLS && (dr != &0 || dc != &0) {
                            let ni = nr * GRID_COLS + nc;
                            new_calcium[ni] += FixedPoint::from_bits(3277);
                        }
                    }
                }
            }
        }
        for i in 0..ASTROCYTES {
            if new_calcium[i] > FixedPoint::ZERO {
                self.cells[i].calcium += new_calcium[i];
                if self.cells[i].calcium > FixedPoint::ONE {
                    self.cells[i].calcium = FixedPoint::ONE;
                }
            }
        }
    }

    pub fn modulate_synaptic_weights(&self, weights: &mut [FixedPoint], neuron_indices: &[usize]) {
        for (w, &ni) in weights.iter_mut().zip(neuron_indices) {
            let astro_idx = ni % ASTROCYTES;
            *w = self.cells[astro_idx].modulate_weight(*w);
        }
    }

    pub fn global_gliotransmitter_level(&self) -> FixedPoint {
        let mut sum = FixedPoint::ZERO;
        for cell in &self.cells {
            sum += cell.gliotransmitter_level;
        }
        sum / FixedPoint::from_int(ASTROCYTES as i32)
    }

    pub fn uptake_adjustment(&self) -> FixedPoint {
        let mut avg = FixedPoint::ZERO;
        for cell in &self.cells {
            avg += cell.glutamate_uptake;
        }
        avg / FixedPoint::from_int(ASTROCYTES as i32)
    }

    pub fn active_count(&self) -> usize {
        self.cells.iter().filter(|c| c.active).count()
    }
}

pub static mut ASTROCYTE_NETWORK: MaybeUninit<AstrocyteNetwork> = MaybeUninit::uninit();

pub fn init_astrocytes() {
    unsafe {
        ASTROCYTE_NETWORK = MaybeUninit::new(AstrocyteNetwork::new());
    }
}

pub fn astrocyte_network() -> &'static mut AstrocyteNetwork {
    unsafe { ASTROCYTE_NETWORK.as_mut_ptr().as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astrocyte_new() {
        let a = Astrocyte::new();
        assert_eq!(a.calcium, FixedPoint::ZERO);
        assert_eq!(a.glutamate_uptake, FixedPoint::ONE);
        assert!(!a.active);
    }

    #[test]
    fn test_astrocyte_step_activation() {
        let mut a = Astrocyte::new();
        let high_activity = FixedPoint::ONE;
        for _ in 0..10 {
            a.step(high_activity, FixedPoint::from_f32(0.1));
        }
        assert!(a.calcium > FixedPoint::ZERO);
        assert!(a.glutamate_uptake < FixedPoint::ONE);
    }

    #[test]
    fn test_astrocyte_modulate_weight_active() {
        let mut a = Astrocyte::new();
        a.calcium = FixedPoint::from_f32(0.5);
        a.modulation_strength = a.calcium * FixedPoint::from_bits(19661);
        a.active = true;
        let w = FixedPoint::from_f32(1.0);
        let modulated = a.modulate_weight(w);
        assert!(modulated > w);
    }

    #[test]
    fn test_astrocyte_modulate_weight_inactive() {
        let mut a = Astrocyte::new();
        a.gliotransmitter_level = FixedPoint::from_f32(0.3);
        a.active = false;
        let w = FixedPoint::from_f32(1.0);
        let modulated = a.modulate_weight(w);
        assert!(modulated < w);
    }

    #[test]
    fn test_slow_oscillation_phase() {
        let p0 = Astrocyte::slow_oscillation_phase(0);
        assert!(p0 >= FixedPoint::ZERO);
        let p1 = Astrocyte::slow_oscillation_phase(SLOW_OSCILLATION_PERIOD / 4);
        assert!(p1 >= p0);
    }

    #[test]
    fn test_astrocyte_network_step_all() {
        let mut net = AstrocyteNetwork::new();
        let inputs = [FixedPoint::from_f32(1.0); ASTROCYTES];
        for _ in 0..10 {
            net.step_all(&inputs, 100, FixedPoint::from_f32(0.2));
        }
        assert!(net.cells[0].calcium > FixedPoint::ZERO);
    }

    #[test]
    fn test_wave_propagation() {
        let mut net = AstrocyteNetwork::new();
        net.cells[0].calcium = FixedPoint::ONE;
        net.cells[0].active = true;
        net.propagate_waves();
        assert!(net.cells[1].calcium > FixedPoint::ZERO);
    }

    #[test]
    fn test_modulate_synaptic_weights() {
        let mut net = AstrocyteNetwork::new();
        net.cells[0].calcium = FixedPoint::from_f32(0.5);
        net.cells[0].modulation_strength = net.cells[0].calcium * FixedPoint::from_bits(19661);
        net.cells[0].active = true;
        let mut weights = [FixedPoint::from_f32(1.0); 3];
        let indices = [0, 10, 20];
        net.modulate_synaptic_weights(&mut weights, &indices);
        assert!(weights[0] > FixedPoint::from_f32(1.0));
    }

    #[test]
    fn test_global_gliotransmitter_level() {
        let mut net = AstrocyteNetwork::new();
        for cell in net.cells.iter_mut() {
            cell.gliotransmitter_level = FixedPoint::from_f32(0.5);
        }
        let level = net.global_gliotransmitter_level();
        assert!(level > FixedPoint::ZERO);
    }

    #[test]
    fn test_uptake_adjustment() {
        let mut net = AstrocyteNetwork::new();
        for cell in net.cells.iter_mut() {
            cell.glutamate_uptake = FixedPoint::from_f32(0.3);
        }
        let adj = net.uptake_adjustment();
        assert_eq!(adj, FixedPoint::from_f32(0.3));
    }

    #[test]
    fn test_init_astrocytes() {
        init_astrocytes();
        let net = astrocyte_network();
        assert_eq!(net.cells[0].calcium, FixedPoint::ZERO);
    }

    #[test]
    fn test_active_count() {
        let mut net = AstrocyteNetwork::new();
        net.cells[0].active = true;
        net.cells[1].active = true;
        assert_eq!(net.active_count(), 2);
    }
}
