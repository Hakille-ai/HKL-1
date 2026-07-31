use crate::core::math::FixedPoint;
use crate::core::memory::{NeuronFlags, NeuronId, neuron_state, neuron_state_ref};

pub struct SpinalReflexes {
    pub reflex_rules: [ReflexRule; 16],
    pub rule_count: u8,
    pub active_reflexes: u16,
    pub cognitive_override: bool,
}

#[derive(Clone, Copy)]
pub struct ReflexRule {
    pub sensor_neurons: [NeuronId; 4],
    pub motor_neuron: NeuronId,
    pub threshold: FixedPoint,
    pub intensity: FixedPoint,
    pub enabled: bool,
    pub cognitive_override: bool,
    pub description: [u8; 32],
}

impl SpinalReflexes {
    pub const fn new() -> Self {
        Self {
            reflex_rules: [ReflexRule {
                sensor_neurons: [NeuronId::INVALID; 4],
                motor_neuron: NeuronId::INVALID,
                threshold: FixedPoint::ZERO,
                intensity: FixedPoint::ZERO,
                enabled: false,
                cognitive_override: false,
                description: [0; 32],
            }; 16],
            rule_count: 0,
            active_reflexes: 0,
            cognitive_override: false,
        }
    }

    pub fn check_all(&mut self) {
        self.active_reflexes = 0;
        for i in 0..self.rule_count as usize {
            let rule = &self.reflex_rules[i];
            if !rule.enabled {
                continue;
            }

            if self.cognitive_override && rule.cognitive_override {
                continue;
            }

            let mut triggered = false;
            for &sensor in rule.sensor_neurons.iter() {
                if sensor == NeuronId::INVALID {
                    continue;
                }
                let state = neuron_state_ref(sensor);
                if state.membrane_potential >= rule.threshold {
                    triggered = true;
                    break;
                }
            }

            if triggered {
                let motor_state = neuron_state(rule.motor_neuron);
                let attenuation = if self.cognitive_override {
                    FixedPoint::from_f32(0.3)
                } else {
                    FixedPoint::ONE
                };
                motor_state.membrane_potential += rule.intensity * attenuation;
                motor_state.flags.clear(NeuronFlags::REFRACTORY);
                self.active_reflexes |= 1 << i;

                if !self.cognitive_override {
                    crate::snn::neuron::neuromodulators().noradrenaline = FixedPoint::ONE;
                }
            }
        }
    }

    pub fn add_rule(&mut self, rule: ReflexRule) -> bool {
        if self.rule_count as usize >= self.reflex_rules.len() {
            return false;
        }
        self.reflex_rules[self.rule_count as usize] = rule;
        self.rule_count += 1;
        true
    }

    pub fn set_cognitive_override(&mut self, enabled: bool) {
        self.cognitive_override = enabled;
    }
}

impl ReflexRule {
    pub const fn new() -> Self {
        Self {
            sensor_neurons: [NeuronId::INVALID; 4],
            motor_neuron: NeuronId::INVALID,
            threshold: FixedPoint::ZERO,
            intensity: FixedPoint::ZERO,
            enabled: false,
            cognitive_override: false,
            description: [0; 32],
        }
    }
}

pub static mut REFLEXES: SpinalReflexes = SpinalReflexes::new();

pub fn reflexes() -> &'static mut SpinalReflexes {
    unsafe { &mut REFLEXES }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_sensor_motor(sensor_id: u16, motor_id: u16) {
        unsafe {
            let count = &crate::core::memory::NEURON_COUNT;
            count.store(2, core::sync::atomic::Ordering::Relaxed);
            let array = &mut crate::core::memory::NEURON_ARRAY;
            array[sensor_id as usize] =
                core::mem::MaybeUninit::new(crate::core::memory::NeuronState {
                    membrane_potential: FixedPoint::ZERO,
                    threshold: FixedPoint::from_f32(0.5),
                    leak: FixedPoint::ZERO,
                    refractory_remaining: 0,
                    last_spike_time: 0,
                    bias_current: FixedPoint::ZERO,
                    layer: 0,
                    neuron_type: crate::core::memory::NeuronType::REFLEX,
                    flags: crate::core::memory::NeuronFlags(0),
                });
            array[motor_id as usize] =
                core::mem::MaybeUninit::new(crate::core::memory::NeuronState {
                    membrane_potential: FixedPoint::ZERO,
                    threshold: FixedPoint::from_f32(1.0),
                    leak: FixedPoint::ZERO,
                    refractory_remaining: 0,
                    last_spike_time: 0,
                    bias_current: FixedPoint::ZERO,
                    layer: 4,
                    neuron_type: crate::core::memory::NeuronType::LIF,
                    flags: crate::core::memory::NeuronFlags(0),
                });
        }
    }

    #[test]
    fn test_reflex_new() {
        let r = SpinalReflexes::new();
        assert_eq!(r.rule_count, 0);
        assert!(!r.cognitive_override);
    }

    #[test]
    fn test_add_rule() {
        let mut r = SpinalReflexes::new();
        let rule = ReflexRule {
            sensor_neurons: [
                NeuronId::new(0),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(1),
            threshold: FixedPoint::from_f32(0.5),
            intensity: FixedPoint::from_f32(2.0),
            enabled: true,
            cognitive_override: true,
            description: [0; 32],
        };
        assert!(r.add_rule(rule));
        assert_eq!(r.rule_count, 1);
    }

    #[test]
    fn test_check_triggers_on_threshold() {
        setup_sensor_motor(0, 1);
        let mut r = SpinalReflexes::new();
        r.add_rule(ReflexRule {
            sensor_neurons: [
                NeuronId::new(0),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(1),
            threshold: FixedPoint::from_f32(0.3),
            intensity: FixedPoint::from_f32(2.0),
            enabled: true,
            cognitive_override: true,
            description: [0; 32],
        });
        let sensor = crate::core::memory::neuron_state(NeuronId::new(0));
        sensor.membrane_potential = FixedPoint::from_f32(0.8);
        r.check_all();
        let motor = crate::core::memory::neuron_state_ref(NeuronId::new(1));
        assert!(motor.membrane_potential > FixedPoint::ZERO);
        assert!(r.active_reflexes != 0);
    }

    #[test]
    fn test_cognitive_override_skips_rule() {
        setup_sensor_motor(0, 1);
        let mut r = SpinalReflexes::new();
        r.cognitive_override = true;
        r.add_rule(ReflexRule {
            sensor_neurons: [
                NeuronId::new(0),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(1),
            threshold: FixedPoint::from_f32(0.3),
            intensity: FixedPoint::from_f32(2.0),
            enabled: true,
            cognitive_override: true,
            description: [0; 32],
        });
        let sensor = crate::core::memory::neuron_state(NeuronId::new(0));
        sensor.membrane_potential = FixedPoint::from_f32(0.8);
        r.check_all();
        let motor = crate::core::memory::neuron_state_ref(NeuronId::new(1));
        assert_eq!(motor.membrane_potential, FixedPoint::ZERO);
        assert_eq!(r.active_reflexes, 0);
    }

    #[test]
    fn test_cognitive_override_blocks_na_injection() {
        setup_sensor_motor(0, 1);
        let mut r = SpinalReflexes::new();
        r.cognitive_override = true;
        r.add_rule(ReflexRule {
            sensor_neurons: [
                NeuronId::new(0),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(1),
            threshold: FixedPoint::from_f32(0.3),
            intensity: FixedPoint::from_f32(2.0),
            enabled: true,
            cognitive_override: true,
            description: [0; 32],
        });
        let sensor = crate::core::memory::neuron_state(NeuronId::new(0));
        sensor.membrane_potential = FixedPoint::from_f32(0.8);
        let nm = crate::snn::neuron::neuromodulators();
        nm.noradrenaline = FixedPoint::ZERO;
        r.check_all();
        assert_eq!(nm.noradrenaline, FixedPoint::ZERO);
    }

    #[test]
    fn test_set_cognitive_override() {
        let mut r = SpinalReflexes::new();
        assert!(!r.cognitive_override);
        r.set_cognitive_override(true);
        assert!(r.cognitive_override);
    }

    #[test]
    fn test_reflexes_accessor() {
        let r = reflexes();
        assert_eq!(r.rule_count, 0);
    }

    #[test]
    fn test_per_rule_override_flag() {
        let mut r = SpinalReflexes::new();
        r.add_rule(ReflexRule {
            sensor_neurons: [
                NeuronId::new(0),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(1),
            threshold: FixedPoint::from_f32(0.3),
            intensity: FixedPoint::from_f32(1.0),
            enabled: true,
            cognitive_override: false,
            description: [0; 32],
        });
        r.add_rule(ReflexRule {
            sensor_neurons: [
                NeuronId::new(2),
                NeuronId::INVALID,
                NeuronId::INVALID,
                NeuronId::INVALID,
            ],
            motor_neuron: NeuronId::new(3),
            threshold: FixedPoint::from_f32(0.3),
            intensity: FixedPoint::from_f32(1.0),
            enabled: true,
            cognitive_override: true,
            description: [0; 32],
        });
        assert!(!r.reflex_rules[0].cognitive_override);
        assert!(r.reflex_rules[1].cognitive_override);
    }
}
