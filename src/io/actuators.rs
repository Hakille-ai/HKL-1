//! Actuator interfaces with hardware MMIO register writes.
//! Supports PWM, GPIO, and DAC output (STM32F7 addresses by default).

use core::mem::MaybeUninit;
#[cfg(not(any(feature = "std", test)))]
use core::ptr::{read_volatile, write_volatile};

// ---------------------------------------------------------------------------
// MMIO register addresses (STM32F746 reference)
// ---------------------------------------------------------------------------
pub mod mmio {
    // TIM2 (general purpose timer for PWM)
    pub const TIM2_BASE: usize = 0x4000_0000;
    pub const TIM2_CR1: *mut u32 = 0x4000_0000 as *mut u32;
    pub const TIM2_CCMR1: *mut u32 = 0x4000_0018 as *mut u32;
    pub const TIM2_CCER: *mut u32 = 0x4000_0020 as *mut u32;
    pub const TIM2_ARR: *mut u32 = 0x4000_002C as *mut u32;
    pub const TIM2_CCR1: *mut u32 = 0x4000_0034 as *mut u32;
    pub const TIM2_CCR2: *mut u32 = 0x4000_0038 as *mut u32;
    pub const TIM2_CCR3: *mut u32 = 0x4000_003C as *mut u32;
    pub const TIM2_CCR4: *mut u32 = 0x4000_0040 as *mut u32;
    pub const TIM2_EGR: *mut u32 = 0x4000_0014 as *mut u32;

    // DAC
    pub const DAC_BASE: usize = 0x4000_7400;
    pub const DAC_CR: *mut u32 = 0x4000_7400 as *mut u32;
    pub const DAC_DHR12R1: *mut u32 = 0x4000_7408 as *mut u32;
    pub const DAC_DHR12R2: *mut u32 = 0x4000_7414 as *mut u32;
    pub const DAC_SWTRIGR: *mut u32 = 0x4000_7404 as *mut u32;

    // GPIOB (actuator outputs)
    pub const GPIOB_BASE: usize = 0x4002_0400;
    pub const GPIOB_MODER: *mut u32 = 0x4002_0400 as *mut u32;
    pub const GPIOB_ODR: *mut u32 = 0x4002_0414 as *mut u32;
    pub const GPIOB_BSRR: *mut u32 = 0x4002_0418 as *mut u32;
    pub const GPIOB_AFRL: *mut u32 = 0x4002_0420 as *mut u32;
}

// ---------------------------------------------------------------------------
// PWM generator
// ---------------------------------------------------------------------------
pub struct PwmGenerator {
    _channel: u8,
    _period: u16,
}

impl PwmGenerator {
    pub const fn new(channel: u8) -> Self {
        Self {
            _channel: channel,
            _period: 1000,
        }
    }

    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Enable TIM2: set CEN bit, configure edge-aligned PWM mode 1
            let cr1 = 1 << 0; // CEN
            write_volatile(mmio::TIM2_CR1, cr1);

            // Set auto-reload (period)
            write_volatile(mmio::TIM2_ARR, self._period as u32);

            // Configure channel as PWM mode 1 (OCxM=110, CCxS=00)
            let ccmr_shift = if self._channel < 2 { 0 } else { 8 };
            let ch_idx = self._channel % 2;
            let oc_feild = (6 << 4) | (1 << 3); // PWM mode 1 + preload
            let ccmr_val = read_volatile(mmio::TIM2_CCMR1);
            write_volatile(
                mmio::TIM2_CCMR1,
                ccmr_val | (oc_feild << (ccmr_shift + ch_idx * 8)),
            );

            // Enable output compare
            let cc_en = 1 << (self._channel * 4);
            let ccer_val = read_volatile(mmio::TIM2_CCER);
            write_volatile(mmio::TIM2_CCER, ccer_val | cc_en);

            // Generate update event to load registers
            write_volatile(mmio::TIM2_EGR, 1);
        }
    }

    pub fn set_duty(&self, duty_cycle: f32) {
        let duty = (duty_cycle.clamp(0.0, 1.0) * self._period as f32) as u16;
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            match self._channel {
                0 => write_volatile(mmio::TIM2_CCR1, duty as u32),
                1 => write_volatile(mmio::TIM2_CCR2, duty as u32),
                2 => write_volatile(mmio::TIM2_CCR3, duty as u32),
                3 => write_volatile(mmio::TIM2_CCR4, duty as u32),
                _ => {}
            }
        }
        #[cfg(any(feature = "std", test))]
        let _ = duty;
    }
}

// ---------------------------------------------------------------------------
// GPIO pin controller
// ---------------------------------------------------------------------------
pub struct GpioPin {
    _pin: u8,
    _output_mode: bool,
}

impl GpioPin {
    pub const fn new(pin: u8, output_mode: bool) -> Self {
        Self {
            _pin: pin,
            _output_mode: output_mode,
        }
    }

    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            // Set MODER: 01=output, 00=input
            let moder = read_volatile(mmio::GPIOB_MODER);
            let shift = self._pin as u32 * 2;
            let cleared = moder & !(0b11 << shift);
            let set = if self._output_mode {
                0b01 << shift
            } else {
                0b00 << shift
            };
            write_volatile(mmio::GPIOB_MODER, cleared | set);

            if self._output_mode {
                // Alternate function low register (AFRL) — default AF0
                if self._pin < 8 {
                    let afrl = read_volatile(mmio::GPIOB_AFRL);
                    write_volatile(mmio::GPIOB_AFRL, afrl & !(0b1111 << (self._pin as u32 * 4)));
                }
            }
        }
    }

    pub fn set(&self, state: bool) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            if state {
                write_volatile(mmio::GPIOB_BSRR, 1 << self._pin);
            } else {
                write_volatile(mmio::GPIOB_BSRR, 1 << (self._pin + 16));
            }
        }
        #[cfg(any(feature = "std", test))]
        let _ = state;
    }

    pub fn read(&self) -> bool {
        #[cfg(any(feature = "std", test))]
        {
            return false;
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            (read_volatile(mmio::GPIOB_ODR) & (1 << self._pin)) != 0
        }
    }
}

// ---------------------------------------------------------------------------
// DAC output
// ---------------------------------------------------------------------------
pub struct DacOutput {
    _channel: u8, // 0=DAC1_OUT, 1=DAC2_OUT
}

impl DacOutput {
    pub const fn new(channel: u8) -> Self {
        Self { _channel: channel }
    }

    pub fn init(&self) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let cr = read_volatile(mmio::DAC_CR);
            let mask = match self._channel {
                0 => 0b11,       // EN1 + BOFF1
                1 => 0b11 << 16, // EN2 + BOFF2
                _ => 0,
            };
            write_volatile(mmio::DAC_CR, cr | mask);
        }
    }

    pub fn set_voltage(&self, voltage: f32) {
        let value = (voltage.clamp(0.0, 3.3) / 3.3 * 4095.0) as u16;
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            match self._channel {
                0 => write_volatile(mmio::DAC_DHR12R1, value as u32),
                1 => write_volatile(mmio::DAC_DHR12R2, value as u32),
                _ => {}
            }
            // Software trigger
            write_volatile(mmio::DAC_SWTRIGR, 1);
        }
        #[cfg(any(feature = "std", test))]
        let _ = value;
    }
}

// ---------------------------------------------------------------------------
// ActuatorManager — orchestrates all outputs
// ---------------------------------------------------------------------------
pub struct ActuatorManager {
    pub pwm_outputs: [f32; 16],
    pub gpio_states: [bool; 32],
    pub dac_outputs: [f32; 2],
    pwm_generators: [PwmGenerator; 4],
    gpio_pins: [GpioPin; 8],
    dac: [DacOutput; 2],
    initialized: bool,
}

impl ActuatorManager {
    pub fn new() -> Self {
        Self {
            pwm_outputs: [0.0; 16],
            gpio_states: [false; 32],
            dac_outputs: [0.0; 2],
            pwm_generators: [
                PwmGenerator::new(0),
                PwmGenerator::new(1),
                PwmGenerator::new(2),
                PwmGenerator::new(3),
            ],
            gpio_pins: [
                GpioPin::new(0, true),
                GpioPin::new(1, true),
                GpioPin::new(2, true),
                GpioPin::new(3, true),
                GpioPin::new(4, true),
                GpioPin::new(5, true),
                GpioPin::new(6, true),
                GpioPin::new(7, true),
            ],
            dac: [DacOutput::new(0), DacOutput::new(1)],
            initialized: false,
        }
    }

    pub fn init(&mut self) {
        for pwm in &self.pwm_generators {
            pwm.init();
        }
        for pin in &self.gpio_pins {
            pin.init();
        }
        self.dac[0].init();
        self.dac[1].init();
        self.initialized = true;
    }

    /// Write a PWM channel with duty cycle [0.0 – 1.0]
    pub fn set_pwm(&mut self, channel: usize, duty: f32) {
        if channel < 16 {
            self.pwm_outputs[channel] = duty;
            if self.initialized && channel < 4 {
                let duty = duty.clamp(0.0, 1.0);
                self.pwm_generators[channel].set_duty(duty);
            }
        }
    }

    /// Set GPIO pin state
    pub fn set_gpio(&mut self, pin: usize, state: bool) {
        if pin < 32 {
            self.gpio_states[pin] = state;
            if self.initialized && pin < 8 {
                self.gpio_pins[pin].set(state);
            }
        }
    }

    /// Set DAC output voltage (0.0 – 3.3V)
    pub fn set_dac(&mut self, channel: usize, voltage: f32) {
        if channel < 2 {
            self.dac_outputs[channel] = voltage;
            if self.initialized {
                self.dac[channel].set_voltage(voltage);
            }
        }
    }

    /// Read motor neuron outputs and apply to actuators
    pub fn read_motor_outputs(&mut self) {
        let count = crate::core::memory::NEURON_COUNT.load(core::sync::atomic::Ordering::Relaxed);
        for i in 0..count.min(16) as u16 {
            let id = crate::core::memory::NeuronId::new(i);
            let state = crate::core::memory::neuron_state_ref(id);
            if state.layer == 4 {
                let potential = state.membrane_potential.to_f32();
                self.set_pwm(i as usize, potential.abs().clamp(0.0, 1.0));
                self.set_gpio(i as usize, potential > 0.0);
            }
        }
    }
}

/// Global actuator manager instance
pub static mut ACTUATOR_MANAGER: MaybeUninit<ActuatorManager> = MaybeUninit::uninit();

static INITIALIZED_ACTUATOR_MANAGER: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_actuator_manager() {
    unsafe {
        let am = ACTUATOR_MANAGER.write(ActuatorManager::new());
        am.init();
        INITIALIZED_ACTUATOR_MANAGER.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn actuator_manager() -> &'static mut ActuatorManager {
    unsafe {
        if !INITIALIZED_ACTUATOR_MANAGER.load(core::sync::atomic::Ordering::Relaxed) {
            init_actuator_manager();
        }
        &mut *ACTUATOR_MANAGER.as_mut_ptr()
    }
}
