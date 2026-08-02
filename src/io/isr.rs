//! Interrupt Service Routines — hardware interrupt → spike queue bridge.
//! NVIC (ARM Cortex-M7) and CLINT/PLIC (RISC-V) register definitions,
//! ISR handler bodies, and interrupt configuration.

#[allow(unused_imports)]
use crate::core::atomic::FetchAtomic;
use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(any(test, target_arch = "arm"))]
use crate::core::memory::NeuronId;
#[cfg(any(test, target_arch = "arm"))]
use crate::snn::neuron::SpikeEvent;

// ---------------------------------------------------------------------------
// NVIC (ARM Cortex-M7) — System Control Block registers
// ---------------------------------------------------------------------------
#[cfg(target_arch = "arm")]
pub mod nvic {
    use core::ptr::write_volatile;

    /// NVIC register block (STM32F7 reference)
    pub const NVIC_BASE: usize = 0xE000_E100;
    pub const NVIC_ISER0: *mut u32 = 0xE000_E100 as *mut u32; // IRQ 0-31
    pub const NVIC_ISER1: *mut u32 = 0xE000_E104 as *mut u32; // IRQ 32-63
    pub const NVIC_ICER0: *mut u32 = 0xE000_E180 as *mut u32;
    pub const NVIC_ICER1: *mut u32 = 0xE000_E184 as *mut u32;
    pub const NVIC_ISPR0: *mut u32 = 0xE000_E200 as *mut u32;
    pub const NVIC_IPR_BASE: usize = 0xE000_E400;

    /// System Control Block
    pub const SCB_ICSR: *mut u32 = 0xE000_ED04 as *mut u32;

    /// System Tick
    pub const SYST_CSR: *mut u32 = 0xE000_E010 as *mut u32;
    pub const SYST_RVR: *mut u32 = 0xE000_E014 as *mut u32;
    pub const SYST_CVR: *mut u32 = 0xE000_E018 as *mut u32;

    /// Enable interrupt in NVIC
    pub fn enable_irq(irq: u8) {
        if irq < 32 {
            unsafe {
                write_volatile(NVIC_ISER0, 1 << irq);
            }
        } else {
            unsafe {
                write_volatile(NVIC_ISER1, 1 << (irq - 32));
            }
        }
    }

    /// Disable interrupt in NVIC
    pub fn disable_irq(irq: u8) {
        if irq < 32 {
            unsafe {
                write_volatile(NVIC_ICER0, 1 << irq);
            }
        } else {
            unsafe {
                write_volatile(NVIC_ICER1, 1 << (irq - 32));
            }
        }
    }

    /// Set interrupt priority (0=highest, 15=lowest for 4-bit priority)
    pub fn set_priority(irq: u8, prio: u8) {
        let addr = (NVIC_IPR_BASE + (irq as usize)) as *mut u8;
        unsafe {
            write_volatile(addr, prio << 4);
        }
    }

    /// Set pending flag (for software-triggered interrupt)
    pub fn set_pending(irq: u8) {
        if irq < 32 {
            unsafe {
                write_volatile(NVIC_ISPR0, 1 << irq);
            }
        } else {
            // ISPR1 at NVIC_BASE + 0x200 + 0x04 for IRQ 32-63
            let ispr1 = 0xE000_E204 as *mut u32;
            unsafe {
                write_volatile(ispr1, 1 << (irq - 32));
            }
        }
    }

    /// Enable SysTick: 1ms period at 100 MHz → 100000 ticks
    pub fn enable_systick(reload: u32) {
        unsafe {
            write_volatile(SYST_RVR, reload & 0x00FF_FFFF);
            write_volatile(SYST_CVR, 0);
            // Enable SysTick: CLKSOURCE=1 (processor clock), TICKINT=1, ENABLE=1
            write_volatile(SYST_CSR, 0x07);
        }
    }
}

// ---------------------------------------------------------------------------
// STM32F7 interrupt numbers
// ---------------------------------------------------------------------------
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum IrqNumber {
    Wwdg = 0,
    Pvd = 1,
    TampStamp = 2,
    RtcWkup = 3,
    Flash = 4,
    Rcc = 5,
    Exti0 = 6,
    Exti1 = 7,
    Exti2 = 8,
    Exti3 = 9,
    Exti4 = 10,
    Dma1S0 = 11,
    Dma1S1 = 12,
    Dma1S2 = 13,
    Dma1S3 = 14,
    Dma1S4 = 15,
    Dma1S5 = 16,
    Dma1S6 = 17,
    Adc = 18,
    Tim2 = 28,
    Tim3 = 29,
    Tim4 = 30,
    I2c1Ev = 31,
    I2c1Er = 32,
    Spi1 = 35,
    Exti9_5 = 23,
    Tim5 = 46,
    Exti15_10 = 40,
    Dma2S0 = 56,
    Dma2S1 = 57,
    Dma2S2 = 58,
    Dma2S3 = 59,
    Dma2S4 = 60,
    Dma2S5 = 61,
    Dma2S6 = 62,
    Dma2S7 = 63,
}

// ---------------------------------------------------------------------------
// ISR dispatch flags (set from ISR, polled by main loop for deferred work)
// ---------------------------------------------------------------------------
pub static ISR_PENDING_FLAGS: AtomicU32 = AtomicU32::new(0);

pub const ISR_FLAG_TIM2: u32 = 1 << 0;
pub const ISR_FLAG_ADC: u32 = 1 << 1;
pub const ISR_FLAG_EXTI0: u32 = 1 << 2;
pub const ISR_FLAG_EXTI1: u32 = 1 << 3;
pub const ISR_FLAG_SPI1: u32 = 1 << 4;
pub const ISR_FLAG_I2C1: u32 = 1 << 5;
pub const ISR_FLAG_DMA: u32 = 1 << 6;
pub const ISR_FLAG_SYSTICK: u32 = 1 << 7;

/// Called from main loop to handle deferred ISR work
pub fn handle_pending_isrs() {
    let flags = take_pending_isr_flags();
    if flags == 0 {
        return;
    }

    if flags & ISR_FLAG_SYSTICK != 0 {
        on_systick();
    }
}

/// Atomically drain pending ISR flags without losing concurrently posted flags.
pub fn take_pending_isr_flags() -> u32 {
    ISR_PENDING_FLAGS.swap(0, Ordering::AcqRel)
}

// ---------------------------------------------------------------------------
// Hardware register base addresses for clearing interrupt flags
// ---------------------------------------------------------------------------
pub mod peripheral {
    use crate::io::sensors::mmio;
    use core::ptr::{read_volatile, write_volatile};

    pub const TIM2_SR: *mut u32 = 0x4000_0010 as *mut u32;
    pub const TIM2_SR_UIF_MASK: u32 = 1; // Update interrupt flag

    pub const ADC1_SR: *mut u32 = 0x4001_2000 as *mut u32;
    pub const ADC1_SR_EOC_MASK: u32 = 1 << 1; // End of conversion

    pub const EXTI_PR: *mut u32 = 0x4001_3C14 as *mut u32;

    pub const SPI1_SR: *mut u32 = 0x4001_3008 as *mut u32;
    pub const SPI1_DR: *mut u32 = 0x4001_300C as *mut u32;

    pub const I2C1_SR1: *mut u32 = 0x4000_5414 as *mut u32;

    /// STM32F7 TIM2 interrupt flag clear
    pub fn clear_tim2_uif() {
        unsafe {
            write_volatile(TIM2_SR, read_volatile(TIM2_SR) & !TIM2_SR_UIF_MASK);
        }
    }

    /// STM32F7 ADC end-of-conversion clear
    pub fn clear_adc_eoc() {
        // Read SR then DR to clear EOC
        unsafe {
            let _sr = read_volatile(ADC1_SR);
            let _dr = read_volatile(mmio::ADC1_DR);
        }
    }

    /// STM32F7 EXTI pending clear
    pub fn clear_exti_line(line: u8) {
        unsafe {
            write_volatile(EXTI_PR, 1 << line);
        }
    }

    /// STM32F7 SPI RXNE clear (read DR)
    pub fn clear_spi_rxne() -> u8 {
        unsafe { read_volatile(SPI1_DR) as u8 }
    }

    /// STM32F7 I2C interrupt clear (read SR1)
    pub fn clear_i2c_flags() {
        unsafe {
            let _sr1 = read_volatile(I2C1_SR1);
        }
    }
}

// ---------------------------------------------------------------------------
// ISR handler bodies — called from interrupt context or from dispatch
// ---------------------------------------------------------------------------

/// Push a spike event into GLOBAL_SPIKE_QUEUE from ISR context
#[cfg(any(test, target_arch = "arm"))]
fn isr_push_spike(neuron_id: u16, intensity: crate::core::math::FixedPoint, timestamp: u32) {
    let event = SpikeEvent {
        neuron_id: NeuronId::new(neuron_id),
        timestamp,
        layer: (intensity.to_f32() * 7.0) as u8, // Encode intensity as layer hint
        is_predictor: false,
    };
    unsafe {
        if let Some(ptr) = crate::io::buffers::GLOBAL_SPIKE_QUEUE.reserve_write() {
            core::ptr::write(ptr, event);
            crate::io::buffers::GLOBAL_SPIKE_QUEUE.commit_write();
        }
    }
}

#[cfg(target_arch = "arm")]
fn on_tim2_isr() {
    let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
    isr_push_spike(0xFFFF, crate::core::math::FixedPoint::ONE, now);
    peripheral::clear_tim2_uif();
}

#[cfg(target_arch = "arm")]
fn on_adc_isr() {
    if let Some(raw) = crate::io::sensors::AdcChannel::new(0).read_raw() {
        let intensity = crate::core::math::FixedPoint::from_f32((raw as f32) / 4095.0);
        let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
        isr_push_spike(0x100, intensity, now);
    }
    peripheral::clear_adc_eoc();
}

#[cfg(target_arch = "arm")]
fn on_exti0_isr() {
    let state = crate::io::sensors::read_gpio_pin(0);
    let intensity = if state {
        crate::core::math::FixedPoint::ONE
    } else {
        crate::core::math::FixedPoint::ZERO
    };
    let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
    isr_push_spike(0x200, intensity, now);
    peripheral::clear_exti_line(0);
}

#[cfg(target_arch = "arm")]
fn on_exti1_isr() {
    let state = crate::io::sensors::read_gpio_pin(1);
    let intensity = if state {
        crate::core::math::FixedPoint::ONE
    } else {
        crate::core::math::FixedPoint::ZERO
    };
    let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
    isr_push_spike(0x201, intensity, now);
    peripheral::clear_exti_line(1);
}

#[cfg(target_arch = "arm")]
fn on_spi1_isr() {
    let data = peripheral::clear_spi_rxne();
    let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
    isr_push_spike(
        0x300,
        crate::core::math::FixedPoint::from_f32((data as f32) / 255.0),
        now,
    );
}

#[cfg(target_arch = "arm")]
fn on_i2c1_isr() {
    peripheral::clear_i2c_flags();
    let now = unsafe { crate::core::time::METABOLIC_CLOCK.now_us() as u32 };
    isr_push_spike(0x400, crate::core::math::FixedPoint::ONE, now);
}

/// SysTick handler — periodic 1ms tick
fn on_systick() {
    unsafe {
        crate::core::time::METABOLIC_CLOCK.isr_tick_1khz();
    }
}

// ---------------------------------------------------------------------------
// Cortex-M7 interrupt handler symbols (linked by vector table)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "arm")]
mod handlers {
    use super::*;

    #[unsafe(no_mangle)]
    pub extern "C" fn TIM2_IRQHandler() {
        on_tim2_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ADC_IRQHandler() {
        on_adc_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn EXTI0_IRQHandler() {
        on_exti0_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn EXTI1_IRQHandler() {
        on_exti1_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn SPI1_IRQHandler() {
        on_spi1_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn I2C1_EV_IRQHandler() {
        on_i2c1_isr();
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn SysTick_Handler() {
        ISR_PENDING_FLAGS.fetch_or(ISR_FLAG_SYSTICK, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// ISR interrupt configuration
// ---------------------------------------------------------------------------
pub struct IsrConfig {
    pub timer_interval_us: u32,
    pub enable_adc: bool,
    pub enable_exti: bool,
    pub enable_spi: bool,
    pub enable_i2c: bool,
}

impl Default for IsrConfig {
    fn default() -> Self {
        Self {
            timer_interval_us: 1000,
            enable_adc: true,
            enable_exti: true,
            enable_spi: false,
            enable_i2c: false,
        }
    }
}

/// Initialize interrupt system
#[cfg(target_arch = "arm")]
fn init_isr_arm(config: &IsrConfig) {
    // Enable SysTick at 1ms (100 MHz / 100000)
    nvic::enable_systick(100_000);

    // Configure TIM2 for periodic interrupt
    if config.timer_interval_us > 0 {
        configure_tim2_interrupt(config.timer_interval_us);
        nvic::enable_irq(IrqNumber::Tim2 as u8);
        nvic::set_priority(IrqNumber::Tim2 as u8, 2);
    }

    // Enable ADC interrupt
    if config.enable_adc {
        nvic::enable_irq(IrqNumber::Adc as u8);
        nvic::set_priority(IrqNumber::Adc as u8, 3);
    }

    // Enable EXTI0/EXTI1 for GPIO button/sensor
    if config.enable_exti {
        nvic::enable_irq(IrqNumber::Exti0 as u8);
        nvic::set_priority(IrqNumber::Exti0 as u8, 4);
        nvic::enable_irq(IrqNumber::Exti1 as u8);
        nvic::set_priority(IrqNumber::Exti1 as u8, 4);
    }

    // Enable SPI1 interrupt
    if config.enable_spi {
        nvic::enable_irq(IrqNumber::Spi1 as u8);
        nvic::set_priority(IrqNumber::Spi1 as u8, 3);
    }

    // Enable I2C1 event interrupt
    if config.enable_i2c {
        nvic::enable_irq(IrqNumber::I2c1Ev as u8);
        nvic::set_priority(IrqNumber::I2c1Ev as u8, 3);
    }
}

pub fn init_isr_system(_config: &IsrConfig) {
    #[cfg(target_arch = "arm")]
    init_isr_arm(_config);
}

/// Configure TIM2 for periodic interrupt
#[cfg(target_arch = "arm")]
fn configure_tim2_interrupt(interval_us: u32) {
    #[cfg(target_arch = "arm")]
    {
        use core::ptr::write_volatile;
        // TIM2 base: 0x4000_0000
        const TIM2_PSC: *mut u32 = 0x4000_0028 as *mut u32;
        const TIM2_ARR: *mut u32 = 0x4000_002C as *mut u32;
        const TIM2_DIER: *mut u32 = 0x4000_000C as *mut u32;
        const TIM2_CR1: *mut u32 = 0x4000_0000 as *mut u32;
        const TIM2_EGR: *mut u32 = 0x4000_0014 as *mut u32;

        unsafe {
            write_volatile(TIM2_CR1, 0);

            // Prescaler: 100 MHz / 1000 = 100 kHz (timer clock)
            write_volatile(TIM2_PSC, 1000 - 1);

            // Auto-reload: number of ticks at 100 kHz for desired interval
            let arr = (interval_us as u64 * 100_000_000 / 1_000_000 / 1000) as u32;
            write_volatile(TIM2_ARR, arr.saturating_sub(1));

            // Enable update interrupt (UIE)
            write_volatile(TIM2_DIER, 1);

            // Generate update event to load registers
            write_volatile(TIM2_EGR, 1);

            // Enable timer
            write_volatile(TIM2_CR1, 1);
        }
    }
}

/// Software-triggered interrupt (for testing ISR path on host)
pub fn software_isr_trigger(flag: u32) {
    ISR_PENDING_FLAGS.fetch_or(flag, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static ISR_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn lock_isr_state() -> MutexGuard<'static, ()> {
        ISR_TEST_LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn drain_global_spike_queue() {
        unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.clear() };
    }

    #[test]
    fn isr_pending_flags_default() {
        let _guard = lock_isr_state();
        ISR_PENDING_FLAGS.store(0, Ordering::SeqCst);
        assert_eq!(ISR_PENDING_FLAGS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn software_isr_triggers_systick() {
        let _guard = lock_isr_state();
        ISR_PENDING_FLAGS.store(0, Ordering::SeqCst);
        software_isr_trigger(ISR_FLAG_SYSTICK);
        assert_eq!(ISR_PENDING_FLAGS.load(Ordering::Relaxed), ISR_FLAG_SYSTICK);
    }

    #[test]
    fn take_pending_flags_drains_once() {
        let _guard = lock_isr_state();
        ISR_PENDING_FLAGS.store(0, Ordering::SeqCst);
        software_isr_trigger(ISR_FLAG_SYSTICK | ISR_FLAG_ADC);

        let flags = take_pending_isr_flags();

        assert_eq!(flags, ISR_FLAG_SYSTICK | ISR_FLAG_ADC);
        assert_eq!(take_pending_isr_flags(), 0);
    }

    #[test]
    fn isr_push_spike_to_queue() {
        let _guard = lock_isr_state();
        drain_global_spike_queue();
        assert!(unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }.is_none());

        // Push via ISR path
        isr_push_spike(42, crate::core::math::FixedPoint::from_f32(0.5), 100);

        // Verify spike arrived
        let spike = unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() };
        assert!(spike.is_some());
        if let Some(s) = spike {
            assert_eq!(s.neuron_id.index(), 42);
            assert_eq!(s.timestamp, 100);
        }
        drain_global_spike_queue();
    }

    #[test]
    fn handle_pending_systick() {
        let _guard = lock_isr_state();
        ISR_PENDING_FLAGS.store(0, Ordering::SeqCst);
        software_isr_trigger(ISR_FLAG_SYSTICK);
        assert!(ISR_PENDING_FLAGS.load(Ordering::Relaxed) != 0);
        handle_pending_isrs();
        assert_eq!(ISR_PENDING_FLAGS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn handle_pending_no_flags() {
        let _guard = lock_isr_state();
        ISR_PENDING_FLAGS.store(0, Ordering::SeqCst);
        handle_pending_isrs(); // Should not panic
    }

    #[test]
    fn isr_config_default() {
        let cfg = IsrConfig::default();
        assert_eq!(cfg.timer_interval_us, 1000);
        assert!(cfg.enable_adc);
        assert!(cfg.enable_exti);
    }

    #[test]
    fn irq_number_values() {
        assert_eq!(IrqNumber::Tim2 as u8, 28);
        assert_eq!(IrqNumber::Adc as u8, 18);
        assert_eq!(IrqNumber::Exti0 as u8, 6);
        assert_eq!(IrqNumber::Spi1 as u8, 35);
    }

    #[test]
    fn multiple_isr_spikes_in_queue() {
        let _guard = lock_isr_state();
        drain_global_spike_queue();

        isr_push_spike(1, crate::core::math::FixedPoint::ZERO, 10);
        isr_push_spike(2, crate::core::math::FixedPoint::ONE, 20);
        isr_push_spike(3, crate::core::math::FixedPoint::from_f32(0.5), 30);

        assert!(unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }.is_some());
        assert!(unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }.is_some());
        assert!(unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }.is_some());
        assert!(unsafe { crate::io::buffers::GLOBAL_SPIKE_QUEUE.pop_front() }.is_none());
        drain_global_spike_queue();
    }
}
