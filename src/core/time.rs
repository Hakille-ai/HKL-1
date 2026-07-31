//! Metabolic clock providing biological time perception from 1 Hz to 1 MHz,
//! a time-warper for accelerated predictor simulation, and a five-scale
//! temporal hierarchy buffer for multi-timescale processing.

use core::sync::atomic::{AtomicU32, Ordering};



/// Hardware timer register offsets (generic)
const TIMER_CTRL: usize = 0;
const TIMER_PRESCALER: usize = 1;
const TIMER_COUNTER: usize = 2;
const TIMER_COMPARE: usize = 3;

/// Metabolic clock - provides biological time perception
/// Generates 1Hz "heartbeat" for temporal grounding
pub struct MetabolicClock {
    // Timer base address (memory-mapped)
    timer_base: *mut u32,
    // Tick counters at different frequencies
    pub tick_1hz: AtomicU32,   // 1 Hz - metabolic heartbeat
    pub tick_10hz: AtomicU32,  // 10 Hz - sensor fusion
    pub tick_100hz: AtomicU32, // 100 Hz - motor control
    pub tick_1khz: AtomicU32,  // 1 kHz - neural simulation
    pub tick_1mhz: AtomicU32,  // 1 MHz - high-res timing
    // High-resolution cycle counter
    cycles: AtomicU32,
    // Calibration
    cpu_freq_hz: u32,
    timer_freq_hz: u32,
    // Phase offsets for multi-scale temporal hierarchy (Section 33)
    phase_ultrafast: u32, // < 1ms
    phase_fast: u32,      // 1-100ms
    phase_medium: u32,    // 100ms-10s
    phase_slow: u32,      // 10s-1000s
    phase_ultraslow: u32, // > 1000s
}

impl MetabolicClock {
    pub const fn new() -> Self {
        Self {
            timer_base: core::ptr::null_mut(),
            tick_1hz: AtomicU32::new(0),
            tick_10hz: AtomicU32::new(0),
            tick_100hz: AtomicU32::new(0),
            tick_1khz: AtomicU32::new(0),
            tick_1mhz: AtomicU32::new(0),
            cycles: AtomicU32::new(0),
            cpu_freq_hz: 0,
            timer_freq_hz: 0,
            phase_ultrafast: 0,
            phase_fast: 0,
            phase_medium: 0,
            phase_slow: 0,
            phase_ultraslow: 0,
        }
    }

    /// Initialize hardware timer (called from boot sequence)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn init_hardware(&mut self, timer_base: *mut u32, cpu_freq: u32, timer_freq: u32) {
        self.timer_base = timer_base;
        self.cpu_freq_hz = cpu_freq;
        self.timer_freq_hz = timer_freq;

        if !timer_base.is_null() {
            unsafe {
                // Disable timer
                *timer_base.add(TIMER_CTRL) = 0;
                // Set prescaler for 1MHz base
                let prescaler = if timer_freq >= 1_000_000 {
                    timer_freq / 1_000_000
                } else {
                    1
                };
                *timer_base.add(TIMER_PRESCALER) = prescaler;
                // Set compare for 1kHz interrupt (every 1000 cycles at 1MHz)
                *timer_base.add(TIMER_COMPARE) = 1000;
                // Clear counter
                *timer_base.add(TIMER_COUNTER) = 0;
                // Enable timer with interrupt
                *timer_base.add(TIMER_CTRL) = 0b11; // Enable + Interrupt enable
            }
        }
    }

    pub fn isr_tick_1khz(&mut self) {
        let t = self.tick_1khz.fetch_add(1, Ordering::Relaxed) + 1;
        self.cycles
            .fetch_add(self.cpu_freq_hz / 1000, Ordering::Relaxed);

        // Derive other frequencies
        if t % 10 == 0 {
            self.tick_100hz.fetch_add(1, Ordering::Relaxed);
        }
        if t % 100 == 0 {
            self.tick_10hz.fetch_add(1, Ordering::Relaxed);
        }
        if t % 1000 == 0 {
            self.tick_1hz.fetch_add(1, Ordering::Relaxed);
            // Metabolic heartbeat - triggers pacemaker neurons
            crate::snn::network::trigger_metabolic_heartbeat();
        }

        // Update phase counters for temporal hierarchy
        self.phase_ultrafast = t % 1000;
        self.phase_fast = (t / 10) % 10000;
        self.phase_medium = (t / 1000) % 100000;
        self.phase_slow = (t / 10000) % 100000;
        self.phase_ultraslow = t / 10000000;
    }

    /// Get current time in milliseconds
    #[inline(always)]
    pub fn now_ms(&self) -> u32 {
        self.tick_1khz.load(Ordering::Relaxed)
    }

    /// Get current time in microseconds
    #[inline(always)]
    pub fn now_us(&self) -> u64 {
        self.cycles.load(Ordering::Relaxed) as u64 / (self.cpu_freq_hz as u64 / 1_000_000)
    }

    /// Get metabolic phase (0.0 - 1.0) for each temporal scale
    #[inline(always)]
    pub fn phase_ultrafast(&self) -> f32 {
        (self.phase_ultrafast as f32) / 1000.0
    }

    #[inline(always)]
    pub fn phase_fast(&self) -> f32 {
        (self.phase_fast as f32) / 10000.0
    }

    #[inline(always)]
    pub fn phase_medium(&self) -> f32 {
        (self.phase_medium as f32) / 100000.0
    }

    #[inline(always)]
    pub fn phase_slow(&self) -> f32 {
        (self.phase_slow as f32) / 100000.0
    }

    #[inline(always)]
    pub fn phase_ultraslow(&self) -> f32 {
        (self.phase_ultraslow as f32) / 100000.0
    }

    /// Get ticks since boot at specific frequency
    #[inline(always)]
    pub fn ticks_1hz(&self) -> u32 {
        self.tick_1hz.load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub fn ticks_10hz(&self) -> u32 {
        self.tick_10hz.load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub fn ticks_100hz(&self) -> u32 {
        self.tick_100hz.load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub fn ticks_1khz(&self) -> u32 {
        self.tick_1khz.load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub fn cycles(&self) -> u64 {
        self.cycles.load(Ordering::Relaxed) as u64
    }

    /// Sleep for specified milliseconds (busy wait - for bare metal)
    pub fn sleep_ms(&self, ms: u32) {
        let target = self.now_ms() + ms;
        while self.now_ms() < target {
            core::hint::spin_loop();
        }
    }

    /// Sleep for microseconds
    pub fn sleep_us(&self, us: u32) {
        let start = self.now_us();
        let target = start + us as u64;
        while self.now_us() < target {
            core::hint::spin_loop();
        }
    }

    /// Wait for next metabolic heartbeat (1Hz)
    pub fn wait_heartbeat(&self) {
        let current = self.ticks_1hz();
        while self.ticks_1hz() == current {
            core::hint::spin_loop();
        }
    }
}

/// Time-warping for predictor network (Section 6)
/// Accelerates simulation time for latent reasoning
pub struct TimeWarper {
    clock: &'static MetabolicClock,
    warp_factor: u32, // 1x = realtime, 1000x = 1000x speed
    simulated_ticks: u64,
    warp_active: bool,
}

impl TimeWarper {
    pub const fn new(clock: &'static MetabolicClock) -> Self {
        Self {
            clock,
            warp_factor: 1,
            simulated_ticks: 0,
            warp_active: false,
        }
    }

    /// Activate time warp for predictor simulation
    pub fn activate(&mut self, factor: u32) {
        self.warp_factor = factor.max(1);
        self.simulated_ticks = self.clock.ticks_1khz() as u64;
        self.warp_active = true;
    }

    /// Deactivate time warp
    pub fn deactivate(&mut self) {
        self.warp_active = false;
        self.warp_factor = 1;
    }

    /// Get simulated time (accelerated during warp)
    pub fn simulated_time_ms(&self) -> u64 {
        if self.warp_active {
            let real_elapsed = self.clock.ticks_1khz() as u64 - self.simulated_ticks;
            self.simulated_ticks + real_elapsed * self.warp_factor as u64
        } else {
            self.clock.ticks_1khz() as u64
        }
    }

    /// Step simulation by one simulated timestep
    pub fn step_simulation(&mut self) -> u64 {
        if self.warp_active {
            self.simulated_ticks += self.warp_factor as u64;
        } else {
            self.simulated_ticks = self.clock.ticks_1khz() as u64;
        }
        self.simulated_ticks
    }

    /// Check if warp is active
    pub fn is_warping(&self) -> bool {
        self.warp_active
    }
}

/// Temporal hierarchy buffers for multi-scale processing (Section 33)
pub struct TemporalHierarchy {
    // Buffers for each timescale
    pub ultrafast_buffer: [crate::core::math::FixedPoint; 1024], // < 1ms
    pub fast_buffer: [crate::core::math::FixedPoint; 1024],      // 1-100ms
    pub medium_buffer: [crate::core::math::FixedPoint; 1024],    // 100ms-10s
    pub slow_buffer: [crate::core::math::FixedPoint; 1024],      // 10s-1000s
    pub ultraslow_buffer: [crate::core::math::FixedPoint; 1024], // > 1000s

    // Write positions
    ultrafast_pos: AtomicU32,
    fast_pos: AtomicU32,
    medium_pos: AtomicU32,
    slow_pos: AtomicU32,
    ultraslow_pos: AtomicU32,
}

impl TemporalHierarchy {
    pub const fn new() -> Self {
        Self {
            ultrafast_buffer: [crate::core::math::FixedPoint::ZERO; 1024],
            fast_buffer: [crate::core::math::FixedPoint::ZERO; 1024],
            medium_buffer: [crate::core::math::FixedPoint::ZERO; 1024],
            slow_buffer: [crate::core::math::FixedPoint::ZERO; 1024],
            ultraslow_buffer: [crate::core::math::FixedPoint::ZERO; 1024],
            ultrafast_pos: AtomicU32::new(0),
            fast_pos: AtomicU32::new(0),
            medium_pos: AtomicU32::new(0),
            slow_pos: AtomicU32::new(0),
            ultraslow_pos: AtomicU32::new(0),
        }
    }

    pub fn record_spike(&mut self, value: crate::core::math::FixedPoint) {
        let uf = self.ultrafast_pos.fetch_add(1, Ordering::Relaxed) % 1024;
        self.ultrafast_buffer[uf as usize] = value;

        // Downsample to slower scales
        if uf % 10 == 0 {
            let f = self.fast_pos.fetch_add(1, Ordering::Relaxed) % 1024;
            self.fast_buffer[f as usize] = value;
        }
        if uf % 100 == 0 {
            let m = self.medium_pos.fetch_add(1, Ordering::Relaxed) % 1024;
            self.medium_buffer[m as usize] = value;
        }
        if uf % 1000 == 0 {
            let s = self.slow_pos.fetch_add(1, Ordering::Relaxed) % 1024;
            self.slow_buffer[s as usize] = value;
        }
        if uf % 10000 == 0 {
            let us = self.ultraslow_pos.fetch_add(1, Ordering::Relaxed) % 1024;
            self.ultraslow_buffer[us as usize] = value;
        }
    }

    /// Get recent activity at each timescale
    pub fn recent_activity(&self) -> [crate::core::math::FixedPoint; 5] {
        [
            self.ultrafast_buffer[(self.ultrafast_pos.load(Ordering::Relaxed) % 1024) as usize],
            self.fast_buffer[(self.fast_pos.load(Ordering::Relaxed) % 1024) as usize],
            self.medium_buffer[(self.medium_pos.load(Ordering::Relaxed) % 1024) as usize],
            self.slow_buffer[(self.slow_pos.load(Ordering::Relaxed) % 1024) as usize],
            self.ultraslow_buffer[(self.ultraslow_pos.load(Ordering::Relaxed) % 1024) as usize],
        ]
    }
}

/// Global metabolic clock instance
pub static mut METABOLIC_CLOCK: MetabolicClock = MetabolicClock::new();
pub static mut TIME_WARPER: TimeWarper = TimeWarper::new(unsafe { &METABOLIC_CLOCK });
pub static mut TEMPORAL_HIERARCHY: TemporalHierarchy = TemporalHierarchy::new();

pub fn init_clock(timer_base: *mut u32, cpu_freq: u32, timer_freq: u32) {
    unsafe {
        METABOLIC_CLOCK.init_hardware(timer_base, cpu_freq, timer_freq);
        TIME_WARPER = TimeWarper::new(&METABOLIC_CLOCK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::FixedPoint;

    #[test]
    fn metabolic_clock_new() {
        let clock = MetabolicClock::new();
        assert_eq!(clock.tick_1hz.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick_10hz.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick_100hz.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick_1khz.load(Ordering::Relaxed), 0);
        assert_eq!(clock.tick_1mhz.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn time_warper_new() {
        let clock = MetabolicClock::new();
        let clock_ref: &'static MetabolicClock =
            alloc::boxed::Box::leak(alloc::boxed::Box::new(clock));
        let warper = TimeWarper::new(clock_ref);
        assert_eq!(warper.warp_factor, 1);
        assert!(!warper.warp_active);
    }

    #[test]
    fn time_warper_activate_deactivate() {
        let clock = MetabolicClock::new();
        let clock_ref: &'static MetabolicClock =
            alloc::boxed::Box::leak(alloc::boxed::Box::new(clock));
        let mut warper = TimeWarper::new(clock_ref);
        warper.activate(100);
        assert!(warper.warp_active);
        warper.deactivate();
        assert!(!warper.warp_active);
    }

    #[test]
    fn temporal_hierarchy_new() {
        let th = TemporalHierarchy::new();
        assert_eq!(th.ultrafast_pos.load(Ordering::Relaxed), 0);
        assert_eq!(th.fast_pos.load(Ordering::Relaxed), 0);
        assert_eq!(th.medium_pos.load(Ordering::Relaxed), 0);
        assert_eq!(th.slow_pos.load(Ordering::Relaxed), 0);
        assert_eq!(th.ultraslow_pos.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn temporal_hierarchy_record() {
        let mut th = TemporalHierarchy::new();
        th.record_spike(FixedPoint::ZERO);
        assert_eq!(th.ultrafast_pos.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metabolic_clock_init() {
        let mut dummy_regs = [0u32; 4];
        let timer_base = dummy_regs.as_mut_ptr();
        let mut clock = MetabolicClock::new();
        clock.init_hardware(timer_base, 160_000_000, 16_000_000);
        assert_eq!(clock.cpu_freq_hz, 160_000_000);
    }
}
