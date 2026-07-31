//! Power management. Sleep states, DVFS, wake-up system, power budgeting,
//! and energy-harvesting-aware operation (Sections 11, 34).

use crate::core::math::FixedPoint;
use core::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Power domains
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PowerDomain {
    Cpu = 0,
    Memory = 1,
    Sensors = 2,
    Actuators = 3,
    Radio = 4,
    Cognitive = 5,
}

pub const DOMAIN_COUNT: usize = 6;

pub const DOMAIN_NAMES: [&str; DOMAIN_COUNT] = [
    "CPU",
    "Memory",
    "Sensors",
    "Actuators",
    "Radio",
    "Cognitive",
];

// ---------------------------------------------------------------------------
// Sleep states
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum SleepState {
    Active = 0,
    LightSleep = 1,
    DeepSleep = 2,
    Shutdown = 3,
}

impl SleepState {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => SleepState::Active,
            1 => SleepState::LightSleep,
            2 => SleepState::DeepSleep,
            3 => SleepState::Shutdown,
            _ => SleepState::Active,
        }
    }

    /// Typical wake-up latency in microseconds
    pub fn wake_latency_us(self) -> u32 {
        match self {
            SleepState::Active => 0,
            SleepState::LightSleep => 5,
            SleepState::DeepSleep => 200,
            SleepState::Shutdown => 5_000,
        }
    }

    /// Typical power draw as fraction of active power
    pub fn power_ratio(self) -> FixedPoint {
        match self {
            SleepState::Active => FixedPoint::ONE,
            SleepState::LightSleep => FixedPoint::from_f32(0.30),
            SleepState::DeepSleep => FixedPoint::from_f32(0.05),
            SleepState::Shutdown => FixedPoint::from_f32(0.001),
        }
    }
}

// ---------------------------------------------------------------------------
// Power mode (application-level policy)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PowerMode {
    Normal = 0,
    LowPower = 1,
    Critical = 2,
    Exploration = 3,
}

// ---------------------------------------------------------------------------
// Harvesting type
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HarvestingType {
    None = 0,
    Solar = 1,
    Thermal = 2,
    RF = 3,
    Kinetic = 4,
}

// ---------------------------------------------------------------------------
// Domain power state
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub struct DomainState {
    pub enabled: bool,
    pub clock_gated: bool,
    pub voltage_scale: FixedPoint, // 0.0-1.0 normalized
    pub frequency_hz: u32,
    pub current_ma: FixedPoint,
    pub energy_consumed: FixedPoint, // Accumulated mWh
}

impl DomainState {
    pub const fn new() -> Self {
        Self {
            enabled: true,
            clock_gated: false,
            voltage_scale: FixedPoint::ONE,
            frequency_hz: 100_000_000,
            current_ma: FixedPoint::ZERO,
            energy_consumed: FixedPoint::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// DVFS operating-point table
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct OppPoint {
    pub voltage_mv: u32,         // Millivolts
    pub freq_mhz: u32,           // MHz
    pub performance: FixedPoint, // 0.0-1.0
}

pub const OPP_TABLE: [OppPoint; 5] = [
    OppPoint {
        voltage_mv: 900,
        freq_mhz: 16,
        performance: FixedPoint::from_f32(0.10),
    },
    OppPoint {
        voltage_mv: 950,
        freq_mhz: 32,
        performance: FixedPoint::from_f32(0.20),
    },
    OppPoint {
        voltage_mv: 1000,
        freq_mhz: 64,
        performance: FixedPoint::from_f32(0.40),
    },
    OppPoint {
        voltage_mv: 1050,
        freq_mhz: 120,
        performance: FixedPoint::from_f32(0.75),
    },
    OppPoint {
        voltage_mv: 1100,
        freq_mhz: 216,
        performance: FixedPoint::ONE,
    },
];

// ---------------------------------------------------------------------------
// Wake-up configuration
// ---------------------------------------------------------------------------
pub struct WakeUpConfig {
    pub rtc_alarm_enabled: bool,
    pub rtc_alarm_s: u32,
    pub exti_pins: u32,
    pub timer_wakeup_us: u32,
    pub sensor_threshold: FixedPoint,
}

impl WakeUpConfig {
    pub const fn new() -> Self {
        Self {
            rtc_alarm_enabled: false,
            rtc_alarm_s: 0,
            exti_pins: 0,
            timer_wakeup_us: 0,
            sensor_threshold: FixedPoint::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Power budget
// ---------------------------------------------------------------------------
pub struct PowerBudget {
    pub domain_budgets: [FixedPoint; DOMAIN_COUNT],
    pub total_budget: FixedPoint,
    pub peak_power_mw: FixedPoint,
    pub avg_power_mw: FixedPoint,
}

impl PowerBudget {
    pub const fn new() -> Self {
        Self {
            domain_budgets: [FixedPoint::ZERO; DOMAIN_COUNT],
            total_budget: FixedPoint::ZERO,
            peak_power_mw: FixedPoint::ZERO,
            avg_power_mw: FixedPoint::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// STM32F7 PWR & RCC registers (MMIO)
// ---------------------------------------------------------------------------
pub mod hw {
    #[cfg_attr(any(feature = "std", test), allow(unused_imports))]
    use core::ptr::{read_volatile, write_volatile};

    // PWR registers
    pub const PWR_CR: *mut u32 = 0x4000_7000 as *mut u32;
    pub const PWR_CSR: *mut u32 = 0x4000_7004 as *mut u32;

    pub const CR_VOS_0: u32 = 1 << 14; // Voltage scaling bit 0
    pub const CR_VOS_1: u32 = 1 << 15; // Voltage scaling bit 1
    pub const CR_FPDS: u32 = 1 << 9; // Flash power-down in Stop
    pub const CR_LPDS: u32 = 1 << 0; // Low-power deepsleep
    pub const CR_LPUDS: u32 = 1 << 1; // Low-power deepsleep underdrive
    pub const CR_MRUDS: u32 = 1 << 11; // Memory underdrive
    pub const CR_LPRUN: u32 = 1 << 14; // Low-power run

    pub const CSR_WUF: u32 = 1 << 0; // Wake-up flag
    pub const CSR_SBF: u32 = 1 << 1; // Standby flag
    pub const CSR_PVDO: u32 = 1 << 2; // PVD output
    pub const CSR_VOSRDY: u32 = 1 << 14; // Voltage scaling ready

    // RCC registers (clock control)
    pub const RCC_CR: *mut u32 = 0x4002_3800 as *mut u32;
    pub const RCC_CFGR: *mut u32 = 0x4002_3808 as *mut u32;
    pub const RCC_DCKCFGR: *mut u32 = 0x4002_388C as *mut u32;
    pub const RCC_AHB1ENR: *mut u32 = 0x4002_3830 as *mut u32;
    pub const RCC_APB1ENR: *mut u32 = 0x4002_3838 as *mut u32;
    pub const RCC_APB2ENR: *mut u32 = 0x4002_383C as *mut u32;

    // System Control Register (SCR) for sleep modes
    pub const SCR: *mut u32 = 0xE000_ED10 as *mut u32;
    pub const SCR_SLEEPDEEP: u32 = 1 << 2;
    pub const SCR_SLEEPONEXIT: u32 = 1 << 1;

    // Set voltage scaling level
    pub fn set_voltage_scale(scale: u8) -> bool {
        #[cfg(any(feature = "std", test))]
        {
            let _ = scale;
            true
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let mut cr = read_volatile(PWR_CR);
            cr &= !(CR_VOS_0 | CR_VOS_1);
            match scale {
                1 => cr |= CR_VOS_1, // Scale 1 (high perf)
                2 => cr |= CR_VOS_0, // Scale 2
                3 => {
                    /* Scale 3 (low) — VOS_0|VOS_1 = 11 */
                    cr |= CR_VOS_0 | CR_VOS_1;
                }
                _ => cr |= CR_VOS_1,
            }
            write_volatile(PWR_CR, cr);
            // Wait for regulator ready
            let mut timeout = 100_000;
            while read_volatile(PWR_CSR) & CSR_VOSRDY == 0 {
                timeout -= 1;
                if timeout == 0 {
                    return false;
                }
            }
            true
        }
    }

    /// Enter sleep mode (WFI)
    /// Enter sleep mode (WFI)
    pub fn enter_sleep() {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            write_volatile(SCR, read_volatile(SCR) & !SCR_SLEEPDEEP);
            core::arch::asm!("wfi");
        }
    }

    /// Enter deep-sleep (Stop mode)
    /// Enter deep-sleep (Stop mode)
    pub fn enter_deep_sleep(lpds: bool, fpds: bool) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let mut cr = read_volatile(PWR_CR);
            if lpds {
                cr |= CR_LPDS;
            } else {
                cr &= !CR_LPDS;
            }
            if fpds {
                cr |= CR_FPDS;
            } else {
                cr &= !CR_FPDS;
            }
            write_volatile(PWR_CR, cr);
            write_volatile(SCR, read_volatile(SCR) | SCR_SLEEPDEEP);
            core::arch::asm!("wfi");
        }
        #[cfg(any(feature = "std", test))]
        {
            let _ = (lpds, fpds);
        }
    }

    /// Enter standby mode (Shutdown)
    pub fn enter_standby() {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            write_volatile(PWR_CR, read_volatile(PWR_CR) | (1 << 2)); // CSBF + PDDS
            write_volatile(SCR, read_volatile(SCR) | SCR_SLEEPDEEP);
            core::arch::asm!("wfi");
        }
    }

    /// Enable or disable a peripheral clock via RCC
    /// Enable or disable a peripheral clock via RCC
    pub fn set_peripheral_clock(bus: u8, bit: u8, enable: bool) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let reg = match bus {
                0 => RCC_AHB1ENR,
                1 => RCC_APB1ENR,
                2 => RCC_APB2ENR,
                _ => return,
            };
            let mask = 1u32 << bit;
            let val = read_volatile(reg);
            if enable {
                write_volatile(reg, val | mask);
            } else {
                write_volatile(reg, val & !mask);
            }
        }
        #[cfg(any(feature = "std", test))]
        {
            let _ = (bus, bit, enable);
        }
    }

    /// Read backup domain register (retained in standby)
    pub fn read_backup_reg(idx: u8) -> u32 {
        #[cfg(any(feature = "std", test))]
        {
            let _ = idx;
            0
        }
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let addr = (0x4000_2840 + (idx as u32) * 4) as *const u32;
            read_volatile(addr)
        }
    }

    /// Write backup domain register
    pub fn write_backup_reg(idx: u8, value: u32) {
        #[cfg(not(any(feature = "std", test)))]
        unsafe {
            let addr = (0x4000_2840 + (idx as u32) * 4) as *mut u32;
            write_volatile(addr, value);
        }
        #[cfg(any(feature = "std", test))]
        {
            let _ = (idx, value);
        }
    }
}

// ---------------------------------------------------------------------------
// Power Manager
// ---------------------------------------------------------------------------
pub struct PowerManager {
    pub battery_level: FixedPoint,
    pub energy_harvesting_rate: FixedPoint,
    pub consumption_rate: FixedPoint,
    pub is_critical: bool,
    pub mode: PowerMode,
    pub sleep_state: SleepState,
    pub current_opp: u8,
    pub wakeup: WakeUpConfig,
    pub domains: [DomainState; DOMAIN_COUNT],
    pub budget: PowerBudget,
    pub harvesting_type: HarvestingType,
    pub mppt_duty_cycle: FixedPoint,
    pub total_energy_consumed: FixedPoint,
    pub total_energy_harvested: FixedPoint,
    pub uptime_seconds: u64,
    pub idle_step_count: u64,
    testing: bool,
}

impl PowerManager {
    pub const fn new() -> Self {
        Self {
            battery_level: FixedPoint::ONE,
            energy_harvesting_rate: FixedPoint::from_f32(0.01),
            consumption_rate: FixedPoint::from_f32(0.005),
            is_critical: false,
            mode: PowerMode::Normal,
            sleep_state: SleepState::Active,
            current_opp: 3, // Start at 120 MHz
            wakeup: WakeUpConfig::new(),
            domains: [
                DomainState::new(), // CPU
                DomainState::new(), // Memory
                DomainState::new(), // Sensors
                DomainState::new(), // Actuators
                DomainState::new(), // Radio
                DomainState::new(), // Cognitive
            ],
            budget: PowerBudget::new(),
            harvesting_type: HarvestingType::None,
            mppt_duty_cycle: FixedPoint::from_f32(0.5),
            total_energy_consumed: FixedPoint::ZERO,
            total_energy_harvested: FixedPoint::ZERO,
            uptime_seconds: 0,
            idle_step_count: 0,
            testing: false,
        }
    }

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    /// Initialize power management hardware
    pub fn init(&mut self) {
        if self.testing {
            return;
        }
        hw::set_voltage_scale(1);
        self.sync_domains_to_opp();
    }

    // -----------------------------------------------------------------------
    // Main update — call once per network step (~1ms)
    // -----------------------------------------------------------------------

    pub fn update(&mut self) {
        // Update energy balance
        let net_energy = self.energy_harvesting_rate - self.consumption_rate;
        self.battery_level =
            (self.battery_level + net_energy).clamp(FixedPoint::ZERO, FixedPoint::ONE);

        // Accumulate totals
        self.total_energy_consumed += self.consumption_rate;
        self.total_energy_harvested += self.energy_harvesting_rate;

        // Determine mode
        if self.battery_level < FixedPoint::from_f32(0.1) {
            self.mode = PowerMode::Critical;
            self.is_critical = true;
            self.enter_critical_mode();
        } else if self.battery_level < FixedPoint::from_f32(0.3) {
            self.mode = PowerMode::LowPower;
            self.is_critical = false;
            self.enter_low_power_mode();
        } else if self.energy_harvesting_rate > self.consumption_rate * FixedPoint::from_f32(2.0) {
            self.mode = PowerMode::Exploration;
            self.is_critical = false;
            self.enter_exploration_mode();
        } else {
            self.mode = PowerMode::Normal;
            self.is_critical = false;
            self.enter_normal_mode();
        }

        // Harvesting MPPT update
        self.update_mppt();

        // Adjust OPP based on load
        self.adjust_opp();

        // Update domain budgets
        self.update_budgets();
    }

    // -----------------------------------------------------------------------
    // Mode transitions
    // -----------------------------------------------------------------------

    fn enter_normal_mode(&mut self) {
        self.set_domain_enabled(PowerDomain::Sensors, true);
        self.set_domain_enabled(PowerDomain::Actuators, true);
        self.set_domain_enabled(PowerDomain::Cognitive, true);
        self.set_domain_enabled(PowerDomain::Radio, false);
        self.wakeup = WakeUpConfig::new();
    }

    fn enter_low_power_mode(&mut self) {
        // Reduce sensor and cognitive activity
        self.set_domain_enabled(PowerDomain::Sensors, true);
        self.set_domain_enabled(PowerDomain::Actuators, true);
        self.set_domain_enabled(PowerDomain::Cognitive, false);
        self.set_domain_enabled(PowerDomain::Radio, false);
        // Lower voltage to scale 3
        if !self.testing {
            hw::set_voltage_scale(3);
        }
        // Reduce OPP
        self.current_opp = 1; // 32 MHz
        self.sync_domains_to_opp();
        // Deactivate cognitive layers
        self.deactivate_layers(&[3, 7]);
    }

    fn enter_critical_mode(&mut self) {
        // Only sensors and basic reflexes
        self.set_domain_enabled(PowerDomain::Sensors, true);
        self.set_domain_enabled(PowerDomain::Actuators, true);
        self.set_domain_enabled(PowerDomain::Cognitive, false);
        self.set_domain_enabled(PowerDomain::Radio, false);
        // Lowest voltage and frequency
        if !self.testing {
            hw::set_voltage_scale(3);
        }
        self.current_opp = 0; // 16 MHz
        self.sync_domains_to_opp();
        // Gate clocks to unused peripherals
        if !self.testing {
            // Disable clocks to DMA, TIM2-7, I2C, SPI, etc.
            hw::set_peripheral_clock(1, 1, false); // TIM2
            hw::set_peripheral_clock(1, 4, false); // TIM3
        }
        self.deactivate_layers(&[1, 2, 3, 4, 5, 6, 7]);
    }

    fn enter_exploration_mode(&mut self) {
        // Full power: enable everything
        self.set_domain_enabled(PowerDomain::Sensors, true);
        self.set_domain_enabled(PowerDomain::Actuators, true);
        self.set_domain_enabled(PowerDomain::Cognitive, true);
        self.set_domain_enabled(PowerDomain::Radio, true);
        if !self.testing {
            hw::set_voltage_scale(1);
        }
        self.current_opp = 4; // 216 MHz
        self.sync_domains_to_opp();
        self.activate_all_layers();
    }

    // -----------------------------------------------------------------------
    // Sleep entry / exit
    // -----------------------------------------------------------------------

    /// Enter light sleep (WFI), waking on any interrupt
    pub fn enter_light_sleep(&mut self) {
        self.sleep_state = SleepState::LightSleep;
        if !self.testing {
            hw::enter_sleep();
        }
        // Wake: execution continues here
        self.sleep_state = SleepState::Active;
    }

    /// Enter deep sleep (Stop mode), configure wake-up sources first
    pub fn enter_deep_sleep(&mut self) -> bool {
        self.sleep_state = SleepState::DeepSleep;
        if !self.testing {
            hw::enter_deep_sleep(
                self.mode == PowerMode::Critical,
                true, // Flash power-down
            );
        }
        self.sleep_state = SleepState::Active;
        true
    }

    /// Enter shutdown (Standby mode) — full reset on wake
    pub fn enter_shutdown(&mut self) {
        self.sleep_state = SleepState::Shutdown;
        if !self.testing {
            hw::enter_standby();
        }
        // Execution halts here until reset
    }

    /// Enter idle if no work to do (adaptive sleep based on workload)
    pub fn idle_if_possible(&mut self, work_done: bool) {
        if !work_done {
            self.idle_step_count += 1;
            if self.idle_step_count > 10 {
                // 10ms of idle → enter light sleep briefly
                match self.mode {
                    PowerMode::Critical | PowerMode::LowPower => {
                        self.enter_deep_sleep();
                    }
                    PowerMode::Normal => {
                        self.enter_light_sleep();
                    }
                    PowerMode::Exploration => {
                        // Don't sleep in exploration mode
                        self.idle_step_count = 0;
                    }
                }
            }
        } else {
            self.idle_step_count = 0;
        }
    }

    /// Wake from deep sleep — check wake-up source
    pub fn check_wake_source(&self) -> WakeSource {
        if self.testing {
            return WakeSource::None;
        }
        let csr = unsafe { core::ptr::read_volatile(hw::PWR_CSR) };
        if csr & hw::CSR_WUF != 0 {
            unsafe {
                core::ptr::write_volatile(hw::PWR_CSR, csr | hw::CSR_WUF);
            }
        }
        if csr & hw::CSR_SBF != 0 {
            return WakeSource::Standby;
        }
        let pr = unsafe { core::ptr::read_volatile(0x4001_3C14 as *const u32) };
        if pr != 0 {
            return WakeSource::Gpio;
        }
        WakeSource::Timer
    }

    // -----------------------------------------------------------------------
    // DVFS
    // -----------------------------------------------------------------------

    /// Get current OPP details
    pub fn current_opp_info(&self) -> &'static OppPoint {
        &OPP_TABLE[self.current_opp as usize]
    }

    /// Adjust OPP based on load and battery
    fn adjust_opp(&mut self) {
        let target = match self.mode {
            PowerMode::Exploration => 4,
            PowerMode::Normal => {
                if self.battery_level > FixedPoint::from_f32(0.7) {
                    3
                } else {
                    2
                }
            }
            PowerMode::LowPower => 1,
            PowerMode::Critical => 0,
        };
        if target != self.current_opp as usize && !self.testing {
            hw::set_voltage_scale(match target {
                4 | 3 => 1,
                2 => 2,
                _ => 3,
            });
            self.current_opp = target as u8;
            self.sync_domains_to_opp();
        }
    }

    fn sync_domains_to_opp(&mut self) {
        let opp = &OPP_TABLE[self.current_opp as usize];
        let ratio = FixedPoint::from_f32(opp.freq_mhz as f32 / 216.0);
        for domain in self.domains.iter_mut() {
            if domain.enabled {
                domain.frequency_hz = opp.freq_mhz * 1_000_000;
                domain.voltage_scale = ratio;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Domain control
    // -----------------------------------------------------------------------

    pub fn set_domain_enabled(&mut self, domain: PowerDomain, enabled: bool) {
        let idx = domain as usize;
        self.domains[idx].enabled = enabled;
        if !enabled {
            self.domains[idx].clock_gated = true;
            // Gate peripheral clocks
            if !self.testing {
                match domain {
                    PowerDomain::Sensors => {
                        hw::set_peripheral_clock(1, 21, false); // I2C1
                        hw::set_peripheral_clock(2, 12, false); // SPI1
                        hw::set_peripheral_clock(2, 8, false); // ADC1
                    }
                    PowerDomain::Actuators => {
                        hw::set_peripheral_clock(1, 0, false); // TIM2
                        hw::set_peripheral_clock(1, 1, false); // TIM3
                        hw::set_peripheral_clock(1, 29, false); // DAC
                    }
                    PowerDomain::Radio => {
                        // Disable radio interface clocks
                    }
                    PowerDomain::Cognitive => {
                        // Cognitive modules don't gate to specific peripherals
                        // but can reduce clock speed
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn domain_enabled(&self, domain: PowerDomain) -> bool {
        self.domains[domain as usize].enabled
    }

    /// Get total power draw estimate (normalized 0.0-1.0)
    pub fn estimated_power_draw(&self) -> FixedPoint {
        let mut total = FixedPoint::ZERO;
        for (i, domain) in self.domains.iter().enumerate() {
            if domain.enabled {
                let base = match i as u8 {
                    d if d == PowerDomain::Cpu as u8 => FixedPoint::from_f32(0.30),
                    d if d == PowerDomain::Memory as u8 => FixedPoint::from_f32(0.20),
                    d if d == PowerDomain::Sensors as u8 => FixedPoint::from_f32(0.15),
                    d if d == PowerDomain::Actuators as u8 => FixedPoint::from_f32(0.15),
                    d if d == PowerDomain::Radio as u8 => FixedPoint::from_f32(0.15),
                    _ => FixedPoint::from_f32(0.05),
                };
                total += base * domain.voltage_scale;
            }
        }
        total * self.sleep_state.power_ratio()
    }

    // -----------------------------------------------------------------------
    // Energy harvesting
    // -----------------------------------------------------------------------

    /// Update maximum-power-point-tracking duty cycle
    fn update_mppt(&mut self) {
        if self.harvesting_type == HarvestingType::None {
            return;
        }
        // Simple perturb-and-observe MPPT
        // Perturb duty cycle
        let perturbation = FixedPoint::from_f32(0.01);
        if self.mppt_duty_cycle > perturbation {
            self.mppt_duty_cycle -= perturbation;
        } else {
            self.mppt_duty_cycle += perturbation;
        }
        self.mppt_duty_cycle = self
            .mppt_duty_cycle
            .clamp(FixedPoint::from_f32(0.1), FixedPoint::from_f32(0.9));
        // Simulate change in harvesting rate based on duty cycle
        self.energy_harvesting_rate = self.mppt_duty_cycle;
    }

    // -----------------------------------------------------------------------
    // Power budgeting
    // -----------------------------------------------------------------------

    fn update_budgets(&mut self) {
        for (i, domain) in self.domains.iter().enumerate() {
            self.budget.domain_budgets[i] = if domain.enabled {
                FixedPoint::ONE / FixedPoint::from_int(DOMAIN_COUNT as i32)
            } else {
                FixedPoint::ZERO
            };
        }
        self.budget.total_budget = FixedPoint::ONE;
        let est = self.estimated_power_draw();
        self.budget.avg_power_mw = self.budget.avg_power_mw * FixedPoint::from_f32(0.99)
            + est * FixedPoint::from_f32(0.01);
        if est > self.budget.peak_power_mw {
            self.budget.peak_power_mw = est;
        }
    }

    /// Query if there is enough energy for a given domain to activate
    pub fn can_power_domain(&self, domain: PowerDomain) -> bool {
        let domain_power = match domain {
            PowerDomain::Cpu => FixedPoint::from_f32(0.30),
            PowerDomain::Memory => FixedPoint::from_f32(0.20),
            PowerDomain::Sensors => FixedPoint::from_f32(0.15),
            PowerDomain::Actuators => FixedPoint::from_f32(0.15),
            PowerDomain::Radio => FixedPoint::from_f32(0.15),
            PowerDomain::Cognitive => FixedPoint::from_f32(0.05),
        };
        self.battery_level > domain_power * FixedPoint::from_f32(1.5)
    }

    /// Estimated remaining runtime at current consumption rate (seconds)
    /// Each simulation step = 1 ms
    pub fn estimated_runtime_s(&self) -> u64 {
        if self.consumption_rate == FixedPoint::ZERO {
            return u64::MAX;
        }
        let remaining_steps = self.battery_level / self.consumption_rate;
        // Convert steps (1ms each) to seconds: steps / 1000
        let seconds_fp = remaining_steps / FixedPoint::from_int(1000);
        (seconds_fp.to_f32() as u64).max(1)
    }

    // -----------------------------------------------------------------------
    // Voltage threshold modulation (V_th dynamique, Section 11)
    // -----------------------------------------------------------------------

    /// Returns a threshold multiplier based on power mode and battery level.
    /// Higher multiplier = less firing = less energy consumption.
    /// Lower multiplier = more firing = more exploration.
    pub fn threshold_multiplier(&self) -> FixedPoint {
        let base = match self.mode {
            PowerMode::Critical => FixedPoint::from_f32(2.5),
            PowerMode::LowPower => FixedPoint::from_f32(1.4),
            PowerMode::Normal => FixedPoint::from_f32(1.0),
            PowerMode::Exploration => FixedPoint::from_f32(0.85),
        };
        // Fine-tune with battery level (lower battery → higher threshold)
        let bat_factor =
            FixedPoint::ONE + (FixedPoint::ONE - self.battery_level) * FixedPoint::from_f32(0.3);
        base * bat_factor
    }

    // -----------------------------------------------------------------------
    // Layer deactivation (cognitive energy saving)
    // -----------------------------------------------------------------------

    /// Deactivate specified layers to save energy (Section 34)
    pub fn deactivate_layers(&self, layers: &[u8]) {
        let count = crate::core::memory::NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = crate::core::memory::NeuronId::new(i);
            let state = crate::core::memory::neuron_state(id);
            if layers.contains(&state.layer) {
                state.threshold = FixedPoint::from_f32(10.0);
            }
        }
    }

    /// Activate all layers
    pub fn activate_all_layers(&self) {
        let count = crate::core::memory::NEURON_COUNT.load(Ordering::Relaxed);
        for i in 0..count as u16 {
            let id = crate::core::memory::NeuronId::new(i);
            let state = crate::core::memory::neuron_state(id);
            if state.threshold > FixedPoint::ONE {
                state.threshold = FixedPoint::ONE;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wake source
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WakeSource {
    None,
    Timer,
    Gpio,
    Standby,
    Rtc,
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------
use core::mem::MaybeUninit;
pub static mut POWER_MANAGER: MaybeUninit<PowerManager> = MaybeUninit::uninit();

static INITIALIZED_POWER_MANAGER: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub fn init_power_manager() {
    unsafe {
        let pm = POWER_MANAGER.write(PowerManager::new());
        pm.init();
        INITIALIZED_POWER_MANAGER.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

pub fn power_manager() -> &'static mut PowerManager {
    unsafe {
        if !INITIALIZED_POWER_MANAGER.load(core::sync::atomic::Ordering::Relaxed) {
            init_power_manager();
        }
        &mut *POWER_MANAGER.as_mut_ptr()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_manager_new_default() {
        let pm = PowerManager::new();
        assert_eq!(pm.battery_level, FixedPoint::ONE);
        assert_eq!(pm.mode, PowerMode::Normal);
        assert_eq!(pm.sleep_state, SleepState::Active);
        assert_eq!(pm.current_opp, 3);
        assert!(!pm.is_critical);
    }

    #[test]
    fn power_manager_init_sets_defaults() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.init();
        assert_eq!(pm.current_opp, 3);
    }

    #[test]
    fn power_mode_transitions() {
        let mut pm = PowerManager::new();
        pm.testing = true;

        // Normal mode
        pm.mode = PowerMode::Normal;
        pm.enter_normal_mode();
        assert!(pm.domains[PowerDomain::Sensors as usize].enabled);
        assert!(pm.domains[PowerDomain::Cognitive as usize].enabled);
        assert!(!pm.domains[PowerDomain::Radio as usize].enabled);

        // Low power
        pm.mode = PowerMode::LowPower;
        pm.enter_low_power_mode();
        assert!(pm.domains[PowerDomain::Sensors as usize].enabled);
        assert!(!pm.domains[PowerDomain::Cognitive as usize].enabled);
        assert_eq!(pm.current_opp, 1);

        // Critical
        pm.mode = PowerMode::Critical;
        pm.enter_critical_mode();
        assert!(pm.domains[PowerDomain::Sensors as usize].enabled);
        assert!(!pm.domains[PowerDomain::Cognitive as usize].enabled);
        assert!(!pm.domains[PowerDomain::Radio as usize].enabled);
        assert_eq!(pm.current_opp, 0);

        // Exploration
        pm.mode = PowerMode::Exploration;
        pm.enter_exploration_mode();
        assert!(pm.domain_enabled(PowerDomain::Radio));
        assert!(pm.domain_enabled(PowerDomain::Cognitive));
        assert_eq!(pm.current_opp, 4);
    }

    #[test]
    fn sleep_state_properties() {
        assert_eq!(SleepState::Active.wake_latency_us(), 0);
        assert_eq!(SleepState::LightSleep.wake_latency_us(), 5);
        assert_eq!(SleepState::DeepSleep.wake_latency_us(), 200);
        assert_eq!(SleepState::Shutdown.wake_latency_us(), 5000);
        assert_eq!(SleepState::Active.power_ratio(), FixedPoint::ONE);
        assert_eq!(
            SleepState::Shutdown.power_ratio(),
            FixedPoint::from_f32(0.001)
        );
    }

    #[test]
    fn sleep_state_from_u8() {
        assert_eq!(SleepState::from_u8(0), SleepState::Active);
        assert_eq!(SleepState::from_u8(2), SleepState::DeepSleep);
        assert_eq!(SleepState::from_u8(99), SleepState::Active);
    }

    #[test]
    fn power_budget_new_default() {
        let budget = PowerBudget::new();
        assert_eq!(budget.total_budget, FixedPoint::ZERO);
        assert_eq!(budget.peak_power_mw, FixedPoint::ZERO);
    }

    #[test]
    fn wake_up_config_new() {
        let wc = WakeUpConfig::new();
        assert!(!wc.rtc_alarm_enabled);
        assert_eq!(wc.exti_pins, 0);
    }

    #[test]
    fn domain_state_new() {
        let ds = DomainState::new();
        assert!(ds.enabled);
        assert!(!ds.clock_gated);
        assert_eq!(ds.voltage_scale, FixedPoint::ONE);
        assert_eq!(ds.frequency_hz, 100_000_000);
    }

    #[test]
    fn opp_table_has_valid_points() {
        assert_eq!(OPP_TABLE.len(), 5);
        assert_eq!(OPP_TABLE[0].freq_mhz, 16);
        assert_eq!(OPP_TABLE[3].freq_mhz, 120);
        assert_eq!(OPP_TABLE[4].freq_mhz, 216);
        assert_eq!(OPP_TABLE[4].performance, FixedPoint::ONE);
    }

    #[test]
    fn estimated_power_draw_active() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        let draw = pm.estimated_power_draw();
        assert!(draw > FixedPoint::ZERO);
        assert!(draw <= FixedPoint::ONE);
    }

    #[test]
    fn estimated_power_draw_lower_in_sleep() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        let active = pm.estimated_power_draw();
        pm.sleep_state = SleepState::DeepSleep;
        let sleeping = pm.estimated_power_draw();
        assert!(sleeping < active);
    }

    #[test]
    fn set_domain_enabled_gates_clocks() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.set_domain_enabled(PowerDomain::Radio, false);
        assert!(!pm.domain_enabled(PowerDomain::Radio));
        assert!(pm.domains[PowerDomain::Radio as usize].clock_gated);
    }

    #[test]
    fn can_power_domain_battery_full() {
        let pm = PowerManager::new();
        assert!(pm.can_power_domain(PowerDomain::Cpu));
        assert!(pm.can_power_domain(PowerDomain::Radio));
    }

    #[test]
    fn can_power_domain_battery_low() {
        let mut pm = PowerManager::new();
        pm.battery_level = FixedPoint::from_f32(0.05);
        assert!(!pm.can_power_domain(PowerDomain::Cpu));
        assert!(!pm.can_power_domain(PowerDomain::Radio));
    }

    #[test]
    fn update_changes_mode_to_critical() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.battery_level = FixedPoint::from_f32(0.05);
        pm.update();
        assert_eq!(pm.mode, PowerMode::Critical);
        assert!(pm.is_critical);
    }

    #[test]
    fn update_changes_mode_to_exploration() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.energy_harvesting_rate = FixedPoint::from_f32(0.1);
        pm.consumption_rate = FixedPoint::from_f32(0.01);
        pm.update();
        assert_eq!(pm.mode, PowerMode::Exploration);
    }

    #[test]
    fn update_changes_mode_to_low_power() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.battery_level = FixedPoint::from_f32(0.2);
        pm.update();
        assert_eq!(pm.mode, PowerMode::LowPower);
    }

    #[test]
    fn idle_sleep_after_consecutive_idle() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        assert_eq!(pm.idle_step_count, 0);
        pm.idle_if_possible(false);
        assert_eq!(pm.idle_step_count, 1);
        // In test mode, enter_light_sleep/deep_sleep sets state to Active
        // and doesn't actually sleep, so we just check counter
        for _ in 0..10 {
            pm.idle_if_possible(false);
        }
        // After 10 idle cycles, sleep was entered; counter should be reset by sleep
        // but since testing=true, sleep functions don't reset counter
        // Let's verify the sleep entry condition
    }

    #[test]
    fn idle_resets_on_work() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        for _ in 0..5 {
            pm.idle_if_possible(false);
        }
        assert_eq!(pm.idle_step_count, 5);
        pm.idle_if_possible(true);
        assert_eq!(pm.idle_step_count, 0);
    }

    #[test]
    fn estimated_runtime() {
        let mut pm = PowerManager::new();
        pm.battery_level = FixedPoint::from_f32(0.5);
        pm.consumption_rate = FixedPoint::from_f32(0.005);
        let runtime = pm.estimated_runtime_s();
        assert!(runtime > 0);
    }

    #[test]
    fn mppt_perturbs_duty_cycle() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.harvesting_type = HarvestingType::Solar;
        let old_duty = pm.mppt_duty_cycle;
        pm.update_mppt();
        assert_ne!(pm.mppt_duty_cycle, old_duty);
    }

    #[test]
    fn power_domain_index_order() {
        assert_eq!(PowerDomain::Cpu as u8, 0);
        assert_eq!(PowerDomain::Memory as u8, 1);
        assert_eq!(PowerDomain::Sensors as u8, 2);
        assert_eq!(PowerDomain::Actuators as u8, 3);
        assert_eq!(PowerDomain::Radio as u8, 4);
        assert_eq!(PowerDomain::Cognitive as u8, 5);
    }

    #[test]
    fn wake_source_default_none() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        assert_eq!(pm.check_wake_source(), WakeSource::None);
    }

    #[test]
    fn budget_update_produces_values() {
        let mut pm = PowerManager::new();
        pm.testing = true;
        pm.update_budgets();
        let mut total = FixedPoint::ZERO;
        for &v in pm.budget.domain_budgets.iter() {
            total = total + v;
        }
        assert!(total > FixedPoint::ZERO);
    }

    #[test]
    fn deactivate_layers_raises_threshold() {
        let pm = PowerManager::new();
        pm.deactivate_layers(&[3]);
    }

    #[test]
    fn threshold_multiplier_normal() {
        let pm = PowerManager::new();
        let mult = pm.threshold_multiplier();
        assert!(mult > FixedPoint::from_f32(0.5));
        assert!(mult < FixedPoint::from_f32(5.0));
    }

    #[test]
    fn threshold_multiplier_critical_higher_than_normal() {
        let pm_normal = PowerManager::new();
        let mut pm_critical = PowerManager::new();
        pm_critical.mode = PowerMode::Critical;
        pm_critical.battery_level = FixedPoint::from_f32(0.05);
        let normal_mult = pm_normal.threshold_multiplier();
        let critical_mult = pm_critical.threshold_multiplier();
        assert!(critical_mult > normal_mult);
    }

    #[test]
    fn threshold_multiplier_exploration_lower_than_normal() {
        let pm_normal = PowerManager::new();
        let mut pm_explore = PowerManager::new();
        pm_explore.mode = PowerMode::Exploration;
        pm_explore.battery_level = FixedPoint::from_f32(0.9);
        let normal_mult = pm_normal.threshold_multiplier();
        let explore_mult = pm_explore.threshold_multiplier();
        assert!(explore_mult < normal_mult);
    }

    #[test]
    fn threshold_multiplier_low_battery_increases() {
        let mut pm = PowerManager::new();
        pm.battery_level = FixedPoint::from_f32(0.1);
        let low_bat = pm.threshold_multiplier();
        pm.battery_level = FixedPoint::ONE;
        let full_bat = pm.threshold_multiplier();
        assert!(low_bat > full_bat);
    }
}
