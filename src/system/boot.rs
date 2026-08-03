//! Boot sequence and main loop. Initializes hardware, starts metabolic clock, runs forever.
use crate::cognitive::actor::init_cognitive_actor;
use crate::cognitive::predictor::init_cognitive_predictor;
use crate::cognitive::temporal::init_temporal_cognition;
use crate::core::math::{FixedPoint, XorShift64Star};
use crate::core::memory::{NEURON_ARRAY, NEURON_COUNT, NeuronFlags, NeuronState, NeuronType};
use crate::io::buffers::init_buffers;
use crate::snn::neuron::init_population;
use crate::snn::synapse::init_connectivity;
use crate::system::ota::{init_ota, ota_manager};
use crate::system::persistence::{PersistenceManager, init_persistence};
use crate::system::power::{init_power_manager, power_manager};
use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;

/// Boot sequence (Section 9):
/// t=0ms:   Power on -> Stack pointer init (hardware)
/// t=2ms:   Static matrix reservation (BSS zeroing - hardware)
/// t=5ms:   Memory Mapped I/O Read -> Flash read
/// t=15ms:  Load saved synaptic weights (J-0)
/// t=20ms:  Activate Ring Buffers -> Multimodal ingestion
/// t=22ms:  HKL-1 operational

pub struct BootSequence;

impl BootSequence {
    /// Entry point - called from reset vector
    pub fn init_hardware() {
        // t=0ms: Set up hardware peripherals
        Self::init_hardware_peripherals();

        // t=2ms: Memory system initialized (BSS zeroed by startup)
        Self::init_memory_system();

        // t=5ms: Read Flash for boot config
        Self::read_boot_config();

        // t=8ms: Initialize OTA system (bank detection, rollback handling)
        init_ota();
        if ota_manager().status == crate::system::ota::OtaStatus::RolledBack {
            // Notify that a rollback occurred
        }

        // t=9ms: Initialize power management
        init_power_manager();

        // t=10ms: Initialize RNG from hardware entropy
        let seed = Self::get_boot_seed();
        let mut rng = XorShift64Star::new(seed);

        // t=12ms: Create initial neuron population
        init_population(&mut rng);

        // t=14ms: Create initial synaptic connectivity
        init_connectivity(&mut rng, 0.1); // 10% sparsity

        // t=14.5ms: Initialize persistence buffers
        init_persistence();

        // t=15ms: Load persistence
        PersistenceManager::load_slot(0);

        // t=18ms: Initialize I/O buffers
        init_buffers();

        // t=18ms: Initialize I/O managers + ISR system
        crate::io::sensors::init_sensor_manager();
        crate::io::actuators::init_actuator_manager();
        crate::io::isr::init_isr_system(&crate::io::isr::IsrConfig::default());

        // t=19ms: Initialize cognitive modules
        init_cognitive_actor();
        init_cognitive_predictor();
        init_temporal_cognition();
        crate::cognitive::episodic::init_episodic_memory();
        crate::cognitive::attention::init_attention_router();

        // t=19.2ms: Initialize bio-inspired modules
        crate::bio::astrocytes::init_astrocytes();
        crate::bio::striosome::init_striosome_matrix();
        crate::bio::thalamus::init_thalamus();
        crate::bio::hippocampus::init_hippocampus();
        crate::bio::cerebellum::init_cerebellum();

        // t=19.5ms: Initialize eFPGA bio-compilation engine
        crate::efpga::init_efpga_engine();

        // t=19.6ms: Initialize entropy monitor + XAI / telemetry
        crate::core::entropy::init_entropy(seed);
        crate::telemetry::spike_trace::init_logger();
        crate::telemetry::xai::init_xai();

        // t=20ms: Enable sensor ISRs
        Self::enable_sensor_interrupts();

        // t=21ms: OTA lifecycle — apply staged update if available
        if ota_manager().check_for_update() {
            ota_manager().apply_update();
            // apply_update triggers soft reset; never returns
        }

        // t=21.5ms: Confirm OTA stability (boot_count check)
        ota_manager().confirm_stable();

        // t=22ms: Main loop starts
    }

    /// Initialize hardware peripherals (platform-specific)
    fn init_hardware_peripherals() {
        #[cfg(all(feature = "stm32f7", target_arch = "arm"))]
        {
            // 1. Enable FPU (CPACR register)
            const CPACR: *mut u32 = 0xE000_ED88u32 as *mut u32;
            unsafe {
                core::ptr::write_volatile(CPACR, core::ptr::read_volatile(CPACR) | 0x00F00000);
                core::arch::asm!("dsb", "isb");
            }

            // 2. Configure vector table offset (SCB->VTOR)
            const SCB_VTOR: *mut u32 = 0xE000_ED08u32 as *mut u32;
            unsafe {
                core::ptr::write_volatile(SCB_VTOR, 0x08000000);
            }

            // 3. Enable MPU with background region
            const MPU_CTRL: *mut u32 = 0xE000_ED94u32 as *mut u32;
            unsafe {
                core::ptr::write_volatile(MPU_CTRL, 5);
            }

            // 4. Enable I-cache, D-cache
            const SCS_CCR: *mut u32 = 0xE000_ED14u32 as *mut u32;
            unsafe {
                let ccr = core::ptr::read_volatile(SCS_CCR);
                core::ptr::write_volatile(SCS_CCR, ccr | 0x10000);
                core::arch::asm!("dsb", "isb");
            }
        }
    }

    /// Initialize memory system
    fn init_memory_system() {
        // Clear neuron array
        for i in 0..crate::MAX_NEURONS {
            unsafe {
                NEURON_ARRAY[i] = MaybeUninit::new(NeuronState {
                    membrane_potential: crate::core::math::FixedPoint::ZERO,
                    threshold: crate::core::math::FixedPoint::ZERO,
                    leak: crate::core::math::FixedPoint::ZERO,
                    refractory_remaining: 0,
                    last_spike_time: 0,
                    bias_current: crate::core::math::FixedPoint::ZERO,
                    layer: 0,
                    neuron_type: NeuronType::LIF,
                    flags: NeuronFlags(0),
                });
            }
        }
        NEURON_COUNT.store(0, Ordering::Relaxed);
    }

    /// Read boot configuration from Flash
    fn read_boot_config() {
        #[cfg(all(feature = "stm32f7", target_arch = "arm"))]
        {
            // Read UID from STM32F7 OTP area (0x1FFF_7A10)
            const UID_BASE: *const u32 = 0x1FFF_7A10 as *const u32;
            let uid0 = unsafe { core::ptr::read_volatile(UID_BASE) };
            let _uid1 = unsafe { core::ptr::read_volatile(UID_BASE.add(1)) };
            let _uid2 = unsafe { core::ptr::read_volatile(UID_BASE.add(2)) };
            // Hardware version from OTP (first byte of UID)
            let _hw_version = (uid0 & 0xFF) as u8;
            // PUF calibration would read from OTP here
        }
    }

    /// Get entropy seed for RNG
    fn get_boot_seed() -> u64 {
        unsafe {
            let clock = &crate::core::time::METABOLIC_CLOCK;
            let c1 = clock.cycles();
            let c2 = clock.now_us();
            c1 ^ c2
        }
    }

    /// Enable sensor interrupt handlers
    fn enable_sensor_interrupts() {
        #[cfg(all(feature = "stm32f7", target_arch = "arm"))]
        {
            const NVIC_ISER0: *mut u32 = 0xE000_E100 as *mut u32;
            const NVIC_ISER1: *mut u32 = 0xE000_E104 as *mut u32;
            const NVIC_ISER2: *mut u32 = 0xE000_E108 as *mut u32;
            unsafe {
                core::ptr::write_volatile(NVIC_ISER0, 0xFFFFFFFF); // IRQ 0-31
                core::ptr::write_volatile(NVIC_ISER1, 0xFFFFFFFF); // IRQ 32-63
                core::ptr::write_volatile(NVIC_ISER2, 0xFFFFFFFF); // IRQ 64-95
            }
        }
    }

    /// Main loop - runs forever
    pub fn run_main_loop() -> ! {
        // Start the metabolic clock
        crate::core::time::init_clock(
            core::ptr::null_mut(), // timer_base (platform-specific)
            100_000_000,           // 100 MHz CPU
            1_000_000,             // 1 MHz timer
        );

        // Enable interrupts
        #[cfg(target_arch = "arm")]
        unsafe {
            core::arch::asm!("cpsie i");
        }
        #[cfg(target_arch = "riscv32")]
        unsafe {
            core::arch::asm!("csrsi mstatus, 8");
        }

        // Main loop
        loop {
            // Handle pending ISRs (deferred work)
            crate::io::isr::handle_pending_isrs();

            // Step the network
            let net = crate::snn::network::network();
            net.step();

            // Update power management
            power_manager().update();

            // Check for emergency events
            check_emergencies(net);

            // Handle telemetry
            if net.time.is_multiple_of(100) {
                handle_telemetry();
            }

            // Sync energy level to network for threshold modulation
            net.energy_level = power_manager().battery_level;

            // Enter low-power idle based on network activity
            let work_done = net.time.is_multiple_of(10);
            power_manager().idle_if_possible(work_done);
        }
    }
}

/// Emergency handler
fn check_emergencies(net: &mut crate::snn::network::Network) {
    // Section 19: Spinal reflex check
    unsafe {
        crate::safety::reflexes::REFLEXES.check_all();
    }

    // Section 31: Memory degradation check
    let count = NEURON_COUNT.load(Ordering::Relaxed);
    if count == 0 || count > crate::MAX_NEURONS {
        net.energy_level = FixedPoint::ZERO; // Force energy-critical shutdown
    }

    // Section 7.2: Entropy health
    let entropy_state = unsafe { crate::safety::entropy_monitor::ENTROPY_MONITOR.check_health() };
    match entropy_state {
        crate::safety::entropy_monitor::EntropyState::HighEntropy => {
            net.threshold_modulation = FixedPoint::from_f32(2.0); // Raise thresholds
        }
        crate::safety::entropy_monitor::EntropyState::LowEntropy => unsafe {
            crate::cognitive::curiosity::CURIOSITY_ENGINE.activate_dreaming();
        },
        _ => {}
    }
}

/// Telemetry output handler
fn handle_telemetry() {
    // Analyze causal relationships from spike trace
    crate::telemetry::xai::analyze_current_trace();
}

// Note: init_hardware_peripherals is now the Self:: method above
