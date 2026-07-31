//! Hardware Target Validation & Emulation Test Suite for HKL-1 Engine
//!
//! Validates:
//! 1. Hardware Detector auto-profiling and capacity scaling.
//! 2. Peripheral MMIO ring-buffer ISR spike injection.
//! 3. BSP board initialization sequence for Cortex-M7 (STM32F7) and RISC-V (HiFive1/ESP32-C6).

#![cfg(test)]

use hkl1::system::hardware_detect::HardwareDetector;
use hkl1::io::buffers::{RingBuffer, LockFreeRingBuffer};
use hkl1::core::memory::NeuronId;
use hkl1::snn::neuron::SpikeEvent;

#[test]
fn test_hardware_detector_emulated_profiling() {
    let profile = HardwareDetector::detect();

    // Verify auto-profile bounds
    assert!(profile.cpu_cores >= 1, "System must detect at least 1 core");
    assert!(profile.recommended_max_neurons >= 64, "Recommended neurons must be >= 64");
    assert!(profile.recommended_max_synapses >= 256, "Recommended synapses must be >= 256");
    assert!(profile.recommended_worker_threads >= 1, "Worker threads must be >= 1");
}

#[test]
fn test_ring_buffer_isr_spike_injection() {
    let mut ring: RingBuffer<u32, 64> = RingBuffer::new();

    for i in 0..32 {
        let ok = ring.push(100 + i);
        assert!(ok, "Ring buffer push must succeed under capacity");
    }

    for i in 0..32 {
        let val = ring.pop();
        assert_eq!(val, Some(100 + i), "Popped item must match pushed value");
    }
}

#[test]
fn test_lockfree_isr_spike_queue() {
    let ring: LockFreeRingBuffer<64> = LockFreeRingBuffer::new();

    if let Some(ptr) = ring.reserve_write() {
        unsafe {
            *ptr = SpikeEvent {
                neuron_id: NeuronId::new(42),
                timestamp: 12345,
                layer: 0,
                is_predictor: false,
            };
        }
        ring.commit_write();
    }

    let popped = ring.pop_front();
    assert!(popped.is_some(), "Popped spike must exist");
    let spike = popped.unwrap();
    assert_eq!(spike.neuron_id.index(), 42, "Neuron ID must match pushed ID 42");
    assert_eq!(spike.timestamp, 12345, "Timestamp must match pushed timestamp");
}

#[test]
fn test_bsp_initialization_routines() {
    // Test BSP module exports
    #[cfg(feature = "stm32f7")]
    {
        use core::sync::atomic::Ordering;
        let freq = hkl1::bsp::stm32f7::CPU_FREQ_HZ.load(Ordering::Relaxed);
        assert_eq!(freq, 216_000_000, "STM32F7 CPU frequency must be 216 MHz");
    }

    #[cfg(feature = "hifive1")]
    {
        use core::sync::atomic::Ordering;
        let freq = hkl1::bsp::hifive1::CPU_FREQ_HZ.load(Ordering::Relaxed);
        assert_eq!(freq, 320_000_000, "HiFive1 CPU frequency must be 320 MHz");
    }

    #[cfg(feature = "esp32c6")]
    {
        use core::sync::atomic::Ordering;
        let freq = hkl1::bsp::esp32c6::CPU_FREQ_HZ.load(Ordering::Relaxed);
        assert_eq!(freq, 160_000_000, "ESP32-C6 CPU frequency must be 160 MHz");
    }
}
