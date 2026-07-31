#![cfg(feature = "std")]

use core::mem::MaybeUninit;
use core::sync::atomic::Ordering;
use hkl1::core::math::FixedPoint;
use hkl1::core::memory::{NEURON_ARRAY, NEURON_COUNT, NeuronFlags, NeuronState, NeuronType};
use hkl1::snn::network::network;
use hkl1::system::hardware_detect::HardwareDetector;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn setup_network() -> std::sync::MutexGuard<'static, ()> {
    let guard = TEST_LOCK.lock().unwrap();
    hkl1::system::power::init_power_manager();
    hkl1::telemetry::spike_trace::init_logger();
    hkl1::telemetry::xai::init_xai();
    hkl1::cognitive::temporal::init_temporal_cognition();
    hkl1::snn::plasticity::reset_traces();
    hkl1::core::time::init_clock(core::ptr::null_mut(), 100_000_000, 1_000_000);

    let net = network();
    net.time = 0;

    unsafe {
        for i in 0..hkl1::MAX_NEURONS {
            NEURON_ARRAY[i] = MaybeUninit::new(NeuronState {
                membrane_potential: FixedPoint::ZERO,
                threshold: FixedPoint::from_f32(0.5),
                leak: FixedPoint::from_f32(0.01),
                refractory_remaining: 0,
                last_spike_time: 0,
                bias_current: FixedPoint::ZERO,
                layer: (i / 2) as u8,
                neuron_type: NeuronType::LIF,
                flags: NeuronFlags(0),
            });
        }
        NEURON_COUNT.store(256, Ordering::Relaxed);
        hkl1::snn::synapse::SYNAPSE_COUNT.store(0, Ordering::Relaxed);
        hkl1::core::memory::ADAPTIVE_MEMORY.set_capacity(256, 0);
    }
    guard
}

#[test]
fn test_parallel_step_single_thread() {
    let _guard = setup_network();
    let net = network();
    let initial_time = net.time;
    net.step_parallel(1);
    assert_eq!(net.time, initial_time + 1);
}

#[test]
fn test_parallel_step_multi_threads() {
    let _guard = setup_network();
    let net = network();
    let initial_time = net.time;

    let profile = HardwareDetector::detect();
    let threads = profile.recommended_worker_threads.max(2);

    for i in 0..10 {
        println!("Starting step {}", i + 1);
        net.step_parallel(threads);
        println!("Finished step {}", i + 1);
    }

    assert_eq!(net.time, initial_time + 10);
}

#[test]
fn test_parallel_step_scaled_capacity() {
    let _guard = setup_network();
    let net = network();
    net.scale_capacity(131_072, 2_097_152);

    let initial_time = net.time;
    for _ in 0..5 {
        net.step_parallel(4);
    }
    assert_eq!(net.time, initial_time + 5);

    // Reset back
    net.scale_capacity(hkl1::MAX_NEURONS, hkl1::MAX_SYNAPSES);
}
