use hkl1::core::memory::ADAPTIVE_MEMORY;
use hkl1::system::hardware_detect::{HardwareDetector, HardwareProfile};

#[test]
fn test_hardware_detector_auto_profile() {
    let profile = HardwareDetector::detect();
    assert!(profile.recommended_max_neurons >= hkl1::MAX_NEURONS);
    assert!(profile.recommended_max_synapses >= hkl1::MAX_SYNAPSES);
    assert!(profile.recommended_worker_threads >= 1);
}

#[test]
fn test_scale_factor_calculation() {
    let profile = HardwareProfile::bare_metal_default();
    let scale = HardwareDetector::calculate_scale_factor(&profile);
    assert!((scale.to_f32() - 1.0).abs() < 0.001);
}

#[test]
fn test_network_auto_adapt_hardware() {
    let profile = HardwareDetector::detect();
    ADAPTIVE_MEMORY.set_capacity(
        profile.recommended_max_neurons,
        profile.recommended_max_synapses,
    );

    let (cap_neurons, cap_synapses) = ADAPTIVE_MEMORY.current_capacity();
    assert_eq!(cap_neurons, profile.recommended_max_neurons);
    assert_eq!(cap_synapses, profile.recommended_max_synapses);
}

#[test]
fn test_manual_dynamic_capacity_scaling() {
    ADAPTIVE_MEMORY.set_capacity(524_288, 8_388_608);

    let (cap_n, cap_s) = ADAPTIVE_MEMORY.current_capacity();
    assert_eq!(cap_n, 524_288);
    assert_eq!(cap_s, 8_388_608);

    // Reset back to baseline
    ADAPTIVE_MEMORY.set_capacity(hkl1::MAX_NEURONS, hkl1::MAX_SYNAPSES);
    let (reset_n, reset_s) = ADAPTIVE_MEMORY.current_capacity();
    assert_eq!(reset_n, hkl1::MAX_NEURONS);
    assert_eq!(reset_s, hkl1::MAX_SYNAPSES);
}
