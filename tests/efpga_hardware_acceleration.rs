#![cfg(feature = "std")]

use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;
use hkl1::efpga::bitstream::BitstreamEncoder;
use hkl1::efpga::hdl_gen::{HdlGenerator, MAX_VERILOG_BUFFER_LEN};
use hkl1::efpga::simulator::EfpgaHardwareSimulator;
use hkl1::efpga::stability::SubnetworkStabilityAnalyzer;
use hkl1::efpga::efpga_engine;

#[test]
fn test_subnetwork_stability_analyzer() {
    let analyzer = SubnetworkStabilityAnalyzer::new();

    // 4 test synapses: (src, tgt, weight, delay_us, variance, age_cycles)
    let synapse_data = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200), // Stable!
        (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.8), 10, FixedPoint::from_f32(0.002), 150), // Stable!
        (NeuronId::new(2), NeuronId::new(3), FixedPoint::from_f32(0.2), 10, FixedPoint::from_f32(0.050), 300), // Unstable var (0.05 > 0.005)
        (NeuronId::new(3), NeuronId::new(4), FixedPoint::from_f32(0.9), 10, FixedPoint::from_f32(0.001), 10),  // Too young age (10 < 100)
    ];

    let subnetwork = analyzer.analyze_and_freeze_subnetwork(&synapse_data, 1);

    assert_eq!(subnetwork.count, 2, "Only 2 stable immutable synapses must be frozen!");
    assert_eq!(subnetwork.id, 1);
}

#[test]
fn test_synthesizable_verilog_hdl_generation() {
    let analyzer = SubnetworkStabilityAnalyzer::new();
    let synapse_data = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200),
        (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.8), 10, FixedPoint::from_f32(0.002), 150),
    ];

    let subnetwork = analyzer.analyze_and_freeze_subnetwork(&synapse_data, 42);
    let mut buffer = [0u8; MAX_VERILOG_BUFFER_LEN];

    let len = HdlGenerator::generate_verilog_hdl(&subnetwork, &mut buffer);
    let verilog_code = std::str::from_utf8(&buffer[..len]).unwrap();

    assert!(verilog_code.contains("module efpga_snn_subnetwork"));
    assert!(verilog_code.contains("V_memb"));
    assert!(verilog_code.contains("endmodule"));
}

#[test]
fn test_embedded_efpga_bitstream_encoding() {
    let analyzer = SubnetworkStabilityAnalyzer::new();
    let synapse_data = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200),
    ];

    let subnetwork = analyzer.analyze_and_freeze_subnetwork(&synapse_data, 7);
    let bitstream = BitstreamEncoder::encode_bitstream(&subnetwork);

    assert_eq!(bitstream.data[0], 0xEB, "Bitstream must start with sync byte 0xEB");
    assert_eq!(bitstream.subnetwork_id, 7);
    assert!(bitstream.valid_bytes > 4);
    assert!(bitstream.checksum > 0);
}

#[test]
fn test_hardware_sub_nanosecond_simulator() {
    let analyzer = SubnetworkStabilityAnalyzer::new();
    let synapse_data = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200),
        (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.8), 10, FixedPoint::from_f32(0.002), 150),
    ];

    let subnetwork = analyzer.analyze_and_freeze_subnetwork(&synapse_data, 100);
    let bitstream = BitstreamEncoder::encode_bitstream(&subnetwork);

    let (_out_spikes, benchmark) = EfpgaHardwareSimulator::simulate_hardware_execution(&bitstream);

    assert!(
        benchmark.latency_picoseconds < 1000,
        "Hardware latency must be sub-nanosecond (<1000 ps)!"
    );
    assert!(
        benchmark.speedup_vs_software > 100.0,
        "Hardware acceleration must provide >100x speedup vs software!"
    );
}

#[test]
fn test_full_efpga_engine_pipeline() {
    let engine = efpga_engine();
    let synapse_data = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.5), 10, FixedPoint::from_f32(0.001), 200),
        (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.8), 10, FixedPoint::from_f32(0.002), 150),
    ];

    let (success, benchmark) = engine.compile_and_accelerate_subnetwork(&synapse_data, 99);

    assert!(success);
    assert_eq!(benchmark.subnetwork_id, 99);
    assert!(engine.verilog_len > 0);
    assert!(engine.last_bitstream.is_some());
}
