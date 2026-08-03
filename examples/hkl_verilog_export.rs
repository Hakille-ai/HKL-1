//! Synthesizable Verilog HDL Exporter & eFPGA Hardware Acceleration Example (`hkl_verilog_export.rs`).
//! Evaluates SNN synapse stability, generates Verilog RTL HDL code, encodes eFPGA LUT4/LUT6
//! bitstreams, and benchmarks hardware execution speedups.

use hkl1::core::math::FixedPoint;
use hkl1::core::memory::NeuronId;
use hkl1::efpga::EfpgaEngine;

fn main() {
    println!("=== ⚡ HKL eFPGA Bio-Compilation & Verilog RTL Exporter ===");

    let mut efpga = EfpgaEngine::new();

    // 1. Prepare Candidate SNN Synapses
    // Format: (SourceNeuron, TargetNeuron, Weight, UpdatesCount, Variance, AgeCycles)
    println!("\n[1] Preparing Candidate SNN Synapse Network...");
    let synapse_candidates = [
        (NeuronId::new(0), NeuronId::new(1), FixedPoint::from_f32(0.85), 150, FixedPoint::from_f32(0.0005), 500),
        (NeuronId::new(1), NeuronId::new(2), FixedPoint::from_f32(0.62), 210, FixedPoint::from_f32(0.0002), 650),
        (NeuronId::new(2), NeuronId::new(3), FixedPoint::from_f32(0.91), 300, FixedPoint::from_f32(0.0001), 1200),
        (NeuronId::new(3), NeuronId::new(4), FixedPoint::from_f32(0.44), 180, FixedPoint::from_f32(0.0008), 450),
    ];
    println!("   Total Synapse Candidates: {}", synapse_candidates.len());

    // 2. Run Bio-Compilation Pipeline
    println!("\n[2] Executing eFPGA Bio-Compilation Pipeline...");
    let (success, benchmark) = efpga.compile_and_accelerate_subnetwork(&synapse_candidates, 101);

    if !success {
        println!("   Bio-compilation failed: Subnetwork did not meet stability criteria.");
        return;
    }

    println!("   Bio-compilation Successful!");
    println!("   Frozen Synapses Count : {}", efpga.last_frozen_subnetwork.as_ref().unwrap().count);

    // 3. Print Generated Verilog HDL Code
    println!("\n[3] Generated Synthesizable Verilog RTL Code ({} bytes):", efpga.verilog_len);
    println!("------------------------------------------------------------------");
    let verilog_code = String::from_utf8_lossy(&efpga.verilog_buffer[..efpga.verilog_len]);
    println!("{}", verilog_code);
    println!("------------------------------------------------------------------");

    // 4. Print Bitstream & Hardware Benchmark Results
    if let Some(bitstream) = &efpga.last_bitstream {
        println!("\n[4] eFPGA LUT4/LUT6 Bitstream Encoding:");
        println!("   Valid Bytes       : {} bytes", bitstream.valid_bytes);
        println!("   Bitstream Checksum: 0x{:08X}", bitstream.checksum);
    }

    println!("\n[5] Hardware Simulation & Sub-Nanosecond Acceleration Benchmark:");
    println!("   Subnetwork ID       : {}", benchmark.subnetwork_id);
    println!("   Latency             : {} picoseconds ({:.3} ns)", benchmark.latency_picoseconds, benchmark.latency_picoseconds as f64 / 1000.0);
    println!("   Clock Cycles        : {}", benchmark.clock_cycles);
    println!("   Hardware Speedup    : {:.2}x vs Software", benchmark.speedup_vs_software);
    println!("   Active LUTs Used    : {}", benchmark.active_luts);

    println!("\n=== ✅ Verilog RTL Export & eFPGA Acceleration Complete ===");
}
