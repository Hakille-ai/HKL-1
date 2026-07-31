//! HKL-1 Performance & Throughput Benchmark Suite
//! Measures bare-metal SNN stepping performance, SIMD fixed-point vector math throughput,
//! multi-thread scaling, and cognitive subsystem latencies.

use hkl1::core::math::{FixedPoint, Vector};
use hkl1::snn::network::network;
use std::time::Instant;

fn main() {
    println!("==================================================");
    println!("     HKL-1 Neuromorphic Engine Benchmark Suite    ");
    println!("==================================================");

    // 1. FixedPoint & SIMD Vector Math Benchmark
    println!("\n[Bench 1] SIMD FixedPoint Vector Operations (1,000,000 ops)...");
    let vec_a = Vector::<64>::splat(FixedPoint::from_f32(1.5));
    let vec_b = Vector::<64>::splat(FixedPoint::from_f32(2.25));

    let start = Instant::now();
    let mut dummy_sum = FixedPoint::ZERO;
    for _ in 0..1_000_000 {
        dummy_sum += vec_a.dot(&vec_b);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = 1_000_000.0 / elapsed.as_secs_f64();
    println!(
        "  Dot Product (64-dim): {:?} ({:.2} M ops/sec | checksum: {:.2})",
        elapsed,
        ops_per_sec / 1_000_000.0,
        dummy_sum.to_f32()
    );

    // 2. Trigonometric Math Benchmark (Bhaskara I Fixed-Point)
    println!("\n[Bench 2] Pure Integer Fixed-Point Trigonometry (1,000,000 sin/cos)...");
    let start = Instant::now();
    let mut trig_sum = FixedPoint::ZERO;
    for i in 0..1_000_000 {
        let angle = FixedPoint::from_f32(i as f32 * 0.001);
        trig_sum += angle.sin() + angle.cos();
    }
    let elapsed = start.elapsed();
    let trig_per_sec = 1_000_000.0 / elapsed.as_secs_f64();
    println!(
        "  Sin/Cos 1M ops: {:?} ({:.2} M ops/sec | checksum: {:.2})",
        elapsed,
        trig_per_sec / 1_000_000.0,
        trig_sum.to_f32()
    );

    // 3. Single-Thread SNN Stepping Benchmark
    println!("\n[Bench 3] Single-Thread SNN Network Stepping (1,000 steps)...");
    hkl1::system::power::init_power_manager();
    hkl1::telemetry::spike_trace::init_logger();
    hkl1::telemetry::xai::init_xai();
    hkl1::cognitive::temporal::init_temporal_cognition();
    hkl1::cognitive::predictor::init_cognitive_predictor();

    let net = network();
    let hw = net.auto_adapt_hardware();

    let start = Instant::now();
    for _ in 0..1_000 {
        net.step();
    }
    let elapsed = start.elapsed();
    let steps_per_sec = 1_000.0 / elapsed.as_secs_f64();
    let neuron_evals_per_sec = steps_per_sec * hw.recommended_max_neurons as f64;
    println!(
        "  1,000 SNN Steps: {:?} ({:.2} steps/sec | {:.2} M neuron-evals/sec)",
        elapsed,
        steps_per_sec,
        neuron_evals_per_sec / 1_000_000.0
    );

    // 4. Parallel SNN Stepping Benchmark
    let threads = hw.cpu_cores.min(4);
    println!(
        "\n[Bench 4] Multi-Thread Parallel SNN Stepping (1,000 steps across {} cores)...",
        threads
    );
    net.time = 0;
    net.cycle_active = false;
    net.warp_active = false;

    let start = Instant::now();
    for _ in 0..1_000 {
        net.step_parallel(threads);
    }
    let elapsed = start.elapsed();
    let parallel_steps_per_sec = 1_000.0 / elapsed.as_secs_f64();
    println!(
        "  1,000 Parallel Steps: {:?} ({:.2} steps/sec | Speedup vs single: {:.2}x)",
        elapsed,
        parallel_steps_per_sec,
        parallel_steps_per_sec / steps_per_sec
    );

    // 5. Audio Cochlea Gammatone Pipeline Benchmark
    println!("\n[Bench 5] Gammatone Cochlear Audio Processing (1,000 PCM Frames)...");
    hkl1::audio::init_audio_engine();
    let audio_eng = hkl1::audio::audio_engine();
    let pcm = [1000i16; 512];

    let start = Instant::now();
    for t in 0..1_000 {
        let _ = audio_eng.process_audio_stream(&pcm, t as u32);
    }
    let elapsed = start.elapsed();
    let audio_fps = 1_000.0 / elapsed.as_secs_f64();
    println!(
        "  1,000 Audio Frames: {:?} ({:.2} FPS | Real-Time Factor: {:.1}x)",
        elapsed,
        audio_fps,
        audio_fps / 31.25 // 32 ms frame rate equivalent
    );

    println!("\n==================================================");
    println!("          Benchmark Execution Completed           ");
    println!("==================================================");
}
