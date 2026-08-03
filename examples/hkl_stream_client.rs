//! High-Frequency Multimodal Streaming Client & Latency Benchmark (`hkl_stream_client.rs`).
//! Connects to HklNativeServer over TCP sockets and streams high-frequency
//! audio PCM, video frames, and e-prop training steps, measuring roundtrip microsecond latency.

#[cfg(feature = "hkl2")]
use hkl1::api::cortex_service::CortexService;
#[cfg(feature = "hkl2")]
use hkl1::api::protocol::{HKL_HEADER_SIZE, HklBinaryPacket, HklCommand};
#[cfg(feature = "hkl2")]
use hkl1::api::server::HklNativeServer;
#[cfg(feature = "hkl2")]
use std::io::{Read, Write};
#[cfg(feature = "hkl2")]
use std::net::TcpStream;
#[cfg(feature = "hkl2")]
use std::time::Instant;

#[cfg(feature = "hkl2")]
fn main() {
    println!("=== ⚡ HKL-2 High-Frequency Multimodal Stream Client Benchmark ===");

    let addr = "127.0.0.1:8989";
    let head_id = [1, 3, 5, 7, 9, 11, 13, 15];
    let service = CortexService::new(head_id, 1);
    let server = HklNativeServer::new(addr, service);

    // 1. Spawn TCP Server in background thread
    println!("\n[1] Starting HklNativeServer on tcp://{}...", addr);
    std::thread::spawn(move || {
        let _ = server.listen();
    });

    // Give server 100ms to bind
    std::thread::sleep(std::time::Duration::from_millis(100));

    // 2. Connect Client Stream
    println!("\n[2] Connecting TCP Client Stream to {}...", addr);
    let mut stream = match TcpStream::connect(addr) {
        Ok(s) => s,
        Err(e) => {
            println!("   Connection failed: {}. (Exiting benchmark)", e);
            return;
        }
    };
    println!("   Connected successfully!");

    // 3. Multimodal Streaming Loop
    let num_packets = 100usize;
    println!(
        "\n[3] Streaming {} Multimodal Packets over TCP...",
        num_packets
    );

    let start_total = Instant::now();
    let mut latencies_us = Vec::with_capacity(num_packets);

    for i in 0..num_packets {
        let t_start = Instant::now();

        // Alternate commands: Perceive -> CognitiveState -> Synthesize -> EpropTrainStep
        let cmd = match i % 4 {
            0 => HklCommand::PerceiveFrame,
            1 => HklCommand::CognitiveState,
            2 => HklCommand::SynthesizeResponse,
            _ => HklCommand::EpropTrainStep,
        };

        let payload = match cmd {
            HklCommand::PerceiveFrame => b"perceive_audio_video_stream".to_vec(),
            HklCommand::SynthesizeResponse => b"hello".to_vec(),
            HklCommand::EpropTrainStep => b"input->target".to_vec(),
            _ => vec![],
        };

        let req_packet = HklBinaryPacket::new(cmd, i as u64, payload);
        let encoded_req = req_packet.encode();

        if stream.write_all(&encoded_req).is_err() {
            println!("   Stream write error at packet {}", i);
            break;
        }

        // Read response header & payload
        let mut header_buf = [0u8; HKL_HEADER_SIZE];
        if stream.read_exact(&mut header_buf).is_err() {
            println!("   Stream read header error at packet {}", i);
            break;
        }

        let payload_len = u32::from_be_bytes([
            header_buf[12],
            header_buf[13],
            header_buf[14],
            header_buf[15],
        ]) as usize;
        let mut payload_buf = vec![0u8; payload_len];
        if stream.read_exact(&mut payload_buf).is_err() {
            println!("   Stream read payload error at packet {}", i);
            break;
        }

        let elapsed_us = t_start.elapsed().as_micros();
        latencies_us.push(elapsed_us);
    }

    let total_elapsed = start_total.elapsed();
    let total_secs = total_elapsed.as_secs_f64();
    let throughput_pps = num_packets as f64 / total_secs;

    let avg_latency = if !latencies_us.is_empty() {
        latencies_us.iter().sum::<u128>() as f64 / latencies_us.len() as f64
    } else {
        0.0
    };

    let min_latency = latencies_us.iter().min().copied().unwrap_or(0);
    let max_latency = latencies_us.iter().max().copied().unwrap_or(0);

    println!("\n[4] Benchmark Results:");
    println!("   Total Packets Transferred : {}", latencies_us.len());
    println!(
        "   Total Execution Time      : {:.3} ms",
        total_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "   Throughput                : {:.1} packets/sec",
        throughput_pps
    );
    println!(
        "   Average Latency           : {:.2} us ({:.3} ms)",
        avg_latency,
        avg_latency / 1000.0
    );
    println!(
        "   Min / Max Latency         : {} us / {} us",
        min_latency, max_latency
    );

    println!("\n=== ✅ Multimodal Streaming Client Benchmark Complete ===");
}

#[cfg(not(feature = "hkl2"))]
fn main() {
    println!("Run with --features hkl2 to execute the streaming client benchmark.");
}
