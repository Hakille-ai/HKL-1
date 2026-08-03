//! Native HKL Network API Server (`HklNativeServer`).
//! Multi-threaded, zero-dependency TCP & WebSocket server listening for
//! HKL Native Protocol (HKL-NP v1) binary packets and JSON control requests.
#![cfg(feature = "hkl2")]

use crate::api::cortex_service::CortexService;
use crate::api::protocol::{HklBinaryPacket, HklCommand, JsonFormatter};
use crate::core::math::FixedPoint;
use alloc::format;
use alloc::string::String;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::println;
use std::sync::{Arc, Mutex};

/// Multi-threaded Native API Server for HKL-1 / HKL-2
pub struct HklNativeServer {
    pub addr: String,
    pub service: Arc<Mutex<CortexService>>,
}

impl HklNativeServer {
    pub fn new(addr: &str, service: CortexService) -> Self {
        Self {
            addr: String::from(addr),
            service: Arc::new(Mutex::new(service)),
        }
    }

    /// Process an incoming binary packet and produce response packet
    pub fn handle_packet(
        service: &Arc<Mutex<CortexService>>,
        packet: &HklBinaryPacket,
    ) -> HklBinaryPacket {
        let mut guard = service.lock().unwrap();

        match packet.command {
            HklCommand::PerceiveFrame => {
                let res = guard.perceive(Some("stream"), None, None);
                let json = format!(
                    "{{\"text_tokens\":{},\"audio_spikes\":{},\"vision_spikes\":{},\"status\":\"{}\"}}",
                    res.text_tokens, res.audio_spikes, res.vision_spikes, res.status
                );
                HklBinaryPacket::new(
                    HklCommand::PerceiveFrame,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::SynthesizeResponse => {
                let text_prompt = String::from_utf8_lossy(&packet.payload);
                let res = guard.synthesize(&text_prompt, 4);
                let json = format!(
                    "{{\"text\":\"{}\",\"tokens_count\":{},\"pcm_samples\":{},\"dopamine\":{:.4}}}",
                    res.generated_text.replace('\n', "\\n").replace('"', "\\\""),
                    res.tokens.len(),
                    res.pcm_audio.len(),
                    res.dopamine
                );
                HklBinaryPacket::new(
                    HklCommand::SynthesizeResponse,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::EpropTrainStep => {
                let res = guard.train_eprop("hello", "world");
                let json = JsonFormatter::format_eprop_result(
                    res.step,
                    FixedPoint::from_f32(res.loss),
                    &res.status,
                );
                HklBinaryPacket::new(
                    HklCommand::EpropTrainStep,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::CognitiveState => {
                let state = guard.get_cognitive_state();
                let json = JsonFormatter::format_cognitive_state(
                    FixedPoint::from_f32(state.dopamine),
                    FixedPoint::from_f32(state.serotonin),
                    FixedPoint::from_f32(state.noradrenaline),
                    FixedPoint::from_f32(state.acetylcholine),
                    FixedPoint::from_f32(state.curiosity_score),
                    state.boredom_score,
                    &state.cognitive_mode,
                );
                HklBinaryPacket::new(
                    HklCommand::CognitiveState,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::XaiCausalTree => {
                let res = guard.explain_xai(42);
                let json = JsonFormatter::format_xai_tree(
                    res.target_neuron,
                    &res.causal_paths,
                    &res.dot_graph,
                );
                HklBinaryPacket::new(
                    HklCommand::XaiCausalTree,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::SiliconCompile => {
                let res = guard.compile_efpga();
                let json = JsonFormatter::format_silicon_compile(
                    res.verilog_lines,
                    res.bitstream_bytes,
                    &res.status,
                );
                HklBinaryPacket::new(
                    HklCommand::SiliconCompile,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            HklCommand::SwarmMeshStatus => {
                let res = guard.swarm_status();
                let json = format!(
                    "{{\"node_id\":\"{}\",\"role\":\"{}\",\"connected_peers\":{},\"active_routes\":{}}}",
                    res.node_id_hex, res.role, res.connected_peers, res.active_routes
                );
                HklBinaryPacket::new(
                    HklCommand::SwarmMeshStatus,
                    packet.timestamp_ms,
                    json.into_bytes(),
                )
            }
            _ => {
                let json = format!(
                    "{{\"error\":\"Unknown command 0x{:04X}\"}}",
                    packet.command as u16
                );
                HklBinaryPacket::new(HklCommand::Unknown, packet.timestamp_ms, json.into_bytes())
            }
        }
    }

    /// Process single TCP connection stream
    pub fn handle_client(mut stream: TcpStream, service: Arc<Mutex<CortexService>>) {
        let mut buffer = [0u8; 4096];
        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break, // Client disconnected
                Ok(bytes_read) => {
                    let mut cursor = 0;
                    while cursor < bytes_read {
                        if let Some((packet, consumed)) =
                            HklBinaryPacket::decode(&buffer[cursor..bytes_read])
                        {
                            cursor += consumed;
                            let response_packet = Self::handle_packet(&service, &packet);
                            let encoded_resp = response_packet.encode();
                            if stream.write_all(&encoded_resp).is_err() {
                                return;
                            }
                        } else {
                            // Check if HTTP request (e.g. GET /hkl/v1/cognitive/state)
                            let req_str = String::from_utf8_lossy(&buffer[cursor..bytes_read]);
                            if req_str.starts_with("GET /") || req_str.starts_with("POST /") {
                                let body = if req_str.contains("/cognitive") {
                                    let state = service.lock().unwrap().get_cognitive_state();
                                    JsonFormatter::format_cognitive_state(
                                        FixedPoint::from_f32(state.dopamine),
                                        FixedPoint::from_f32(state.serotonin),
                                        FixedPoint::from_f32(state.noradrenaline),
                                        FixedPoint::from_f32(state.acetylcholine),
                                        FixedPoint::from_f32(state.curiosity_score),
                                        state.boredom_score,
                                        &state.cognitive_mode,
                                    )
                                } else {
                                    String::from("{\"status\":\"HKL Native Protocol v1 Ready\"}")
                                };

                                let http_resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                                    body.len(),
                                    body
                                );
                                let _ = stream.write_all(http_resp.as_bytes());
                                return;
                            }
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Start the TCP server listening loop
    pub fn listen(&self) -> Result<(), String> {
        let listener = TcpListener::bind(&self.addr).map_err(|e| format!("Bind error: {}", e))?;
        println!(
            "🚀 HKL Native Protocol Server listening on hkl://{}",
            self.addr
        );

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let service_clone = Arc::clone(&self.service);
                    std::thread::spawn(move || {
                        Self::handle_client(stream, service_clone);
                    });
                }
                Err(e) => {
                    println!("Error accepting connection: {}", e);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_packet_cognitive_state() {
        let service = Arc::new(Mutex::new(CortexService::new([1, 2, 3, 4, 5, 6, 7, 8], 1)));
        let req = HklBinaryPacket::new(HklCommand::CognitiveState, 1000, alloc::vec![]);

        let resp = HklNativeServer::handle_packet(&service, &req);
        assert_eq!(resp.command, HklCommand::CognitiveState);

        let json = String::from_utf8_lossy(&resp.payload);
        assert!(json.contains("\"dopamine\":"));
    }
}
