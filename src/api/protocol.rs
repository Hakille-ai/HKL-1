//! Native HKL Protocol v1 (HKL-NP v1) Packet Serialization & Framing.
//! Zero-dependency binary packet framing and JSON payload formatting
//! specifically designed for neuromorphic spiking streams, e-prop training,
//! cognitive telemetry, XAI causal trees, and eFPGA silicon compilation.
#![cfg(feature = "hkl2")]

use crate::core::math::FixedPoint;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub const HKL_MAGIC: [u8; 2] = *b"HK";
pub const HKL_HEADER_SIZE: usize = 16;

/// Command Identifiers for HKL Native Protocol
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HklCommand {
    PerceiveFrame = 0x0001,
    SynthesizeResponse = 0x0002,
    EpropTrainStep = 0x0003,
    CognitiveState = 0x0004,
    XaiCausalTree = 0x0005,
    SiliconCompile = 0x0006,
    SwarmMeshStatus = 0x0007,
    SystemSnapshot = 0x0008,
    Unknown = 0xFFFF,
}

impl From<u16> for HklCommand {
    fn from(val: u16) -> Self {
        match val {
            0x0001 => HklCommand::PerceiveFrame,
            0x0002 => HklCommand::SynthesizeResponse,
            0x0003 => HklCommand::EpropTrainStep,
            0x0004 => HklCommand::CognitiveState,
            0x0005 => HklCommand::XaiCausalTree,
            0x0006 => HklCommand::SiliconCompile,
            0x0007 => HklCommand::SwarmMeshStatus,
            0x0008 => HklCommand::SystemSnapshot,
            _ => HklCommand::Unknown,
        }
    }
}

/// Binary Packet Frame for HKL Native Stream Interface
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HklBinaryPacket {
    pub command: HklCommand,
    pub timestamp_ms: u64,
    pub payload: Vec<u8>,
}

impl HklBinaryPacket {
    pub fn new(command: HklCommand, timestamp_ms: u64, payload: Vec<u8>) -> Self {
        Self {
            command,
            timestamp_ms,
            payload,
        }
    }

    /// Encode packet into binary byte stream
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.payload.len() as u32;
        let mut buf = Vec::with_capacity(HKL_HEADER_SIZE + self.payload.len());

        buf.extend_from_slice(&HKL_MAGIC);
        buf.extend_from_slice(&(self.command as u16).to_be_bytes());
        buf.extend_from_slice(&self.timestamp_ms.to_be_bytes());
        buf.extend_from_slice(&payload_len.to_be_bytes());
        buf.extend_from_slice(&self.payload);

        buf
    }

    /// Decode binary byte stream into packet frame
    pub fn decode(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < HKL_HEADER_SIZE {
            return None;
        }

        if bytes[0] != HKL_MAGIC[0] || bytes[1] != HKL_MAGIC[1] {
            return None;
        }

        let cmd_raw = u16::from_be_bytes([bytes[2], bytes[3]]);
        let timestamp_ms = u64::from_be_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        let payload_len = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

        let total_size = HKL_HEADER_SIZE + payload_len;
        if bytes.len() < total_size {
            return None; // Incomplete payload
        }

        let payload = bytes[HKL_HEADER_SIZE..total_size].to_vec();

        Some((
            HklBinaryPacket {
                command: HklCommand::from(cmd_raw),
                timestamp_ms,
                payload,
            },
            total_size,
        ))
    }
}

/// Helper for generating lightweight JSON representations without external crates
pub struct JsonFormatter;

impl JsonFormatter {
    pub fn format_cognitive_state(
        da: FixedPoint,
        sht: FixedPoint,
        na: FixedPoint,
        ach: FixedPoint,
        curiosity: FixedPoint,
        boredom: f32,
        mode: &str,
    ) -> String {
        format!(
            "{{\"dopamine\":{:.4},\"serotonin\":{:.4},\"noradrenaline\":{:.4},\"acetylcholine\":{:.4},\"curiosity\":{:.4},\"boredom\":{:.4},\"mode\":\"{}\"}}",
            da.to_f32(),
            sht.to_f32(),
            na.to_f32(),
            ach.to_f32(),
            curiosity.to_f32(),
            boredom,
            mode
        )
    }

    pub fn format_eprop_result(step: u64, loss: FixedPoint, status: &str) -> String {
        format!(
            "{{\"step\":{},\"loss\":{:.4},\"status\":\"{}\"}}",
            step,
            loss.to_f32(),
            status
        )
    }

    pub fn format_xai_tree(neuron_id: u16, paths: &[String], dot_graph: &str) -> String {
        let escaped_dot = dot_graph.replace('\n', "\\n").replace('"', "\\\"");
        let paths_count = paths.len();
        format!(
            "{{\"target_neuron\":{},\"causal_paths_count\":{},\"dot_graph\":\"{}\"}}",
            neuron_id, paths_count, escaped_dot
        )
    }

    pub fn format_silicon_compile(
        verilog_lines: usize,
        bitstream_bytes: usize,
        status: &str,
    ) -> String {
        format!(
            "{{\"verilog_lines\":{},\"bitstream_bytes\":{},\"status\":\"{}\"}}",
            verilog_lines, bitstream_bytes, status
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hkl_binary_packet_encode_decode_roundtrip() {
        let payload = alloc::vec![1, 2, 3, 4, 5, 6, 7, 8];
        let packet = HklBinaryPacket::new(HklCommand::PerceiveFrame, 123456789, payload.clone());

        let encoded = packet.encode();
        assert_eq!(encoded.len(), HKL_HEADER_SIZE + payload.len());

        let (decoded, consumed) =
            HklBinaryPacket::decode(&encoded).expect("Packet decoding failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.command, HklCommand::PerceiveFrame);
        assert_eq!(decoded.timestamp_ms, 123456789);
        assert_eq!(decoded.payload, payload);
    }
}
