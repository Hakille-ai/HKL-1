//! Embedded eFPGA Bitstream Configurator for HKL-1.
//! Encodes frozen SNN sub-networks into binary bitstream config arrays for LUT4/LUT6 tables and routing switch matrices.

use crate::efpga::stability::FrozenSubnetwork;

pub const BITSTREAM_SIZE_BYTES: usize = 512;

/// eFPGA Binary Bitstream Descriptor
#[derive(Clone, Copy)]
pub struct BitstreamConfig {
    pub subnetwork_id: u32,
    pub data: [u8; BITSTREAM_SIZE_BYTES],
    pub valid_bytes: usize,
    pub checksum: u32,
}

/// Bitstream & LUT Configurator
pub struct BitstreamEncoder;

impl BitstreamEncoder {
    /// Encode a frozen sub-network into eFPGA binary bitstream configuration array
    pub fn encode_bitstream(subnetwork: &FrozenSubnetwork) -> BitstreamConfig {
        let mut config = BitstreamConfig {
            subnetwork_id: subnetwork.id,
            data: [0u8; BITSTREAM_SIZE_BYTES],
            valid_bytes: 0,
            checksum: 0,
        };

        // Bitstream Header: Subnetwork ID & Synapse Count
        config.data[0] = 0xEB; // Sync byte 'eB'
        config.data[1] = 0x01; // Protocol Version 1
        config.data[2] = (subnetwork.id & 0xFF) as u8;
        config.data[3] = (subnetwork.count & 0xFF) as u8;

        let mut offset = 4;
        let mut crc: u32 = 0x12345678;

        for i in 0..subnetwork.count {
            if let Some(syn) = subnetwork.synapses[i] {
                let src = (syn.source_id.index() & 0xFF) as u8;
                let tgt = (syn.target_id.index() & 0xFF) as u8;
                let w_raw = syn.weight.0;


                config.data[offset] = src;
                config.data[offset + 1] = tgt;
                config.data[offset + 2] = (w_raw & 0xFF) as u8;
                config.data[offset + 3] = ((w_raw >> 8) & 0xFF) as u8;

                crc = crc.wrapping_add(src as u32 + tgt as u32 + w_raw as u32);
                offset += 4;
            }
        }

        config.valid_bytes = offset;
        config.checksum = crc;
        config
    }
}
