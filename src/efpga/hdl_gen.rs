//! Synthesizable Verilog HDL Generator for HKL-1 eFPGA Bio-Compilation.
//! Generates Verilog RTL modules (`module efpga_snn_subnetwork(...)`) from frozen SNN sub-networks.

use crate::efpga::stability::FrozenSubnetwork;

pub const MAX_VERILOG_BUFFER_LEN: usize = 2048;

/// Synthesizable Verilog HDL Generator
pub struct HdlGenerator;

impl HdlGenerator {
    /// Generate synthesizable Verilog HDL code for a frozen SNN sub-network
    pub fn generate_verilog_hdl(
        subnetwork: &FrozenSubnetwork,
        buffer: &mut [u8; MAX_VERILOG_BUFFER_LEN],
    ) -> usize {
        let header = b"// HKL-1 eFPGA Synthesizable Verilog HDL Subnetwork RTL\nmodule efpga_snn_subnetwork (\n  input wire clk,\n  input wire rst_n,\n  input wire [15:0] in_spikes,\n  output reg [15:0] out_spikes\n);\n\n  // Internal LIF Membrane Potentials\n  reg signed [15:0] V_memb [0:15];\n  parameter THRESHOLD = 16'h0100;\n\n  always @(posedge clk or negedge rst_n) begin\n    if (!rst_n) begin\n      out_spikes <= 16'b0;\n    end else begin\n";

        let mut offset = 0;
        let h_len = header.len();
        buffer[..h_len].copy_from_slice(header);
        offset += h_len;

        // Generate Verilog logic statements for frozen synapses
        for i in 0..subnetwork.count {
            if let Some(syn) = subnetwork.synapses[i] {
                let src_idx = syn.source_id.index() % 16;
                let tgt_idx = syn.target_id.index() % 16;
                let w_raw = syn.weight.0;

                let line = alloc::format!(
                    "      if (in_spikes[{}]) V_memb[{}] <= V_memb[{}] + 16'd{};\n",
                    src_idx,
                    tgt_idx,
                    tgt_idx,
                    w_raw
                );
                let bytes = line.as_bytes();
                if offset + bytes.len() < MAX_VERILOG_BUFFER_LEN - 100 {
                    buffer[offset..offset + bytes.len()].copy_from_slice(bytes);
                    offset += bytes.len();
                }
            }
        }

        let footer = b"    end\n  end\nendmodule\n";
        let f_len = footer.len();
        if offset + f_len <= MAX_VERILOG_BUFFER_LEN {
            buffer[offset..offset + f_len].copy_from_slice(footer);
            offset += f_len;
        }

        offset
    }
}
