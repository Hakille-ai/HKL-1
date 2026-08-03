//! Synthesizable Verilog HDL Generator for HKL-1 eFPGA Bio-Compilation.
//! Generates Verilog RTL modules (`module efpga_snn_subnetwork(...)`) from frozen SNN sub-networks.

use crate::efpga::stability::FrozenSubnetwork;
use core::fmt::Write;

pub const MAX_VERILOG_BUFFER_LEN: usize = 2048;

/// Synthesizable Verilog HDL Generator
pub struct HdlGenerator;

impl HdlGenerator {
    /// Generate synthesizable Verilog HDL code for a frozen SNN sub-network
    pub fn generate_verilog_hdl(
        subnetwork: &FrozenSubnetwork,
        buffer: &mut [u8; MAX_VERILOG_BUFFER_LEN],
    ) -> usize {
        let mut writer = crate::core::text::FixedTextBuffer::new(buffer);
        let header = b"// HKL-1 eFPGA Synthesizable Verilog HDL Subnetwork RTL\nmodule efpga_snn_subnetwork (\n  input wire clk,\n  input wire rst_n,\n  input wire [15:0] in_spikes,\n  output reg [15:0] out_spikes\n);\n\n  // Internal LIF Membrane Potentials\n  reg signed [15:0] V_memb [0:15];\n  parameter THRESHOLD = 16'h0100;\n\n  always @(posedge clk or negedge rst_n) begin\n    if (!rst_n) begin\n      out_spikes <= 16'b0;\n    end else begin\n";

        writer.write_bytes(header);

        // Generate Verilog logic statements for frozen synapses
        for i in 0..subnetwork.count {
            if let Some(syn) = subnetwork.synapses[i] {
                let src_idx = syn.source_id.index() % 16;
                let tgt_idx = syn.target_id.index() % 16;
                let w_raw = syn.weight.0;

                let _ = writeln!(
                    writer,
                    "      if (in_spikes[{}]) V_memb[{}] <= V_memb[{}] + 16'd{};\n",
                    src_idx, tgt_idx, tgt_idx, w_raw
                );
            }
        }

        let footer = b"    end\n  end\nendmodule\n";
        writer.write_bytes(footer);

        writer.len()
    }
}
