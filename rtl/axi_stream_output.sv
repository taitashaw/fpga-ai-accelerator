// ===========================================================================
// Module:  axi_stream_output.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   AXI4-Stream master output interface for streaming quantized results.
//   Packs ARRAY_COLS × INT8 results into 64-bit AXI beats.
//   Supports AXI backpressure: holds TVALID high and data stable until
//   TREADY is asserted. TLAST marks the final beat of a tile/layer.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module axi_stream_output
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Quantized data input (from quantize_unit, packed vector)
  // iverilog 12 fix: packed to avoid unpacked array port connection bug
  input  logic [OUT_WIDTH*ARRAY_COLS-1:0] data_in_packed,
  input  logic                          data_valid,
  input  logic                          data_last,   // Last beat of tile/layer

  // AXI4-Stream Master Interface
  output logic [AXI_DATA_WIDTH-1:0]     m_axis_tdata,
  output logic [AXI_KEEP_WIDTH-1:0]     m_axis_tkeep,
  output logic                          m_axis_tvalid,
  input  logic                          m_axis_tready,
  output logic                          m_axis_tlast,

  // Status
  output logic                          backpressure  // Indicates stall (valid & !ready)
);

  // -------------------------------------------------------------------------
  // Pack input data into 64-bit AXI beat
  // -------------------------------------------------------------------------
  // Pack input data into 64-bit AXI beat
  // Input is already a packed vector — direct assignment
  logic [AXI_DATA_WIDTH-1:0] packed_data;
  assign packed_data = data_in_packed;

  // -------------------------------------------------------------------------
  // Output Register with Backpressure Support
  // -------------------------------------------------------------------------
  // When backpressure occurs (tvalid && !tready), hold data stable.
  // New data is accepted only when either:
  //   (a) Output is idle (!tvalid), or
  //   (b) Current beat was consumed (tvalid && tready)

  logic                          out_valid_reg;
  logic [AXI_DATA_WIDTH-1:0]     out_data_reg;
  logic                          out_last_reg;
  logic                          can_accept;

  assign can_accept = !out_valid_reg || (out_valid_reg && m_axis_tready);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid_reg <= 1'b0;
      out_data_reg  <= '0;
      out_last_reg  <= 1'b0;
    end else begin
      if (can_accept && data_valid) begin
        // Accept new data
        out_valid_reg <= 1'b1;
        out_data_reg  <= packed_data;
        out_last_reg  <= data_last;
      end else if (out_valid_reg && m_axis_tready) begin
        // Beat consumed, no new data pending
        out_valid_reg <= 1'b0;
        out_data_reg  <= '0;
        out_last_reg  <= 1'b0;
      end
      // else: hold (backpressure)
    end
  end

  // -------------------------------------------------------------------------
  // AXI4-Stream Output
  // -------------------------------------------------------------------------
  assign m_axis_tdata  = out_data_reg;
  assign m_axis_tvalid = out_valid_reg;
  assign m_axis_tlast  = out_last_reg;
  assign m_axis_tkeep  = {AXI_KEEP_WIDTH{1'b1}}; // All bytes valid

  // -------------------------------------------------------------------------
  // Backpressure indicator
  // -------------------------------------------------------------------------
  assign backpressure = out_valid_reg && !m_axis_tready;

endmodule
