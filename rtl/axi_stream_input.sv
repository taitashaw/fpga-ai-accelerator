// ===========================================================================
// Module:  axi_stream_input.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   AXI4-Stream slave input interface with per-row diagonal skew buffers.
//   Receives 64-bit AXI beats (8 × INT8 activations), unpacks them into
//   individual row activations, and applies diagonal skew so that:
//     - Row 0 gets activation immediately (0-stage delay)
//     - Row 1 gets activation delayed by 1 cycle
//     - Row i gets activation delayed by i cycles
//
//   This diagonal skewing creates the correct wavefront timing required
//   for weight-stationary systolic execution without global synchronization.
//
//   AXI4-Stream protocol: TVALID/TREADY handshake, TKEEP, TLAST.
//   Data transfer occurs only when both TVALID and TREADY are asserted.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module axi_stream_input
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Control
  input  logic                          enable,         // Module enable (shift registers)
  input  logic                          accept,         // Accept new AXI data (gate tready)
  input  logic                          flush,          // Clear skew buffers

  // AXI4-Stream Slave Interface
  input  logic [AXI_DATA_WIDTH-1:0]     s_axis_tdata,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [AXI_KEEP_WIDTH-1:0]     s_axis_tkeep,   // Part of AXI spec, all bytes valid
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                          s_axis_tvalid,
  output logic                          s_axis_tready,
  input  logic                          s_axis_tlast,

  // Skewed activation outputs (one per row, to systolic array west boundary)
  // iverilog 12 fix: use packed output to avoid unpacked array port connection bug
  output logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_out_packed,
  output logic                          act_valid,       // All rows have valid data
  output logic                          frame_last       // Last beat of activation frame
);

  // -------------------------------------------------------------------------
  // AXI Handshake — one-shot per spatial position
  // -------------------------------------------------------------------------
  logic handshake;
  logic beat_done;  // Set after one handshake, prevents accepting more

  assign s_axis_tready = accept && rst_n && !beat_done;
  assign handshake     = s_axis_tvalid && s_axis_tready;

  // Track whether we've consumed one AXI beat this spatial position
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      beat_done <= 1'b0;
    else if (flush)
      beat_done <= 1'b0;
    else if (handshake)
      beat_done <= 1'b1;
  end

  // -------------------------------------------------------------------------
  // Unpack 64-bit AXI beat into 8 × INT8 activations
  // -------------------------------------------------------------------------
  logic signed [ACT_WIDTH-1:0] unpacked [ARRAY_ROWS];

  genvar ui;
  generate
    for (ui = 0; ui < ARRAY_ROWS; ui = ui + 1) begin : gen_unpack
      assign unpacked[ui] = $signed(s_axis_tdata[ui*ACT_WIDTH +: ACT_WIDTH]);
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Diagonal Skew Buffers
  // -------------------------------------------------------------------------
  // Row 0: 0 stages of delay (direct pass-through)
  // Row 1: 1 stage of delay
  // Row i: i stages of delay
  //
  // This creates the diagonal wavefront needed for systolic operation.
  // Each skew buffer is a shift register of depth = row index.

  // Row 0: 1-stage registered (sets the baseline latency)
  logic signed [ACT_WIDTH-1:0] skew_out_0;
  logic                        skew_valid_0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      skew_out_0   <= '0;
      skew_valid_0 <= 1'b0;
    end else if (flush) begin
      skew_out_0   <= '0;
      skew_valid_0 <= 1'b0;
    end else if (enable) begin
      skew_out_0   <= handshake ? unpacked[0] : '0;
      skew_valid_0 <= handshake;
    end
  end

  // Rows 1–7: shift register skew buffers (depth = row index)
  // Use generate to create per-row shift registers
  // iverilog fix: use intermediate regs, assign to output via generate

  // Shift register storage: skew_sr[row][stage]
  // For row r, we need (r+1) stages total to get r cycles of relative delay
  // vs row 0 (which has 1 register stage). Max depth = ARRAY_ROWS = 8.
  logic signed [ACT_WIDTH-1:0] skew_sr [1:ARRAY_ROWS-1][0:ARRAY_ROWS-1];
  logic                        skew_vld_sr [1:ARRAY_ROWS-1][0:ARRAY_ROWS-1];

  genvar ri, si;
  generate
    for (ri = 1; ri < ARRAY_ROWS; ri = ri + 1) begin : gen_skew_row
      // Shift register for row ri: depth = ri+1 stages (ri cycles relative delay)

      // Stage 0: input from unpacker
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          skew_sr[ri][0]     <= '0;
          skew_vld_sr[ri][0] <= 1'b0;
        end else if (flush) begin
          skew_sr[ri][0]     <= '0;
          skew_vld_sr[ri][0] <= 1'b0;
        end else if (enable) begin
          skew_sr[ri][0]     <= handshake ? unpacked[ri] : '0;
          skew_vld_sr[ri][0] <= handshake;
        end
      end

      // Subsequent stages: chain (stages 1 through ri)
      for (si = 1; si <= ri; si = si + 1) begin : gen_stage
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            skew_sr[ri][si]     <= '0;
            skew_vld_sr[ri][si] <= 1'b0;
          end else if (flush) begin
            skew_sr[ri][si]     <= '0;
            skew_vld_sr[ri][si] <= 1'b0;
          end else if (enable) begin
            skew_sr[ri][si]     <= skew_sr[ri][si-1];
            skew_vld_sr[ri][si] <= skew_vld_sr[ri][si-1];
          end
        end
      end
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Output Assignment: pack into flat vector (iverilog 12 compatibility)
  // -------------------------------------------------------------------------
  // Row 0: direct from registered input
  assign act_out_packed[0*ACT_WIDTH +: ACT_WIDTH] = skew_out_0;

  // Rows 1+: output from last stage of shift register
  genvar oi;
  generate
    for (oi = 1; oi < ARRAY_ROWS; oi = oi + 1) begin : gen_out
      assign act_out_packed[oi*ACT_WIDTH +: ACT_WIDTH] = skew_sr[oi][oi];
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Valid: asserted when row 0 has valid data
  // (downstream pipeline manages latency alignment)
  // -------------------------------------------------------------------------
  assign act_valid = skew_valid_0;

  // -------------------------------------------------------------------------
  // TLAST tracking
  // -------------------------------------------------------------------------
  logic tlast_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      tlast_reg <= 1'b0;
    else if (flush)
      tlast_reg <= 1'b0;
    else if (handshake)
      tlast_reg <= s_axis_tlast;
    else
      tlast_reg <= 1'b0;
  end

  assign frame_last = tlast_reg;

endmodule
