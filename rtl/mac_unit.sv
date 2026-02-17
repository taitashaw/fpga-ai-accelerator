// ===========================================================================
// Module:  mac_unit.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   3-stage pipelined Multiply-Accumulate (MAC) processing element for the
//   systolic array. Each PE receives an activation from the west and a weight
//   from the north, computes act × wgt, and accumulates into a 32-bit register.
//
//   Pipeline stages (maps to DSP48E2 A/B → M → P):
//     Stage 1: Input register (captures act and wgt)
//     Stage 2: Signed 8×8 → 16-bit multiply
//     Stage 3: 32-bit accumulate with overflow detection
//
//   The PE also passes activation east and weight south for systolic flow.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module mac_unit
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Control
  input  logic                          clear_acc,   // Reset accumulator to 0
  input  logic                          enable,      // Pipeline enable (stall when low)

  // West input (activation from left neighbor or input buffer)
  input  logic signed [ACT_WIDTH-1:0]   act_in,
  // North input (weight from above neighbor or SRAM)
  input  logic signed [WGT_WIDTH-1:0]   wgt_in,

  // East output (activation passed to right neighbor)
  output logic signed [ACT_WIDTH-1:0]   act_out,
  // South output (weight passed to below neighbor)
  output logic signed [WGT_WIDTH-1:0]   wgt_out,

  // Accumulator output
  output logic signed [ACC_WIDTH-1:0]   acc_out,

  // Status
  output logic                          overflow
);

  // -------------------------------------------------------------------------
  // Stage 1: Input Registers (DSP48E2 A/B registers)
  // -------------------------------------------------------------------------
  logic signed [ACT_WIDTH-1:0] act_reg;
  logic signed [WGT_WIDTH-1:0] wgt_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_reg <= '0;
      wgt_reg <= '0;
    end else if (clear_acc) begin
      act_reg <= '0;
      wgt_reg <= '0;
    end else if (enable) begin
      act_reg <= act_in;
      wgt_reg <= wgt_in;
    end
  end

  // Systolic data flow: pass activation east and weight south
  // These are registered outputs — 1 cycle delay per PE hop
  assign act_out = act_reg;
  assign wgt_out = wgt_reg;

  // -------------------------------------------------------------------------
  // Stage 2: Multiply (DSP48E2 M register)
  // -------------------------------------------------------------------------
  // Signed 8×8 → 16-bit product
  (* use_dsp = "yes" *)
  logic signed [ACT_WIDTH+WGT_WIDTH-1:0] mul_result;
  logic signed [ACT_WIDTH+WGT_WIDTH-1:0] mul_reg;

  // Combinational multiply
  assign mul_result = act_reg * wgt_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mul_reg <= '0;
    end else if (clear_acc) begin
      mul_reg <= '0;
    end else if (enable) begin
      mul_reg <= mul_result;
    end
  end

  // -------------------------------------------------------------------------
  // Stage 3: Accumulate (DSP48E2 P register)
  // -------------------------------------------------------------------------
  logic signed [ACC_WIDTH-1:0] acc_reg;
  logic                        ovf_reg;

  // Sign-extend multiply result to accumulator width
  logic signed [ACC_WIDTH-1:0] mul_extended;
  assign mul_extended = {{(ACC_WIDTH-ACT_WIDTH-WGT_WIDTH){mul_reg[ACT_WIDTH+WGT_WIDTH-1]}}, mul_reg};

  // Overflow detection: check if addition overflows
  // Overflow occurs when adding two same-sign values produces different sign
  logic signed [ACC_WIDTH-1:0] sum_next;
  logic                        ovf_detect;

  assign sum_next   = acc_reg + mul_extended;
  assign ovf_detect = (acc_reg[ACC_WIDTH-1] == mul_extended[ACC_WIDTH-1]) &&
                      (sum_next[ACC_WIDTH-1] != acc_reg[ACC_WIDTH-1]);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_reg <= '0;
      ovf_reg <= 1'b0;
    end else if (clear_acc) begin
      acc_reg <= '0;
      ovf_reg <= 1'b0;
    end else if (enable) begin
      acc_reg <= sum_next;
      ovf_reg <= ovf_reg | ovf_detect; // Sticky overflow flag
    end
  end

  assign acc_out  = acc_reg;
  assign overflow = ovf_reg;

endmodule
