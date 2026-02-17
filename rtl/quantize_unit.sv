// ===========================================================================
// Module:  quantize_unit.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   2-stage pipelined requantization unit operating on ARRAY_COLS outputs
//   simultaneously. Converts 32-bit accumulator values to 8-bit outputs.
//
//   Stage 1: Optional ReLU (clamp negatives to zero)
//   Stage 2: Arithmetic right-shift → zero-point add → saturate to [-128, +127]
//
//   Saturation flags are output per-column for coverage tracking.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module quantize_unit
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Control
  input  logic                          enable,
  input  logic                          relu_en,         // Enable ReLU in stage 1
  input  logic [4:0]                    shift_amount,    // Right-shift (0–31)
  input  logic signed [OUT_WIDTH-1:0]   zero_point,      // Output zero-point offset

  // Input: accumulator values from systolic array (packed vector, one per column)
  // iverilog 12 fix: packed to avoid unpacked array port connection bug
  input  logic [ACC_WIDTH*ARRAY_COLS-1:0] acc_in_packed,
  input  logic                          acc_valid,

  // Output: quantized INT8 values (packed vector)
  // iverilog 12 fix: packed to avoid unpacked array port connection bug
  output logic [OUT_WIDTH*ARRAY_COLS-1:0] quant_out_packed,
  output logic                          quant_valid,
  output logic                          any_saturated  // At least one output saturated
);

  // -------------------------------------------------------------------------
  // Stage 1 Registers: ReLU
  // -------------------------------------------------------------------------
  logic signed [ACC_WIDTH-1:0] stage1_data [ARRAY_COLS];
  logic                        stage1_valid;

  genvar s1;
  generate
    for (s1 = 0; s1 < ARRAY_COLS; s1 = s1 + 1) begin : gen_stage1
      // Unpack from packed input vector
      logic signed [ACC_WIDTH-1:0] acc_in_elem;
      assign acc_in_elem = $signed(acc_in_packed[s1*ACC_WIDTH +: ACC_WIDTH]);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          stage1_data[s1] <= '0;
        end else if (enable && acc_valid) begin
          // ReLU: clamp negative to zero if enabled
          if (relu_en && acc_in_elem[ACC_WIDTH-1])
            stage1_data[s1] <= '0;
          else
            stage1_data[s1] <= acc_in_elem;
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      stage1_valid <= 1'b0;
    else if (enable)
      stage1_valid <= acc_valid;
    else
      stage1_valid <= 1'b0;
  end

  // -------------------------------------------------------------------------
  // Stage 2 Registers: Shift → Zero-Point → Saturate
  // -------------------------------------------------------------------------
  logic signed [OUT_WIDTH-1:0] stage2_data [ARRAY_COLS];
  logic                        stage2_valid;
  logic [ARRAY_COLS-1:0]       sat_flags;

  genvar s2;
  generate
    for (s2 = 0; s2 < ARRAY_COLS; s2 = s2 + 1) begin : gen_stage2

      // Combinational: shift → add zero-point → check saturation
      logic signed [ACC_WIDTH-1:0] shifted;
      logic signed [ACC_WIDTH-1:0] with_zp;
      logic                        saturated;

      assign shifted   = stage1_data[s2] >>> shift_amount;
      assign with_zp   = shifted + {{(ACC_WIDTH-OUT_WIDTH){zero_point[OUT_WIDTH-1]}}, zero_point};
      assign saturated = (with_zp > 32'sd127) || (with_zp < -32'sd128);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          stage2_data[s2] <= '0;
        end else if (enable && stage1_valid) begin
          // Saturate to INT8 range
          if (with_zp > 32'sd127)
            stage2_data[s2] <= 8'sd127;
          else if (with_zp < -32'sd128)
            stage2_data[s2] <= -8'sd128;
          else
            stage2_data[s2] <= with_zp[OUT_WIDTH-1:0];
        end
      end

      // Register saturation flag
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          sat_flags[s2] <= 1'b0;
        else if (enable && stage1_valid)
          sat_flags[s2] <= saturated;
        else
          sat_flags[s2] <= 1'b0;
      end

    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      stage2_valid <= 1'b0;
    else if (enable)
      stage2_valid <= stage1_valid;
    else
      stage2_valid <= 1'b0;
  end

  // -------------------------------------------------------------------------
  // Output Assignments
  // -------------------------------------------------------------------------
  // Output Assignments — pack into flat vector
  genvar oi;
  generate
    for (oi = 0; oi < ARRAY_COLS; oi = oi + 1) begin : gen_out
      assign quant_out_packed[oi*OUT_WIDTH +: OUT_WIDTH] = stage2_data[oi];
    end
  endgenerate

  assign quant_valid   = stage2_valid;
  assign any_saturated = |sat_flags;

endmodule
