// ===========================================================================
// Module:  sram_controller.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   Double-buffered (ping-pong) SRAM controller for weight storage.
//   Two 1024×64-bit banks — one serves compute reads while the other
//   accepts DMA weight writes. Banks swap on command after a layer tile
//   completes.
//
//   BRAM inference pattern:
//     - Synchronous read with output register (2-cycle read latency)
//     - No reset on memory array (only output register) for clean inference
//     - Write-first mode for simultaneous read/write to same address
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module sram_controller
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Bank control
  input  logic                          bank_swap,     // Swap active/inactive banks
  output logic                          active_bank,   // Currently active bank for reads

  // Compute read port (from systolic array via FSM)
  input  logic                          rd_en,
  input  logic [SRAM_ADDR_W-1:0]        rd_addr,
  output logic [SRAM_WIDTH-1:0]         rd_data,
  output logic                          rd_valid,      // Data valid (2 cycles after rd_en)

  // DMA write port (from external weight loader)
  input  logic                          wr_en,
  input  logic [SRAM_ADDR_W-1:0]        wr_addr,
  input  logic [SRAM_WIDTH-1:0]         wr_data,

  // Status
  output logic                          bank_conflict  // Read and write to same bank
);

  // -------------------------------------------------------------------------
  // Memory Arrays (BRAM inference)
  // -------------------------------------------------------------------------
  // No reset on memory — critical for BRAM inference on Xilinx/Intel
  (* ram_style = "block", DONT_TOUCH = "yes" *)
  logic [SRAM_WIDTH-1:0] mem_bank0 [SRAM_DEPTH];

  (* ram_style = "block", DONT_TOUCH = "yes" *)
  logic [SRAM_WIDTH-1:0] mem_bank1 [SRAM_DEPTH];

  // -------------------------------------------------------------------------
  // Bank Selection
  // -------------------------------------------------------------------------
  logic bank_sel;  // 0 = bank0 is active (compute reads), 1 = bank1 is active

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      bank_sel <= 1'b0;
    else if (bank_swap)
      bank_sel <= ~bank_sel;
  end

  assign active_bank = bank_sel;

  // Write goes to inactive bank, read from active bank
  logic wr_to_bank0, wr_to_bank1;
  logic rd_from_bank0, rd_from_bank1;

  assign wr_to_bank0   = wr_en && bank_sel;     // bank1 active → write bank0
  assign wr_to_bank1   = wr_en && !bank_sel;    // bank0 active → write bank1
  assign rd_from_bank0 = rd_en && !bank_sel;     // bank0 active → read bank0
  assign rd_from_bank1 = rd_en && bank_sel;      // bank1 active → read bank1

  // -------------------------------------------------------------------------
  // Bank Conflict Detection
  // -------------------------------------------------------------------------
  // Conflict: read and write targeting the same bank
  assign bank_conflict = (rd_en && wr_en) &&
                         ((rd_from_bank0 && wr_to_bank0) ||
                          (rd_from_bank1 && wr_to_bank1));

  // -------------------------------------------------------------------------
  // Bank 0: BRAM Read/Write (synchronous, 2-cycle read)
  // -------------------------------------------------------------------------
  logic [SRAM_WIDTH-1:0] bank0_rd_stage1;  // Stage 1: BRAM output
  logic [SRAM_WIDTH-1:0] bank0_rd_stage2;  // Stage 2: Output register

  // Synchronous write
  always_ff @(posedge clk) begin
    if (wr_to_bank0)
      mem_bank0[wr_addr] <= wr_data;
  end

  // Synchronous read — stage 1 (BRAM internal register)
  always_ff @(posedge clk) begin
    if (rd_from_bank0)
      bank0_rd_stage1 <= mem_bank0[rd_addr];
  end

  // Stage 2: Output register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      bank0_rd_stage2 <= '0;
    else
      bank0_rd_stage2 <= bank0_rd_stage1;
  end

  // -------------------------------------------------------------------------
  // Bank 1: BRAM Read/Write (synchronous, 2-cycle read)
  // -------------------------------------------------------------------------
  logic [SRAM_WIDTH-1:0] bank1_rd_stage1;
  logic [SRAM_WIDTH-1:0] bank1_rd_stage2;

  // Synchronous write
  always_ff @(posedge clk) begin
    if (wr_to_bank1)
      mem_bank1[wr_addr] <= wr_data;
  end

  // Synchronous read — stage 1
  always_ff @(posedge clk) begin
    if (rd_from_bank1)
      bank1_rd_stage1 <= mem_bank1[rd_addr];
  end

  // Stage 2: Output register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      bank1_rd_stage2 <= '0;
    else
      bank1_rd_stage2 <= bank1_rd_stage1;
  end

  // -------------------------------------------------------------------------
  // Read Output Mux
  // -------------------------------------------------------------------------
  assign rd_data = bank_sel ? bank1_rd_stage2 : bank0_rd_stage2;

  // -------------------------------------------------------------------------
  // Read Valid Pipeline (2-cycle latency tracking)
  // -------------------------------------------------------------------------
  logic rd_valid_pipe1;
  logic rd_valid_pipe2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_valid_pipe1 <= 1'b0;
      rd_valid_pipe2 <= 1'b0;
    end else begin
      rd_valid_pipe1 <= rd_en;
      rd_valid_pipe2 <= rd_valid_pipe1;
    end
  end

  assign rd_valid = rd_valid_pipe2;

endmodule
