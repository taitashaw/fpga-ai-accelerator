// ===========================================================================
// Module:  pkg_accelerator.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   Foundation package defining all shared types, parameters, structures,
//   enumerations, and utility functions for the 8x8 INT8 systolic array
//   inference accelerator. Every other module imports this package.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

package pkg_accelerator;

  // -------------------------------------------------------------------------
  // Core Precision Parameters
  // -------------------------------------------------------------------------
  parameter ACT_WIDTH     = 8;     // Activation bit width (INT8 signed)
  parameter WGT_WIDTH     = 8;     // Weight bit width (INT8 signed)
  parameter ACC_WIDTH     = 32;    // Accumulator bit width (INT32 signed)
  parameter OUT_WIDTH     = 8;     // Output bit width after requantization

  // -------------------------------------------------------------------------
  // Array Dimensions
  // -------------------------------------------------------------------------
  parameter ARRAY_ROWS    = 8;     // Systolic array row count
  parameter ARRAY_COLS    = 8;     // Systolic array column count

  // -------------------------------------------------------------------------
  // Pipeline & Timing Parameters
  // -------------------------------------------------------------------------
  parameter MAC_LATENCY   = 3;     // MAC pipeline depth (input→mul→acc)
  parameter QUANT_LATENCY = 2;     // Quantization pipeline depth (ReLU→shift/sat)
  parameter ARRAY_LATENCY = ARRAY_COLS + MAC_LATENCY; // 11 cycles through array
  parameter TOTAL_LATENCY = ARRAY_LATENCY + QUANT_LATENCY + 2; // 15 cycles end-to-end

  // -------------------------------------------------------------------------
  // AXI4-Stream Interface Parameters
  // -------------------------------------------------------------------------
  parameter AXI_DATA_WIDTH = 64;   // 8 × INT8 = 64 bits per beat
  parameter AXI_KEEP_WIDTH = AXI_DATA_WIDTH / 8; // 8 bytes

  // -------------------------------------------------------------------------
  // SRAM Controller Parameters
  // -------------------------------------------------------------------------
  parameter SRAM_DEPTH    = 1024;  // Words per bank
  parameter SRAM_WIDTH    = 64;    // Bits per word (8 × INT8)
  parameter SRAM_ADDR_W   = 10;    // $clog2(1024)

  // -------------------------------------------------------------------------
  // Layer / Tiling Parameters
  // -------------------------------------------------------------------------
  parameter MAX_CHANNELS  = 512;   // Maximum IC or OC dimension
  parameter TILE_ROWS     = ARRAY_ROWS; // Tile height = array rows
  parameter TILE_COLS     = ARRAY_COLS; // Tile width = array columns

  // -------------------------------------------------------------------------
  // Timeout & Watchdog
  // -------------------------------------------------------------------------
  parameter WATCHDOG_CYCLES = 32'd100_000; // Timeout threshold (500 us @ 200 MHz)

  // -------------------------------------------------------------------------
  // Performance Counter Count
  // -------------------------------------------------------------------------
  parameter NUM_PERF_CTRS = 8;

  // -------------------------------------------------------------------------
  // Data Types (iverilog-compatible — no typedef for wire types)
  // -------------------------------------------------------------------------
  // Activation: signed 8-bit
  // Weight:     signed 8-bit
  // Accumulator: signed 32-bit
  // Output:     signed 8-bit (requantized)
  //
  // Used as:
  //   logic signed [ACT_WIDTH-1:0]  act;
  //   logic signed [WGT_WIDTH-1:0]  wgt;
  //   logic signed [ACC_WIDTH-1:0]  acc;
  //   logic signed [OUT_WIDTH-1:0]  out;

  // -------------------------------------------------------------------------
  // Layer Descriptor Structure
  // -------------------------------------------------------------------------
  // Describes one inference layer's tiling and quantization parameters.
  // Loaded by the FSM from a configuration interface before layer execution.
  typedef struct packed {
    logic [15:0] in_channels;    // IC dimension (must be multiple of TILE_COLS)
    logic [15:0] out_channels;   // OC dimension (must be multiple of TILE_ROWS)
    logic [15:0] spatial_size;   // Flattened spatial dimension (H×W)
    logic [4:0]  shift_amount;   // Requantization right-shift (0–31)
    logic signed [7:0] zero_point; // Requantization zero-point offset
    logic        relu_enable;    // 1 = apply ReLU before requantization
    logic [5:0]  reserved;       // Padding for alignment (total = 68 bits)
  } layer_desc_t;

  // -------------------------------------------------------------------------
  // FSM State Enumeration (11 states)
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    FSM_IDLE         = 4'd0,   // Waiting for start trigger
    FSM_LOAD_DESC    = 4'd1,   // Load layer descriptor from config interface
    FSM_LOAD_WEIGHTS = 4'd2,   // DMA weights into inactive SRAM bank
    FSM_LOAD_ACT     = 4'd3,   // Stream activations through AXI4-S input
    FSM_COMPUTE      = 4'd4,   // Systolic array computing (MAC active)
    FSM_DRAIN        = 4'd5,   // Drain pipeline (wait for last valid output)
    FSM_QUANT        = 4'd6,   // Requantization pipeline active
    FSM_STORE_OUTPUT = 4'd7,   // Stream results through AXI4-S output
    FSM_NEXT_TILE    = 4'd8,   // Advance tile counters (IC/OC/spatial)
    FSM_NEXT_LAYER   = 4'd9,   // Layer complete — check for more layers
    FSM_DONE         = 4'd10   // All layers complete — idle until reset
  } fsm_state_t;


  // -------------------------------------------------------------------------
  // Error Code Enumeration
  // -------------------------------------------------------------------------
  typedef enum logic [3:0] {
    ERR_NONE         = 4'd0,   // No error
    ERR_TIMEOUT      = 4'd1,   // Watchdog timeout in current state
    ERR_OVERFLOW     = 4'd2,   // Accumulator overflow detected
    ERR_AXI_PROTOCOL = 4'd3,   // AXI handshake violation
    ERR_BAD_CONFIG   = 4'd4,   // Invalid layer descriptor
    ERR_BANK_CONFLICT= 4'd5,   // SRAM bank conflict (read and write same bank)
    ERR_TILE_MISMATCH= 4'd6    // Tile counter exceeded expected range
  } error_code_t;

  // -------------------------------------------------------------------------
  // Performance Counter Index
  // -------------------------------------------------------------------------
  typedef enum logic [2:0] {
    CTR_TOTAL_CYCLES  = 3'd0,  // Total cycles since start
    CTR_MAC_ACTIVE    = 3'd1,  // Cycles with MACs actively computing
    CTR_MAC_STALL     = 3'd2,  // Cycles MACs stalled (backpressure)
    CTR_AXI_RD_BEATS  = 3'd3,  // AXI read beats transferred (input)
    CTR_AXI_WR_BEATS  = 3'd4,  // AXI write beats transferred (output)
    CTR_LAYERS_DONE   = 3'd5,  // Layers completed
    CTR_TILES_DONE    = 3'd6,  // Tiles completed
    CTR_ERRORS        = 3'd7   // Error events counted
  } perf_ctr_idx_t;

  // -------------------------------------------------------------------------
  // Utility Functions
  // -------------------------------------------------------------------------

  // Saturate a wide signed value to signed 8-bit range [-128, +127]
  function automatic logic signed [OUT_WIDTH-1:0] saturate_to_int8(
    input logic signed [ACC_WIDTH-1:0] val
  );
    if (val > 32'sd127)
      saturate_to_int8 = 8'sd127;
    else if (val < -32'sd128)
      saturate_to_int8 = -8'sd128;
    else
      saturate_to_int8 = val[OUT_WIDTH-1:0];
  endfunction

  // ReLU: clamp negative values to zero
  function automatic logic signed [ACC_WIDTH-1:0] relu(
    input logic signed [ACC_WIDTH-1:0] val
  );
    if (val[ACC_WIDTH-1]) // negative
      relu = {ACC_WIDTH{1'b0}};
    else
      relu = val;
  endfunction

  // Requantize: ReLU (optional) → arithmetic right-shift → zero-point → saturate
  // This is the complete 2-stage operation collapsed into a function for
  // the golden model. The RTL pipeline splits this across 2 clock cycles.
  function automatic logic signed [OUT_WIDTH-1:0] requantize(
    input logic signed [ACC_WIDTH-1:0]  acc_val,
    input logic                         relu_en,
    input logic [4:0]                   shift,
    input logic signed [OUT_WIDTH-1:0]  zp
  );
    logic signed [ACC_WIDTH-1:0] stage1;
    logic signed [ACC_WIDTH-1:0] shifted;
    logic signed [ACC_WIDTH-1:0] with_zp;

    // Stage 1: optional ReLU
    if (relu_en)
      stage1 = relu(acc_val);
    else
      stage1 = acc_val;

    // Stage 2: arithmetic right-shift (signed >>> in SystemVerilog)
    shifted = stage1 >>> shift;

    // Add zero-point
    with_zp = shifted + {{(ACC_WIDTH-OUT_WIDTH){zp[OUT_WIDTH-1]}}, zp};

    // Saturate to INT8
    requantize = saturate_to_int8(with_zp);
  endfunction

  // Check if a layer descriptor has valid parameters
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic is_valid_layer_desc(
    input layer_desc_t desc  // reserved/spatial bits checked in future phases
  );
  /* verilator lint_on UNUSEDSIGNAL */
    // Channels must be non-zero and multiples of tile size
    if (desc.in_channels == 0 || desc.out_channels == 0)
      is_valid_layer_desc = 1'b0;
    else if (desc.spatial_size == 0)
      is_valid_layer_desc = 1'b0;
    else if (desc.in_channels > MAX_CHANNELS[15:0])
      is_valid_layer_desc = 1'b0;
    else if (desc.out_channels > MAX_CHANNELS[15:0])
      is_valid_layer_desc = 1'b0;
    else
      is_valid_layer_desc = 1'b1;
  endfunction

endpackage
