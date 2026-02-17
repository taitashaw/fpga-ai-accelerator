// ===========================================================================
// Module:  inference_accelerator_top.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   Top-level integration module for the AI Inference Accelerator.
//   Instantiates all 8 submodules and routes inter-module signals:
//     - AXI4-Stream input  → diagonal skew → systolic array
//     - SRAM controller    → weight feed to systolic array
//     - Systolic array     → quantize unit → AXI4-Stream output
//     - Layer control FSM  → orchestrates all control signals
//
//   External interfaces:
//     - AXI4-Stream slave  (activation input)
//     - AXI4-Stream master (result output)
//     - Weight load port   (DMA to SRAM)
//     - Control/status     (start, done, error, perf counters)
//     - Layer descriptor   (configuration input)
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module inference_accelerator_top
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // ===================== Control Interface =====================
  input  logic                          start,
  input  logic                          abort_req,
  output logic                          busy,
  output logic                          done,

  // Layer descriptor
  input  layer_desc_t                   layer_desc_in,
  input  logic                          layer_desc_valid,
  output logic                          layer_desc_ready,

  // ===================== AXI4-Stream Slave (Activation Input) =====================
  input  logic [AXI_DATA_WIDTH-1:0]     s_axis_tdata,
  input  logic [AXI_KEEP_WIDTH-1:0]     s_axis_tkeep,
  input  logic                          s_axis_tvalid,
  output logic                          s_axis_tready,
  input  logic                          s_axis_tlast,

  // ===================== AXI4-Stream Master (Result Output) =====================
  output logic [AXI_DATA_WIDTH-1:0]     m_axis_tdata,
  output logic [AXI_KEEP_WIDTH-1:0]     m_axis_tkeep,
  output logic                          m_axis_tvalid,
  input  logic                          m_axis_tready,
  output logic                          m_axis_tlast,

  // ===================== Weight Load Port (DMA → SRAM) =====================
  input  logic                          wgt_wr_en,
  input  logic [SRAM_ADDR_W-1:0]        wgt_wr_addr,
  input  logic [SRAM_WIDTH-1:0]         wgt_wr_data,
  input  logic                          wgt_load_done,

  // ===================== Status & Debug =====================
  output logic [3:0]                    current_state,
  output logic                          error_flag,
  output error_code_t                   error_code,
  output logic                          any_overflow,
  output logic [31:0]                   perf_counters [NUM_PERF_CTRS],

  // Debug port: exposes internal status for monitoring/testbench
  output logic                          dbg_sram_active_bank,
  output logic                          dbg_sram_bank_conflict,
  output logic                          dbg_sram_rd_valid,
  output logic                          dbg_act_valid,
  output logic                          dbg_act_frame_last,
  output logic                          dbg_quant_valid,
  output logic                          dbg_quant_saturated,
  output logic                          dbg_out_backpressure,
  output logic                          dbg_wgt_load_start
);

  // =========================================================================
  // Internal Wires
  // =========================================================================

  // --- FSM → SRAM ---
  logic                    fsm_sram_rd_en;
  logic [SRAM_ADDR_W-1:0]  fsm_sram_rd_addr;
  logic                    fsm_sram_bank_swap;

  // --- SRAM → Array ---
  logic [SRAM_WIDTH-1:0]   sram_rd_data;
  logic                    sram_rd_valid;
  logic                    sram_active_bank;
  logic                    sram_bank_conflict;

  // --- FSM → AXI Input ---
  logic                    fsm_act_enable;
  logic                    fsm_act_accept;
  logic                    fsm_act_flush;

  // --- AXI Input → Array ---
  // iverilog 12 fix: use packed vectors for inter-module connections
  logic [ACT_WIDTH*ARRAY_ROWS-1:0] skew_act_packed;
  logic                        skew_act_valid;
  logic                        skew_frame_last;

  // --- SRAM data unpacked to weight columns ---
  logic signed [WGT_WIDTH-1:0] wgt_cols [ARRAY_COLS];

  // --- FSM → Array ---
  logic                    fsm_array_enable;
  logic                    fsm_array_clear;
  logic                    fsm_array_valid_reset; // Reset pipeline valid counter each tile
  // fsm_array_data_valid removed — array uses sram_rd_valid as data_valid_in
  // fsm_out_valid removed — output uses quant_valid_pulse directly
  logic fsm_array_data_valid_unused; /* verilator lint_off UNUSEDSIGNAL */
  logic fsm_out_valid_unused;        /* verilator lint_off UNUSEDSIGNAL */

  // --- Array → Quantize ---
  logic [ACC_WIDTH*ARRAY_COLS-1:0] array_acc_packed;
  logic                        array_data_valid_out;
  logic                        array_overflow;

  // --- FSM → Quantize ---
  logic                    fsm_quant_enable;
  logic                    fsm_quant_relu;
  logic [4:0]              fsm_quant_shift;
  logic signed [OUT_WIDTH-1:0] fsm_quant_zp;

  // --- Quantize → Output ---
  logic [OUT_WIDTH*ARRAY_COLS-1:0] quant_data_packed;
  logic                        quant_valid;
  logic                        quant_valid_d;   // Previous cycle
  logic                        quant_valid_pulse; // Rising-edge pulse

  // Detect rising edge of quant_valid → 1-cycle pulse
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      quant_valid_d <= 1'b0;
    else
      quant_valid_d <= quant_valid;
  end
  assign quant_valid_pulse = quant_valid && !quant_valid_d;
  logic                        quant_saturated;

  // --- FSM → Output ---
  // fsm_out_valid removed — output uses quant_valid_pulse directly
  logic                    fsm_out_last;
  logic                    out_backpressure;

  // --- FSM misc ---
  logic                    fsm_wgt_load_start;
  logic                    out_accepted;

  // =========================================================================
  // Weight Unpack: SRAM 64-bit word → 8 × INT8 weight columns
  // =========================================================================
  genvar wi;
  generate
    for (wi = 0; wi < ARRAY_COLS; wi = wi + 1) begin : gen_wgt_unpack
      assign wgt_cols[wi] = sram_rd_valid ?
        $signed(sram_rd_data[wi*WGT_WIDTH +: WGT_WIDTH]) : {WGT_WIDTH{1'b0}};
    end
  endgenerate

  // =========================================================================
  // Note: iverilog 12 bridge generates removed — packed vectors used instead
  // =========================================================================

  // =========================================================================
  // Module Instantiations
  // =========================================================================

  // --- Layer Control FSM ---
  layer_ctrl_fsm u_fsm (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .abort_req             (abort_req),
    .busy              (busy),
    .done              (done),
    .layer_desc_in     (layer_desc_in),
    .layer_desc_valid  (layer_desc_valid),
    .layer_desc_ready  (layer_desc_ready),
    .sram_rd_en        (fsm_sram_rd_en),
    .sram_rd_addr      (fsm_sram_rd_addr),
    .sram_bank_swap    (fsm_sram_bank_swap),
    .wgt_load_start    (fsm_wgt_load_start),
    .wgt_load_done     (wgt_load_done),
    .act_input_enable  (fsm_act_enable),
    .act_input_accept  (fsm_act_accept),
    .act_input_flush   (fsm_act_flush),
    .array_enable      (fsm_array_enable),
    .array_clear_acc   (fsm_array_clear),
    .array_valid_reset (fsm_array_valid_reset),
    .array_data_valid  (fsm_array_data_valid_unused),
    .quant_enable      (fsm_quant_enable),
    .quant_relu_en     (fsm_quant_relu),
    .quant_shift       (fsm_quant_shift),
    .quant_zero_point  (fsm_quant_zp),
    .output_data_valid (fsm_out_valid_unused),
    .output_data_last  (fsm_out_last),
    .output_accepted   (out_accepted),
    .current_state     (current_state),
    .error_flag        (error_flag),
    .error_code        (error_code),
    .perf_counters     (perf_counters)
  );

  // --- SRAM Controller ---
  sram_controller u_sram (
    .clk            (clk),
    .rst_n          (rst_n),
    .bank_swap      (fsm_sram_bank_swap),
    .active_bank    (sram_active_bank),
    .rd_en          (fsm_sram_rd_en),
    .rd_addr        (fsm_sram_rd_addr),
    .rd_data        (sram_rd_data),
    .rd_valid       (sram_rd_valid),
    .wr_en          (wgt_wr_en),
    .wr_addr        (wgt_wr_addr),
    .wr_data        (wgt_wr_data),
    .bank_conflict  (sram_bank_conflict)
  );

  // --- AXI4-Stream Input with Diagonal Skew ---
  axi_stream_input u_axi_in (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (fsm_act_enable),
    .accept         (fsm_act_accept),
    .flush          (fsm_act_flush),
    .s_axis_tdata   (s_axis_tdata),
    .s_axis_tkeep   (s_axis_tkeep),
    .s_axis_tvalid  (s_axis_tvalid),
    .s_axis_tready  (s_axis_tready),
    .s_axis_tlast   (s_axis_tlast),
    .act_out_packed (skew_act_packed),
    .act_valid      (skew_act_valid),
    .frame_last     (skew_frame_last)
  );

  // --- 8×8 Systolic Array ---
  systolic_array u_array (
    .clk             (clk),
    .rst_n           (rst_n),
    .enable          (fsm_array_enable),
    .clear_acc       (fsm_array_clear),
    .valid_reset     (fsm_array_valid_reset),
    .act_in_packed   (skew_act_packed),
    .wgt_in          (wgt_cols),
    .data_valid_in   (sram_rd_valid),
    .data_valid_out  (array_data_valid_out),
    .acc_out_packed  (array_acc_packed),
    .any_overflow    (array_overflow)
  );

  // --- Quantization Unit ---
  quantize_unit u_quant (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (fsm_quant_enable),
    .relu_en        (fsm_quant_relu),
    .shift_amount   (fsm_quant_shift),
    .zero_point     (fsm_quant_zp),
    .acc_in_packed  (array_acc_packed),
    .acc_valid      (array_data_valid_out),
    .quant_out_packed (quant_data_packed),
    .quant_valid    (quant_valid),
    .any_saturated  (quant_saturated)
  );

  // --- AXI4-Stream Output ---
  // Output data_valid driven by quantize pipeline's actual validity,
  // not FSM state, to ensure real data is sent
  axi_stream_output u_axi_out (
    .clk            (clk),
    .rst_n          (rst_n),
    .data_in_packed (quant_data_packed),
    .data_valid     (quant_valid_pulse),
    .data_last      (fsm_out_last),
    .m_axis_tdata   (m_axis_tdata),
    .m_axis_tkeep   (m_axis_tkeep),
    .m_axis_tvalid  (m_axis_tvalid),
    .m_axis_tready  (m_axis_tready),
    .m_axis_tlast   (m_axis_tlast),
    .backpressure   (out_backpressure)
  );

  // =========================================================================
  // Output Handshake
  // =========================================================================
  // Track whether output was already sent (may happen during DRAIN, before
  // FSM reaches STORE_OUTPUT). Reset at start of each spatial position.
  logic output_was_sent;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      output_was_sent <= 1'b0;
    else if (fsm_array_clear)
      output_was_sent <= 1'b0;
    else if (m_axis_tvalid && m_axis_tready)
      output_was_sent <= 1'b1;
  end

  assign out_accepted = (m_axis_tvalid && m_axis_tready) || output_was_sent;

  // =========================================================================
  // Global Overflow Status
  // =========================================================================
  assign any_overflow = array_overflow;

  // =========================================================================
  // Debug Port Assignments
  // =========================================================================
  assign dbg_sram_active_bank  = sram_active_bank;
  assign dbg_sram_bank_conflict= sram_bank_conflict;
  assign dbg_sram_rd_valid     = sram_rd_valid;
  assign dbg_act_valid         = skew_act_valid;
  assign dbg_act_frame_last    = skew_frame_last;
  assign dbg_quant_valid       = quant_valid;
  assign dbg_quant_saturated   = quant_saturated;
  assign dbg_out_backpressure  = out_backpressure;
  assign dbg_wgt_load_start    = fsm_wgt_load_start;

endmodule
