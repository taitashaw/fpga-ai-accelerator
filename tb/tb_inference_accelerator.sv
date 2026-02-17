// ===========================================================================
// Module:  tb_inference_accelerator.sv
// Project: AI Inference Accelerator SoC — Verification
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   Self-checking testbench for the complete inference accelerator.
//   Drives the DUT through the full inference flow:
//     1. Load layer descriptor
//     2. DMA weights into SRAM
//     3. Stream activations via AXI4-Stream
//     4. Capture and check outputs against golden model
//
//   Features:
//     - File-based stimulus ($readmemh for packed data, $fscanf for config)
//     - Streaming scoreboard (compares each output beat)
//     - Portable coverage counters (no SVA/covergroups)
//     - Configurable via plusargs for regression automation
//
// Compatibility: iverilog 12.0 (no SVA, no covergroups, portable checks)
// ===========================================================================

`timescale 1ns / 1ps

module tb_inference_accelerator;

  import pkg_accelerator::*;

  // =========================================================================
  // Parameters
  // =========================================================================
  parameter CLK_PERIOD = 5.0;  // 200 MHz (5 ns period)
  parameter TIMEOUT_CYCLES = 200_000;
  parameter MAX_WEIGHTS = 1024;
  parameter MAX_ACTIVATIONS = 1024;
  parameter MAX_OUTPUTS = 1024;

  // =========================================================================
  // Testbench Signals
  // =========================================================================
  logic clk;
  logic rst_n;

  // Control
  logic start;
  logic abort_req;
  logic busy;
  logic done_out;

  // Layer descriptor
  layer_desc_t layer_desc_in;
  logic layer_desc_valid;
  logic layer_desc_ready;

  // AXI4-Stream Slave (activation input)
  logic [AXI_DATA_WIDTH-1:0] s_axis_tdata;
  logic [AXI_KEEP_WIDTH-1:0] s_axis_tkeep;
  logic s_axis_tvalid;
  logic s_axis_tready;
  logic s_axis_tlast;

  // AXI4-Stream Master (result output)
  logic [AXI_DATA_WIDTH-1:0] m_axis_tdata;
  logic [AXI_KEEP_WIDTH-1:0] m_axis_tkeep;
  logic m_axis_tvalid;
  logic m_axis_tready;
  logic m_axis_tlast;

  // Weight load port
  logic wgt_wr_en;
  logic [SRAM_ADDR_W-1:0] wgt_wr_addr;
  logic [SRAM_WIDTH-1:0] wgt_wr_data;
  logic wgt_load_done;

  // Status & Debug
  logic [3:0] current_state;
  logic error_flag;
  error_code_t error_code;
  logic any_overflow;
  logic [31:0] perf_counters [NUM_PERF_CTRS];
  logic dbg_sram_active_bank;
  logic dbg_sram_bank_conflict;
  logic dbg_sram_rd_valid;
  logic dbg_act_valid;
  logic dbg_act_frame_last;
  logic dbg_quant_valid;
  logic dbg_quant_saturated;
  logic dbg_out_backpressure;
  logic dbg_wgt_load_start;

  // =========================================================================
  // DUT Instantiation
  // =========================================================================
  inference_accelerator_top dut (
    .clk                  (clk),
    .rst_n                (rst_n),
    .start                (start),
    .abort_req            (abort_req),
    .busy                 (busy),
    .done                 (done_out),
    .layer_desc_in        (layer_desc_in),
    .layer_desc_valid     (layer_desc_valid),
    .layer_desc_ready     (layer_desc_ready),
    .s_axis_tdata         (s_axis_tdata),
    .s_axis_tkeep         (s_axis_tkeep),
    .s_axis_tvalid        (s_axis_tvalid),
    .s_axis_tready        (s_axis_tready),
    .s_axis_tlast         (s_axis_tlast),
    .m_axis_tdata         (m_axis_tdata),
    .m_axis_tkeep         (m_axis_tkeep),
    .m_axis_tvalid        (m_axis_tvalid),
    .m_axis_tready        (m_axis_tready),
    .m_axis_tlast         (m_axis_tlast),
    .wgt_wr_en            (wgt_wr_en),
    .wgt_wr_addr          (wgt_wr_addr),
    .wgt_wr_data          (wgt_wr_data),
    .wgt_load_done        (wgt_load_done),
    .current_state        (current_state),
    .error_flag           (error_flag),
    .error_code           (error_code),
    .any_overflow         (any_overflow),
    .perf_counters        (perf_counters),
    .dbg_sram_active_bank (dbg_sram_active_bank),
    .dbg_sram_bank_conflict(dbg_sram_bank_conflict),
    .dbg_sram_rd_valid    (dbg_sram_rd_valid),
    .dbg_act_valid        (dbg_act_valid),
    .dbg_act_frame_last   (dbg_act_frame_last),
    .dbg_quant_valid      (dbg_quant_valid),
    .dbg_quant_saturated  (dbg_quant_saturated),
    .dbg_out_backpressure (dbg_out_backpressure),
    .dbg_wgt_load_start   (dbg_wgt_load_start)
  );

  // =========================================================================
  // Clock Generation
  // =========================================================================
  initial clk = 0;
  always #(CLK_PERIOD / 2.0) clk = ~clk;

  // =========================================================================
  // Test Data Storage
  // =========================================================================
  // Layer config
  integer cfg_in_channels;
  integer cfg_out_channels;
  integer cfg_spatial_size;
  integer cfg_shift;
  integer cfg_zero_point;
  integer cfg_relu_en;

  // Packed weight data
  logic [63:0] weight_mem [0:MAX_WEIGHTS-1];
  integer num_weight_words;

  // Packed activation data
  logic [63:0] act_mem [0:MAX_ACTIVATIONS-1];
  integer num_act_words;

  // Expected outputs
  integer expected_outputs [0:MAX_OUTPUTS-1];
  integer num_expected;

  // Test directory path
  reg [256*8-1:0] test_dir;
  reg [256*8-1:0] filepath;

  // =========================================================================
  // Coverage Counters (portable — no covergroups)
  // =========================================================================
  integer cov_fsm_states [0:15];  // Count visits per FSM state
  integer cov_overflow_events;
  integer cov_saturation_events;
  integer cov_backpressure_events;
  integer cov_axi_in_transfers;   // valid && ready on input
  integer cov_axi_out_transfers;  // valid && ready on output
  integer cov_axi_in_idle;        // !valid && !ready
  integer cov_axi_in_backpressure;// valid && !ready
  integer cov_axi_in_stall;       // !valid && ready
  integer cov_bank_conflicts;

  // Track FSM state coverage
  always @(posedge clk) begin
    if (rst_n) begin
      if (current_state < 16)
        cov_fsm_states[current_state] = cov_fsm_states[current_state] + 1;
      if (any_overflow)
        cov_overflow_events = cov_overflow_events + 1;
      if (dbg_quant_saturated)
        cov_saturation_events = cov_saturation_events + 1;
      if (dbg_out_backpressure)
        cov_backpressure_events = cov_backpressure_events + 1;
      if (dbg_sram_bank_conflict)
        cov_bank_conflicts = cov_bank_conflicts + 1;

      // AXI input handshake coverage (4 combinations)
      if (s_axis_tvalid && s_axis_tready)
        cov_axi_in_transfers = cov_axi_in_transfers + 1;
      if (s_axis_tvalid && !s_axis_tready)
        cov_axi_in_backpressure = cov_axi_in_backpressure + 1;
      if (!s_axis_tvalid && s_axis_tready)
        cov_axi_in_stall = cov_axi_in_stall + 1;
      if (!s_axis_tvalid && !s_axis_tready)
        cov_axi_in_idle = cov_axi_in_idle + 1;

      // AXI output transfers
      if (m_axis_tvalid && m_axis_tready)
        cov_axi_out_transfers = cov_axi_out_transfers + 1;
    end
  end

  // =========================================================================
  // Scoreboard
  // =========================================================================
  integer output_count;
  integer mismatch_count;
  integer match_count;

  task automatic check_output;
    input [AXI_DATA_WIDTH-1:0] data;
    integer col;
    integer actual_val;
    integer expected_val;
    integer idx;
    begin
      for (col = 0; col < ARRAY_COLS; col = col + 1) begin
        idx = output_count * ARRAY_COLS + col;
        if (idx < num_expected) begin
          // Extract signed byte
          actual_val = $signed(data[col*8 +: 8]);
          expected_val = expected_outputs[idx];
          if (actual_val !== expected_val) begin
            $display("  MISMATCH [%0d]: got %0d, expected %0d (col=%0d, beat=%0d)",
                     idx, actual_val, expected_val, col, output_count);
            mismatch_count = mismatch_count + 1;
          end else begin
            match_count = match_count + 1;
          end
        end
      end
      output_count = output_count + 1;
    end
  endtask

  // =========================================================================
  // File Loading Tasks
  // =========================================================================
  task automatic load_config;
    input [256*8-1:0] dir;
    integer fd, r;
    reg [256*8-1:0] path;
    begin
      path = {dir, "/config.txt"};
      fd = $fopen(path, "r");
      if (fd == 0) begin
        $display("ERROR: Cannot open %0s", path);
        $finish;
      end
      r = $fscanf(fd, "%d\n", cfg_in_channels);
      r = $fscanf(fd, "%d\n", cfg_out_channels);
      r = $fscanf(fd, "%d\n", cfg_spatial_size);
      r = $fscanf(fd, "%d\n", cfg_shift);
      r = $fscanf(fd, "%d\n", cfg_zero_point);
      r = $fscanf(fd, "%d\n", cfg_relu_en);
      $fclose(fd);
      $display("  Config: IC=%0d OC=%0d Spatial=%0d Shift=%0d ZP=%0d ReLU=%0d",
               cfg_in_channels, cfg_out_channels, cfg_spatial_size,
               cfg_shift, cfg_zero_point, cfg_relu_en);
    end
  endtask

  task automatic load_weights;
    input [256*8-1:0] dir;
    reg [256*8-1:0] path;
    begin
      path = {dir, "/weights.hex"};
      $readmemh(path, weight_mem);
      // Count non-zero entries to determine size
      num_weight_words = (cfg_in_channels / ARRAY_COLS) *
                         (cfg_out_channels / ARRAY_ROWS) * ARRAY_ROWS;
      $display("  Loaded %0d weight words", num_weight_words);
    end
  endtask

  task automatic load_activations;
    input [256*8-1:0] dir;
    reg [256*8-1:0] path;
    begin
      path = {dir, "/activations.hex"};
      $readmemh(path, act_mem);
      num_act_words = (cfg_in_channels / ARRAY_COLS) * cfg_spatial_size *
                      (cfg_out_channels / ARRAY_ROWS);
      $display("  Loaded %0d activation words", num_act_words);
    end
  endtask

  task automatic load_expected;
    input [256*8-1:0] dir;
    integer fd, r, val, count;
    reg [256*8-1:0] path;
    begin
      path = {dir, "/expected.txt"};
      fd = $fopen(path, "r");
      if (fd == 0) begin
        $display("ERROR: Cannot open %0s", path);
        $finish;
      end
      count = 0;
      while (!$feof(fd)) begin
        r = $fscanf(fd, "%d\n", val);
        if (r == 1) begin
          expected_outputs[count] = val;
          count = count + 1;
        end
      end
      $fclose(fd);
      num_expected = count;
      $display("  Loaded %0d expected output values", num_expected);
    end
  endtask

  // =========================================================================
  // DMA Weight Loader
  // =========================================================================
  task automatic dma_load_weights;
    integer i;
    begin
      $display("  DMA: Loading %0d weight words into SRAM...", num_weight_words);
      wgt_load_done = 0;
      for (i = 0; i < num_weight_words; i = i + 1) begin
        @(posedge clk);
        #1;
        wgt_wr_en   = 1;
        wgt_wr_addr = i[SRAM_ADDR_W-1:0];
        wgt_wr_data = weight_mem[i];
      end
      @(posedge clk);
      #1;
      wgt_wr_en = 0;
      wgt_load_done = 1;
      @(posedge clk);
      #1;
      wgt_load_done = 0;
      $display("  DMA: Weight load complete");
    end
  endtask

  // =========================================================================
  // AXI4-Stream Activation Driver
  // =========================================================================
  task automatic stream_activations;
    integer i;
    begin
      $display("  AXI: Streaming %0d activation words...", num_act_words);
      for (i = 0; i < num_act_words; i = i + 1) begin
        @(posedge clk);
        #1;
        s_axis_tdata  = act_mem[i];
        s_axis_tkeep  = {AXI_KEEP_WIDTH{1'b1}};
        s_axis_tvalid = 1;
        s_axis_tlast  = (i == num_act_words - 1);
        // Wait for ready
        while (!s_axis_tready) begin
          @(posedge clk);
          #1;
        end
      end
      @(posedge clk);
      #1;
      s_axis_tvalid = 0;
      s_axis_tlast  = 0;
      $display("  AXI: Activation streaming complete");
    end
  endtask

  // =========================================================================
  // Output Capture (runs in parallel via fork)
  // =========================================================================
  integer capture_timeout;

  task automatic capture_outputs;
    begin
      output_count = 0;
      mismatch_count = 0;
      match_count = 0;
      capture_timeout = 0;
      m_axis_tready = 1; // Always ready to accept

      while (output_count * ARRAY_COLS < num_expected &&
             capture_timeout < TIMEOUT_CYCLES) begin
        @(posedge clk);
        #1;
        if (m_axis_tvalid && m_axis_tready) begin
          check_output(m_axis_tdata);
          capture_timeout = 0;
        end else begin
          capture_timeout = capture_timeout + 1;
        end
      end

      if (capture_timeout >= TIMEOUT_CYCLES)
        $display("  WARNING: Output capture timed out after %0d cycles", TIMEOUT_CYCLES);
    end
  endtask

  // =========================================================================
  // Initialize Coverage Counters
  // =========================================================================
  task automatic init_coverage;
    integer i;
    begin
      for (i = 0; i < 16; i = i + 1)
        cov_fsm_states[i] = 0;
      cov_overflow_events = 0;
      cov_saturation_events = 0;
      cov_backpressure_events = 0;
      cov_axi_in_transfers = 0;
      cov_axi_in_idle = 0;
      cov_axi_in_backpressure = 0;
      cov_axi_in_stall = 0;
      cov_axi_out_transfers = 0;
      cov_bank_conflicts = 0;
    end
  endtask

  // =========================================================================
  // Print Coverage Report
  // =========================================================================
  task automatic print_coverage;
    begin
      $display("");
      $display("=== COVERAGE REPORT ===");
      $display("  FSM State Visits:");
      $display("    IDLE=%0d LOAD_DESC=%0d LOAD_WEIGHTS=%0d LOAD_ACT=%0d",
               cov_fsm_states[0], cov_fsm_states[1],
               cov_fsm_states[2], cov_fsm_states[3]);
      $display("    COMPUTE=%0d DRAIN=%0d QUANT=%0d STORE_OUTPUT=%0d",
               cov_fsm_states[4], cov_fsm_states[5],
               cov_fsm_states[6], cov_fsm_states[7]);
      $display("    NEXT_TILE=%0d NEXT_LAYER=%0d DONE=%0d",
               cov_fsm_states[8], cov_fsm_states[9], cov_fsm_states[10]);
      $display("  Overflow events: %0d", cov_overflow_events);
      $display("  Saturation events: %0d", cov_saturation_events);
      $display("  Backpressure events: %0d", cov_backpressure_events);
      $display("  Bank conflicts: %0d", cov_bank_conflicts);
      $display("  AXI Input: transfers=%0d idle=%0d backpressure=%0d stall=%0d",
               cov_axi_in_transfers, cov_axi_in_idle,
               cov_axi_in_backpressure, cov_axi_in_stall);
      $display("  AXI Output transfers: %0d", cov_axi_out_transfers);
      $display("  Performance Counters:");
      $display("    Total cycles=%0d  MAC active=%0d  MAC stall=%0d",
               perf_counters[0], perf_counters[1], perf_counters[2]);
      $display("    AXI rd beats=%0d  AXI wr beats=%0d",
               perf_counters[3], perf_counters[4]);
      $display("    Layers done=%0d  Tiles done=%0d  Errors=%0d",
               perf_counters[5], perf_counters[6], perf_counters[7]);
      $display("=======================");
    end
  endtask

  // =========================================================================
  // Main Test Sequence
  // =========================================================================
  initial begin
    // Optional VCD dump
    `ifdef DUMP_VCD
      $dumpfile("sim/waveform.vcd");
      $dumpvars(0, tb_inference_accelerator);
    `endif

    // Get test directory from plusarg
    if (!$value$plusargs("test_dir=%s", test_dir)) begin
      $display("ERROR: No +test_dir=<path> specified");
      $finish;
    end

    // Initialize signals
    rst_n = 0;
    start = 0;
    abort_req = 0;
    layer_desc_valid = 0;
    s_axis_tdata = 0;
    s_axis_tkeep = 0;
    s_axis_tvalid = 0;
    s_axis_tlast = 0;
    m_axis_tready = 0;
    wgt_wr_en = 0;
    wgt_wr_addr = 0;
    wgt_wr_data = 0;
    wgt_load_done = 0;

    init_coverage();

    // Reset
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5) @(posedge clk);

    $display("");
    $display("========================================");
    $display("  Test: %0s", test_dir);
    $display("========================================");

    // Load test data from files
    $display("Loading test vectors...");
    load_config(test_dir);
    load_weights(test_dir);
    load_activations(test_dir);
    load_expected(test_dir);

    // Configure layer descriptor
    layer_desc_in.in_channels   = cfg_in_channels[15:0];
    layer_desc_in.out_channels  = cfg_out_channels[15:0];
    layer_desc_in.spatial_size  = cfg_spatial_size[15:0];
    layer_desc_in.shift_amount  = cfg_shift[4:0];
    layer_desc_in.zero_point    = cfg_zero_point[7:0];
    layer_desc_in.relu_enable   = cfg_relu_en[0];
    layer_desc_in.reserved      = 6'd0;

    // ---------------------------------------------------------------
    // Phase 1: Start FSM
    // ---------------------------------------------------------------
    $display("Starting inference...");
    @(posedge clk); #1;
    start = 1;
    @(posedge clk); #1;
    start = 0;

    // Wait for FSM to reach LOAD_DESC
    wait(current_state == FSM_LOAD_DESC);
    $display("  FSM: LOAD_DESC");

    // ---------------------------------------------------------------
    // Phase 2: Provide layer descriptor
    // ---------------------------------------------------------------
    @(posedge clk); #1;
    layer_desc_valid = 1;
    @(posedge clk); #1;
    layer_desc_valid = 0;

    // Wait for FSM to reach LOAD_WEIGHTS
    wait(current_state == FSM_LOAD_WEIGHTS);
    $display("  FSM: LOAD_WEIGHTS");

    // ---------------------------------------------------------------
    // Phase 3: DMA weight load
    // ---------------------------------------------------------------
    dma_load_weights();

    // Wait for FSM to enter COMPUTE (passes through LOAD_ACT quickly)
    wait(current_state == FSM_LOAD_ACT || current_state == FSM_COMPUTE);
    $display("  FSM: COMPUTE phase");

    // ---------------------------------------------------------------
    // Phase 4: Stream activations + capture outputs (parallel)
    // ---------------------------------------------------------------
    fork
      stream_activations();
      capture_outputs();
    join

    // ---------------------------------------------------------------
    // Phase 5: Wait for FSM completion
    // ---------------------------------------------------------------
    wait(done_out || error_flag);
    repeat(10) @(posedge clk);

    // ---------------------------------------------------------------
    // Report Results
    // ---------------------------------------------------------------
    $display("");
    $display("========================================");
    if (error_flag)
      $display("  FSM ERROR: code=%0d", error_code);

    if (mismatch_count == 0 && match_count > 0) begin
      $display("  RESULT: PASS (%0d/%0d outputs matched)",
               match_count, num_expected);
    end else if (match_count == 0 && mismatch_count == 0) begin
      $display("  RESULT: NO OUTPUT CAPTURED");
      $display("  (FSM may not have reached output stage)");
      $display("  Final state: %0d, busy=%b, done=%b, error=%b",
               current_state, busy, done_out, error_flag);
    end else begin
      $display("  RESULT: FAIL (%0d mismatches out of %0d)",
               mismatch_count, num_expected);
    end
    $display("========================================");

    print_coverage();

    $display("");
    if (mismatch_count == 0 && match_count > 0)
      $display("TEST PASSED");
    else
      $display("TEST FAILED");

    $finish;
  end

  // =========================================================================
  // Global Timeout
  // =========================================================================
  initial begin
    #(CLK_PERIOD * TIMEOUT_CYCLES);
    $display("GLOBAL TIMEOUT: Test exceeded %0d cycles", TIMEOUT_CYCLES);
    $display("  Final state: %0d, busy=%b, done=%b", current_state, busy, done_out);
    print_coverage();
    $display("TEST FAILED (timeout)");
    $finish;
  end

endmodule
