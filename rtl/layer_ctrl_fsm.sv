// ===========================================================================
// Module:  layer_ctrl_fsm.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   11-state FSM orchestrating the complete inference datapath. Manages
//   tiled execution across IC (input channel), OC (output channel), and
//   spatial dimensions. Includes:
//     - Layer descriptor loading and validation
//     - Weight SRAM bank management (ping-pong)
//     - Activation streaming control
//     - Pipeline drain timing
//     - Tile counter advancement (IC → OC → spatial → next layer)
//     - Timeout watchdog with error recovery
//     - 8 hardware performance counters
//
// Tiling Strategy (weight-stationary):
//   For each spatial tile:
//     For each OC tile (out_channels / ARRAY_ROWS):
//       For each IC tile (in_channels / ARRAY_COLS):
//         1. Load weights [ARRAY_ROWS × ARRAY_COLS] into SRAM
//         2. Stream activations [spatial × ARRAY_COLS] through array
//         3. Accumulate partial sums across IC tiles
//       4. Requantize accumulated results
//       5. Output quantized tile
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module layer_ctrl_fsm
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // External control
  input  logic                          start,          // Pulse to begin inference
  input  logic                          abort_req,          // Force return to IDLE
  output logic                          busy,           // FSM is active
  output logic                          done,           // All layers complete

  // Layer descriptor input
  input  layer_desc_t                   layer_desc_in,
  input  logic                          layer_desc_valid,
  output logic                          layer_desc_ready,

  // SRAM control
  output logic                          sram_rd_en,
  output logic [SRAM_ADDR_W-1:0]        sram_rd_addr,
  output logic                          sram_bank_swap,

  // Weight load control (to external DMA)
  output logic                          wgt_load_start,
  input  logic                          wgt_load_done,

  // Activation input control
  output logic                          act_input_enable,
  output logic                          act_input_accept,  // Gate AXI tready (no new data during DRAIN)
  output logic                          act_input_flush,

  // Systolic array control
  output logic                          array_enable,
  output logic                          array_clear_acc,
  output logic                          array_valid_reset, // Reset valid counter each tile
  output logic                          array_data_valid,

  // Quantize control
  output logic                          quant_enable,
  output logic                          quant_relu_en,
  output logic [4:0]                    quant_shift,
  output logic signed [OUT_WIDTH-1:0]   quant_zero_point,

  // Output control
  output logic                          output_data_valid,
  output logic                          output_data_last,
  input  logic                          output_accepted,  // Output handshake complete

  // Status
  output logic [3:0]                    current_state,
  output logic                          error_flag,
  output error_code_t                   error_code,

  // Performance counters (active-low reset to zero)
  output logic [31:0]                   perf_counters [NUM_PERF_CTRS]
);

  // -------------------------------------------------------------------------
  // FSM State Register
  // -------------------------------------------------------------------------
  fsm_state_t state_reg, state_next;

  // -------------------------------------------------------------------------
  // Layer Descriptor Register
  // -------------------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  layer_desc_t layer_desc;  // Some fields (spatial_size, reserved) used in future phases
  /* verilator lint_on UNUSEDSIGNAL */
  logic        desc_loaded;

  // -------------------------------------------------------------------------
  // Tile Counters
  // -------------------------------------------------------------------------
  logic [15:0] ic_tile_idx;     // Current IC tile index
  logic [15:0] oc_tile_idx;     // Current OC tile index
  /* verilator lint_off UNUSEDSIGNAL */
  logic [15:0] spatial_idx;     // Reserved for multi-spatial-tile execution
  /* verilator lint_on UNUSEDSIGNAL */
  logic [15:0] ic_tiles_total;  // Total IC tiles = in_channels / TILE_COLS
  logic [15:0] oc_tiles_total;  // Total OC tiles = out_channels / TILE_ROWS
  logic        last_ic_tile;    // This is the last IC tile for current OC tile
  logic        last_oc_tile;    // This is the last OC tile
  /* verilator lint_off UNUSEDSIGNAL */
  logic        last_spatial;    // Reserved for multi-spatial-tile mode
  /* verilator lint_on UNUSEDSIGNAL */

  assign ic_tiles_total = layer_desc.in_channels >> $clog2(TILE_COLS);  // /8
  assign oc_tiles_total = layer_desc.out_channels >> $clog2(TILE_ROWS); // /8
  assign last_ic_tile   = (ic_tile_idx == ic_tiles_total - 1);
  assign last_oc_tile   = (oc_tile_idx == oc_tiles_total - 1);
  assign last_spatial   = (spatial_idx >= (layer_desc.spatial_size - 16'd1));

  // -------------------------------------------------------------------------
  // Pipeline Drain Counter
  // -------------------------------------------------------------------------
  // After last activation enters, wait ARRAY_LATENCY cycles for results
  logic [$clog2(TOTAL_LATENCY+1):0] drain_cnt;
  logic                              drain_done;

  assign drain_done = (drain_cnt >= TOTAL_LATENCY[$clog2(TOTAL_LATENCY+1):0]);

  // -------------------------------------------------------------------------
  // Weight Address Counter
  // -------------------------------------------------------------------------
  logic [SRAM_ADDR_W-1:0] wgt_addr_cnt;
  logic                    wgt_addr_done;
  // SRAM has 2-cycle read latency, so keep reading for ARRAY_ROWS cycles
  // but stay in COMPUTE for ARRAY_ROWS + 2 to let last read complete
  localparam COMPUTE_CYCLES = ARRAY_ROWS + 2;
  logic [$clog2(COMPUTE_CYCLES+1):0] compute_cnt;
  logic                               compute_done;

  assign wgt_addr_done = (wgt_addr_cnt >= ARRAY_ROWS[SRAM_ADDR_W-1:0]);
  assign compute_done  = (compute_cnt >= COMPUTE_CYCLES[$clog2(COMPUTE_CYCLES+1):0]);

  // -------------------------------------------------------------------------
  // Watchdog Timer
  // -------------------------------------------------------------------------
  logic [31:0] watchdog_cnt;
  logic        watchdog_timeout;

  assign watchdog_timeout = (watchdog_cnt >= WATCHDOG_CYCLES);

  // -------------------------------------------------------------------------
  // Error Register
  // -------------------------------------------------------------------------
  logic        err_flag_reg;
  error_code_t err_code_reg;

  assign error_flag = err_flag_reg;
  assign error_code = err_code_reg;

  // -------------------------------------------------------------------------
  // FSM Next-State Logic
  // -------------------------------------------------------------------------
  always_comb begin
    state_next = state_reg;

    case (state_reg)
      FSM_IDLE: begin
        if (start)
          state_next = FSM_LOAD_DESC;
      end

      FSM_LOAD_DESC: begin
        if (layer_desc_valid)
          state_next = FSM_LOAD_WEIGHTS;
        else if (watchdog_timeout)
          state_next = FSM_IDLE; // Timeout → error → idle
      end

      FSM_LOAD_WEIGHTS: begin
        if (wgt_load_done)
          state_next = FSM_LOAD_ACT;
        else if (watchdog_timeout)
          state_next = FSM_IDLE;
      end

      FSM_LOAD_ACT: begin
        // Transition to compute after activations start streaming
        // (In practice, LOAD_ACT and COMPUTE overlap; simplified here)
        state_next = FSM_COMPUTE;
      end

      FSM_COMPUTE: begin
        if (compute_done)
          state_next = FSM_DRAIN;
        else if (watchdog_timeout)
          state_next = FSM_IDLE;
      end

      FSM_DRAIN: begin
        if (drain_done) begin
          if (last_ic_tile)
            state_next = FSM_QUANT;
          else
            state_next = FSM_NEXT_TILE;
        end
      end

      FSM_QUANT: begin
        // Quantization takes QUANT_LATENCY cycles (2)
        state_next = FSM_STORE_OUTPUT;
      end

      FSM_STORE_OUTPUT: begin
        if (output_accepted) begin
          if (last_spatial && last_ic_tile)
            state_next = FSM_NEXT_TILE;
          else if (!last_spatial)
            state_next = FSM_LOAD_ACT;  // Next spatial position
          else
            state_next = FSM_NEXT_TILE;
        end else if (watchdog_timeout)
          state_next = FSM_IDLE;
      end

      FSM_NEXT_TILE: begin
        if (last_oc_tile && last_ic_tile)
          state_next = FSM_NEXT_LAYER;
        else
          // All weight tiles already in SRAM — just adjust base address
          // and proceed directly to next activation load
          state_next = FSM_LOAD_ACT;
      end

      FSM_NEXT_LAYER: begin
        state_next = FSM_DONE; // Single-layer simplified
      end

      FSM_DONE: begin
        // Stay in DONE until external reset or new start
        state_next = FSM_DONE;
      end

      default: begin
        state_next = FSM_IDLE;
      end
    endcase

    // Abort overrides everything
    if (abort_req)
      state_next = FSM_IDLE;
  end

  // -------------------------------------------------------------------------
  // FSM State Register & Datapath Control
  // -------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_reg       <= FSM_IDLE;
      desc_loaded     <= 1'b0;
      ic_tile_idx     <= '0;
      oc_tile_idx     <= '0;
      spatial_idx     <= '0;
      drain_cnt       <= '0;
      wgt_addr_cnt    <= '0;
      compute_cnt     <= '0;
      watchdog_cnt    <= '0;
      err_flag_reg    <= 1'b0;
      err_code_reg    <= ERR_NONE;
    end else begin
      state_reg <= state_next;

      // Watchdog: reset on state transition, count otherwise
      if (state_reg != state_next)
        watchdog_cnt <= '0;
      else
        watchdog_cnt <= watchdog_cnt + 1;

      // Error detection
      if (watchdog_timeout && state_reg != FSM_IDLE && state_reg != FSM_DONE) begin
        err_flag_reg <= 1'b1;
        err_code_reg <= ERR_TIMEOUT;
      end

      case (state_reg)
        FSM_IDLE: begin
          if (start) begin
            ic_tile_idx  <= '0;
            oc_tile_idx  <= '0;
            spatial_idx  <= '0;
            drain_cnt    <= '0;
            wgt_addr_cnt <= '0;
            compute_cnt  <= '0;
            err_flag_reg <= 1'b0;
            err_code_reg <= ERR_NONE;
          end
        end

        FSM_LOAD_DESC: begin
          if (layer_desc_valid) begin
            layer_desc  <= layer_desc_in;
            desc_loaded <= 1'b1;
          end
        end

        FSM_LOAD_WEIGHTS: begin
          // Reset weight address counter for new tile
          if (state_next == FSM_LOAD_ACT)
            wgt_addr_cnt <= '0;
        end

        FSM_LOAD_ACT: begin
          // Pre-increment address counter (first SRAM read issued this cycle)
          if (!wgt_addr_done)
            wgt_addr_cnt <= wgt_addr_cnt + 1;
        end

        FSM_COMPUTE: begin
          // Increment SRAM read address (stop at ARRAY_ROWS)
          if (!wgt_addr_done)
            wgt_addr_cnt <= wgt_addr_cnt + 1;
          // Increment compute cycle counter (includes SRAM latency)
          compute_cnt <= compute_cnt + 1;
        end

        FSM_DRAIN: begin
          drain_cnt <= drain_cnt + 1;
        end

        FSM_QUANT: begin
          drain_cnt <= '0;
        end

        FSM_STORE_OUTPUT: begin
          // When looping back for next spatial position
          if (output_accepted && !last_spatial) begin
            wgt_addr_cnt <= '0;
            compute_cnt  <= '0;
            drain_cnt    <= '0;
            spatial_idx  <= spatial_idx + 16'd1;
            ic_tile_idx  <= '0;  // Reset IC tile for new spatial position
          end
        end

        FSM_NEXT_TILE: begin
          wgt_addr_cnt <= '0;
          compute_cnt  <= '0;
          drain_cnt    <= '0;
          if (!last_ic_tile) begin
            ic_tile_idx <= ic_tile_idx + 1;
          end else begin
            ic_tile_idx <= '0;
            spatial_idx <= '0;  // Reset spatial for new OC tile
            if (!last_oc_tile)
              oc_tile_idx <= oc_tile_idx + 1;
          end
        end

        FSM_NEXT_LAYER: begin
          oc_tile_idx <= '0;
          ic_tile_idx <= '0;
        end

        default: ;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Output Control Signals
  // -------------------------------------------------------------------------
  assign current_state     = state_reg;
  assign busy              = (state_reg != FSM_IDLE) && (state_reg != FSM_DONE);
  assign done              = (state_reg == FSM_DONE);
  assign layer_desc_ready  = (state_reg == FSM_LOAD_DESC);

  // Pre-read SRAM during LOAD_ACT to compensate for 2-cycle read latency,
  // ensuring first weight arrives at same cycle as first activation
  // -------------------------------------------------------------------------
  // SRAM Base Address for Tiled Weight Access
  // -------------------------------------------------------------------------
  // All weight tiles are pre-loaded into SRAM by DMA at the start.
  // Each tile occupies ARRAY_ROWS words. Layout:
  //   [oc_tile=0,ic_tile=0] [oc_tile=0,ic_tile=1] [oc_tile=1,ic_tile=0] ...
  // Base address = (oc_tile * n_ic_tiles + ic_tile) * ARRAY_ROWS
  logic [SRAM_ADDR_W-1:0] sram_rd_base;

  // Tile base address: (oc_tile × n_ic_tiles + ic_tile) × ARRAY_ROWS
  // Truncation to SRAM_ADDR_W is intentional — overflow = config error
  assign sram_rd_base = SRAM_ADDR_W'(oc_tile_idx * ic_tiles_total + ic_tile_idx)
                        * SRAM_ADDR_W'(ARRAY_ROWS);

  assign sram_rd_en        = ((state_reg == FSM_COMPUTE) || (state_reg == FSM_LOAD_ACT)) &&
                             !wgt_addr_done;
  assign sram_rd_addr      = sram_rd_base + wgt_addr_cnt;
  // Swap banks: after weight load completes (so compute reads new data)
  // AND after each full OC tile with new weights
  assign sram_bank_swap    = (state_reg == FSM_LOAD_WEIGHTS && state_next == FSM_LOAD_ACT);

  assign wgt_load_start    = (state_reg == FSM_LOAD_WEIGHTS) && (state_next == FSM_LOAD_WEIGHTS);
  assign act_input_enable  = (state_reg == FSM_COMPUTE) || (state_reg == FSM_LOAD_ACT) ||
                             (state_reg == FSM_DRAIN);
  assign act_input_accept  = (state_reg == FSM_COMPUTE);
  assign act_input_flush   = (state_reg == FSM_IDLE) || (state_reg == FSM_NEXT_TILE) ||
                             (state_reg == FSM_STORE_OUTPUT);

  assign array_enable      = (state_reg == FSM_COMPUTE) || (state_reg == FSM_DRAIN);
  assign array_clear_acc   = ((state_reg == FSM_LOAD_WEIGHTS) && (ic_tile_idx == 0)) ||
                             ((state_reg == FSM_LOAD_ACT) && (ic_tile_idx == 0));
  assign array_valid_reset = (state_reg == FSM_LOAD_ACT); // Reset valid counter every tile
  assign array_data_valid  = (state_reg == FSM_COMPUTE); // Valid while feeding data

  assign quant_enable      = ((state_reg == FSM_DRAIN) && last_ic_tile) ||
                             (state_reg == FSM_QUANT) ||
                             (state_reg == FSM_STORE_OUTPUT);
  assign quant_relu_en     = desc_loaded ? layer_desc.relu_enable : 1'b0;
  assign quant_shift       = desc_loaded ? layer_desc.shift_amount : 5'd0;
  assign quant_zero_point  = desc_loaded ? layer_desc.zero_point : 8'sd0;

  // Output valid driven by actual quant pipeline, not FSM state
  assign output_data_valid = (state_reg == FSM_STORE_OUTPUT) ||
                             (state_reg == FSM_DRAIN) ||
                             (state_reg == FSM_QUANT);
  assign output_data_last  = (state_reg == FSM_STORE_OUTPUT) && last_oc_tile && last_ic_tile && last_spatial;

  // -------------------------------------------------------------------------
  // Performance Counters (8 × 32-bit)
  // -------------------------------------------------------------------------
  // Use intermediate registers, assign to output array via generate
  logic [31:0] perf_ctr_reg [NUM_PERF_CTRS];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      perf_ctr_reg[0] <= '0;  // TOTAL_CYCLES
      perf_ctr_reg[1] <= '0;  // MAC_ACTIVE
      perf_ctr_reg[2] <= '0;  // MAC_STALL
      perf_ctr_reg[3] <= '0;  // AXI_RD_BEATS
      perf_ctr_reg[4] <= '0;  // AXI_WR_BEATS
      perf_ctr_reg[5] <= '0;  // LAYERS_DONE
      perf_ctr_reg[6] <= '0;  // TILES_DONE
      perf_ctr_reg[7] <= '0;  // ERRORS
    end else if (state_reg == FSM_IDLE && start) begin
      // Clear counters on new inference start
      perf_ctr_reg[0] <= '0;
      perf_ctr_reg[1] <= '0;
      perf_ctr_reg[2] <= '0;
      perf_ctr_reg[3] <= '0;
      perf_ctr_reg[4] <= '0;
      perf_ctr_reg[5] <= '0;
      perf_ctr_reg[6] <= '0;
      perf_ctr_reg[7] <= '0;
    end else if (busy) begin
      // Total cycles
      perf_ctr_reg[0] <= perf_ctr_reg[0] + 1;

      // MAC active
      if (state_reg == FSM_COMPUTE)
        perf_ctr_reg[1] <= perf_ctr_reg[1] + 1;

      // MAC stall (backpressure during compute)
      if (state_reg == FSM_STORE_OUTPUT && !output_accepted)
        perf_ctr_reg[2] <= perf_ctr_reg[2] + 1;

      // AXI read beats (activation input)
      if (act_input_enable)
        perf_ctr_reg[3] <= perf_ctr_reg[3] + 1;

      // AXI write beats (output)
      if (state_reg == FSM_STORE_OUTPUT && output_accepted)
        perf_ctr_reg[4] <= perf_ctr_reg[4] + 1;

      // Layers done
      if (state_reg == FSM_NEXT_LAYER)
        perf_ctr_reg[5] <= perf_ctr_reg[5] + 1;

      // Tiles done
      if (state_reg == FSM_NEXT_TILE)
        perf_ctr_reg[6] <= perf_ctr_reg[6] + 1;

      // Errors
      if (watchdog_timeout)
        perf_ctr_reg[7] <= perf_ctr_reg[7] + 1;
    end
  end

  // Output array assignment (generate for iverilog unpacked array compat)
  genvar ci;
  generate
    for (ci = 0; ci < NUM_PERF_CTRS; ci = ci + 1) begin : gen_perf_out
      assign perf_counters[ci] = perf_ctr_reg[ci];
    end
  endgenerate

endmodule
