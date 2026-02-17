// ===========================================================================
// Module:  systolic_array.sv
// Project: AI Inference Accelerator SoC
// Author:  John Bagshaw, Senior FPGA Design Engineer
// Date:    February 2026
//
// Description:
//   8×8 weight-stationary systolic array with 64 MAC processing elements.
//   Data flow:
//     - Activations enter from the west (left) boundary, flow east
//     - Weights enter from the north (top) boundary, flow south
//     - Accumulator outputs are read from the bottom row
//
//   A valid counter tracks pipeline priming latency. The array produces
//   valid outputs starting ARRAY_COLS + MAC_LATENCY cycles after the
//   first valid activation enters.
//
//   All inter-PE connections are registered (1-cycle hop), creating the
//   systolic timing that avoids global fanout.
//
// Compatibility: iverilog 12.0, Verilator 5.x, VCS, QuestaSim
// ===========================================================================

`timescale 1ns / 1ps

module systolic_array
  import pkg_accelerator::*;
(
  input  logic                          clk,
  input  logic                          rst_n,

  // Control
  input  logic                          enable,      // Global enable (stall when low)
  input  logic                          clear_acc,   // Clear all PE accumulators
  input  logic                          valid_reset, // Reset pipeline valid counter (start of each tile)

  // West boundary: activation inputs (packed vector, one element per row)
  // iverilog 12 fix: packed to avoid unpacked array port connection bug
  input  logic [ACT_WIDTH*ARRAY_ROWS-1:0] act_in_packed,

  // North boundary: weight inputs (one per column)
  input  logic signed [WGT_WIDTH-1:0]   wgt_in  [ARRAY_COLS],

  // Validity tracking
  input  logic                          data_valid_in,  // First valid data entering
  output logic                          data_valid_out, // Valid data at output

  // Bottom boundary: accumulator outputs (packed vector, one per column from bottom row)
  // iverilog 12 fix: packed to avoid unpacked array port connection bug
  output logic [ACC_WIDTH*ARRAY_COLS-1:0] acc_out_packed,

  // Status
  output logic                          any_overflow
);

  // -------------------------------------------------------------------------
  // Inter-PE wiring
  // -------------------------------------------------------------------------
  // Horizontal (activation) wires: [row][col+1] — includes west boundary
  logic signed [ACT_WIDTH-1:0] act_wire [ARRAY_ROWS][ARRAY_COLS+1];

  // South-flow weight wires removed — broadcast mode feeds weights directly
  // PE wgt_out ports connected to dummy wires (not used in broadcast mode)
  logic signed [WGT_WIDTH-1:0] wgt_out_unused [ARRAY_ROWS][ARRAY_COLS]; /* verilator lint_off UNUSEDSIGNAL */

  // Broadcast weight wires: skewed weights from north boundary to all rows
  logic signed [WGT_WIDTH-1:0] wgt_bcast [ARRAY_ROWS][ARRAY_COLS];

  // PE accumulator and overflow outputs
  logic signed [ACC_WIDTH-1:0] pe_acc   [ARRAY_ROWS][ARRAY_COLS];
  logic                        pe_ovf   [ARRAY_ROWS][ARRAY_COLS];

  // -------------------------------------------------------------------------
  // Boundary connections
  // -------------------------------------------------------------------------
  // West boundary: feed activations into column 0 of each row
  // Unpack from packed vector (iverilog 12 compatibility)
  genvar bi;
  generate
    for (bi = 0; bi < ARRAY_ROWS; bi = bi + 1) begin : gen_west_boundary
      assign act_wire[bi][0] = $signed(act_in_packed[bi*ACT_WIDTH +: ACT_WIDTH]);
    end
  endgenerate

  // North boundary: broadcast weights directly to ALL rows (no south flow)
  // Each SRAM read provides weight row k simultaneously to all PEs.
  // With activation skew, PE[r,c] sees a_r at time (r+c) and W[r+c, c] from SRAM.
  // Weight column skew of c cycles ensures W[k,c] enters column c at time (k+c),
  // so PE[r,c] sees W[(r+c)-c, c] = W[r, c] when act a_r arrives. ✓
  //
  // Implementation: broadcast wgt_in to all rows (bypass PE south flow),
  // with per-column skew registers at the north boundary.

  // Column 0: direct broadcast to all rows
  genvar bj, br;
  generate
    for (br = 0; br < ARRAY_ROWS; br = br + 1) begin : gen_wgt_bcast_c0
      assign wgt_bcast[br][0] = wgt_in[0];
    end

    for (bj = 1; bj < ARRAY_COLS; bj = bj + 1) begin : gen_north_skew
      // Shift register of depth bj for column bj
      logic signed [WGT_WIDTH-1:0] wgt_skew [0:bj-1];

      // Stage 0: input from wgt_in
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          wgt_skew[0] <= '0;
        else if (clear_acc)
          wgt_skew[0] <= '0;
        else if (enable)
          wgt_skew[0] <= wgt_in[bj];
      end

      // Subsequent stages
      genvar ws;
      for (ws = 1; ws < bj; ws = ws + 1) begin : gen_wgt_stage
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            wgt_skew[ws] <= '0;
          else if (clear_acc)
            wgt_skew[ws] <= '0;
          else if (enable)
            wgt_skew[ws] <= wgt_skew[ws-1];
        end
      end

      // Broadcast skewed weight to ALL rows of this column
      for (br = 0; br < ARRAY_ROWS; br = br + 1) begin : gen_wgt_bcast
        assign wgt_bcast[br][bj] = wgt_skew[bj-1];
      end
    end
  endgenerate

  // Weight broadcast connections: each PE receives skewed weight from north boundary
  // from PE outputs are overridden by the broadcast assignments above.

  // -------------------------------------------------------------------------
  // PE Grid Instantiation (8×8 = 64 PEs)
  // -------------------------------------------------------------------------
  genvar gi, gj;
  generate
    for (gi = 0; gi < ARRAY_ROWS; gi = gi + 1) begin : gen_row
      for (gj = 0; gj < ARRAY_COLS; gj = gj + 1) begin : gen_col

        mac_unit u_pe (
          .clk       (clk),
          .rst_n     (rst_n),
          .clear_acc (clear_acc),
          .enable    (enable),
          .act_in    (act_wire[gi][gj]),        // From west neighbor
          .wgt_in    (wgt_bcast[gi][gj]),       // From broadcast (skewed north boundary)
          .act_out   (act_wire[gi][gj+1]),      // To east neighbor
          .wgt_out   (wgt_out_unused[gi][gj]),
          .acc_out   (pe_acc[gi][gj]),
          .overflow  (pe_ovf[gi][gj])
        );

      end
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Output: Bottom row accumulators
  // -------------------------------------------------------------------------
  // Output: Column-sum accumulators — sum all PE[r][c] for each column c
  // Each PE[r,c] accumulates act[r] × matching wgt for one input channel.
  // The column sum gives the full dot product across all input channels.
  genvar oi;
  generate
    for (oi = 0; oi < ARRAY_COLS; oi = oi + 1) begin : gen_acc_out
      logic signed [ACC_WIDTH-1:0] col_sum;
      integer ri;
      always_comb begin
        col_sum = '0;
        for (ri = 0; ri < ARRAY_ROWS; ri = ri + 1)
          col_sum = col_sum + pe_acc[ri][oi];
      end
      assign acc_out_packed[oi*ACC_WIDTH +: ACC_WIDTH] = col_sum;
    end
  endgenerate

  // -------------------------------------------------------------------------
  // Overflow aggregation: OR all PE overflow flags
  // -------------------------------------------------------------------------
  logic [ARRAY_ROWS*ARRAY_COLS-1:0] ovf_flat;

  genvar fi, fj;
  generate
    for (fi = 0; fi < ARRAY_ROWS; fi = fi + 1) begin : gen_ovf_row
      for (fj = 0; fj < ARRAY_COLS; fj = fj + 1) begin : gen_ovf_col
        assign ovf_flat[fi*ARRAY_COLS + fj] = pe_ovf[fi][fj];
      end
    end
  endgenerate

  assign any_overflow = |ovf_flat;

  // -------------------------------------------------------------------------
  // Valid Counter
  // -------------------------------------------------------------------------
  // Tracks how many cycles since first valid data entered.
  // Output becomes valid after PRIME_CYCLES of priming.
  // Accounts for: activation east propagation (ARRAY_COLS hops) +
  //               weight column skew (ARRAY_COLS-1 stages) +
  //               MAC pipeline (MAC_LATENCY stages)
  localparam PRIME_CYCLES = ARRAY_COLS + (ARRAY_COLS - 1) + MAC_LATENCY; // 8+7+3=18

  logic [$clog2(PRIME_CYCLES+1):0] valid_cnt;
  logic                            primed;
  logic                            counting; // Set on first data_valid_in

  assign primed = (valid_cnt >= PRIME_CYCLES[($clog2(PRIME_CYCLES+1)):0]);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_cnt <= '0;
      counting  <= 1'b0;
    end else if (clear_acc || valid_reset) begin
      valid_cnt <= '0;
      counting  <= 1'b0;
    end else if (enable) begin
      if (data_valid_in && !counting)
        counting <= 1'b1;
      if ((counting || data_valid_in) && !primed)
        valid_cnt <= valid_cnt + 1'b1;
    end
  end

  assign data_valid_out = primed && enable;

endmodule
