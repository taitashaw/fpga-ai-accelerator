#!/usr/bin/env python3
"""
Golden Model for AI Inference Accelerator
==========================================
Bit-accurate INT8 matmul with INT32 accumulation, optional ReLU,
arithmetic right-shift requantization, and INT8 saturation.

This model produces the exact expected outputs that the RTL must match.
All arithmetic uses Python integers (arbitrary precision) with explicit
clamping to match hardware behavior.

Author: John Bagshaw, Senior FPGA Design Engineer
Date:   February 2026
"""

import numpy as np
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Hardware parameters (must match pkg_accelerator.sv)
# ---------------------------------------------------------------------------
ARRAY_ROWS = 8
ARRAY_COLS = 8
ACT_WIDTH  = 8   # signed
WGT_WIDTH  = 8   # signed
ACC_WIDTH  = 32   # signed
OUT_WIDTH  = 8   # signed

ACT_MIN, ACT_MAX = -128, 127
WGT_MIN, WGT_MAX = -128, 127
ACC_MIN = -(1 << (ACC_WIDTH - 1))
ACC_MAX = (1 << (ACC_WIDTH - 1)) - 1
OUT_MIN, OUT_MAX = -128, 127


# ---------------------------------------------------------------------------
# Core arithmetic functions (bit-accurate to RTL)
# ---------------------------------------------------------------------------
def saturate_int8(val: int) -> int:
    """Saturate a 32-bit signed value to INT8 range [-128, +127]."""
    if val > OUT_MAX:
        return OUT_MAX
    elif val < OUT_MIN:
        return OUT_MIN
    return int(val)


def relu(val: int) -> int:
    """ReLU: clamp negative values to zero."""
    return max(0, val)


def arithmetic_right_shift(val: int, shift: int) -> int:
    """
    Arithmetic right-shift matching SystemVerilog >>> behavior.
    Python's >> on signed integers already does arithmetic shift.
    We need to handle the 32-bit signed interpretation.
    """
    # Ensure val is in 32-bit signed range
    val = val & 0xFFFFFFFF
    if val & 0x80000000:
        val = val - 0x100000000  # Convert to signed
    return val >> shift


def requantize(acc_val: int, relu_en: bool, shift: int, zero_point: int) -> int:
    """
    Full requantization pipeline matching RTL quantize_unit.sv:
      Stage 1: Optional ReLU
      Stage 2: Arithmetic right-shift → zero-point add → saturate
    """
    # Stage 1: Optional ReLU
    if relu_en:
        stage1 = relu(acc_val)
    else:
        stage1 = acc_val

    # Stage 2: Shift, add zero-point, saturate
    shifted = arithmetic_right_shift(stage1, shift)
    with_zp = shifted + zero_point
    return saturate_int8(with_zp)


# ---------------------------------------------------------------------------
# Matrix multiply (bit-accurate tiled execution)
# ---------------------------------------------------------------------------
def matmul_int8(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    INT8 matrix multiply with INT32 accumulation.

    Args:
        activations: shape (M, K) dtype=int8
        weights:     shape (K, N) dtype=int8

    Returns:
        result: shape (M, N) dtype=int32 (accumulated, not yet quantized)

    This uses Python integers to avoid NumPy overflow behavior.
    """
    M, K = activations.shape
    K2, N = weights.shape
    assert K == K2, f"Dimension mismatch: act({M},{K}) x wgt({K2},{N})"

    result = np.zeros((M, N), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                # Signed 8×8 → 16-bit product, accumulated in 32-bit
                product = int(activations[m, k]) * int(weights[k, n])
                acc += product
                # Clamp to 32-bit signed range (overflow detection)
                acc = max(ACC_MIN, min(ACC_MAX, acc))
            result[m, n] = acc
    return result


def inference_layer(activations: np.ndarray, weights: np.ndarray,
                    relu_en: bool, shift: int, zero_point: int) -> np.ndarray:
    """
    Complete inference layer: matmul → requantize.

    Args:
        activations: (M, K) int8
        weights:     (K, N) int8
        relu_en:     Enable ReLU before requantization
        shift:       Right-shift amount (0-31)
        zero_point:  Output zero-point offset (int8)

    Returns:
        output: (M, N) int8 (requantized)
    """
    # Step 1: Matrix multiply with INT32 accumulation
    acc = matmul_int8(activations, weights)

    # Step 2: Requantize each element
    M, N = acc.shape
    output = np.zeros((M, N), dtype=np.int8)
    for m in range(M):
        for n in range(N):
            output[m, n] = requantize(int(acc[m, n]), relu_en, shift, zero_point)

    return output


# ---------------------------------------------------------------------------
# Tiled execution (matches RTL FSM tiling)
# ---------------------------------------------------------------------------
def inference_layer_tiled(activations: np.ndarray, weights: np.ndarray,
                          relu_en: bool, shift: int, zero_point: int) -> np.ndarray:
    """
    Tiled inference matching how the RTL FSM processes data:
      For each OC tile (output channels in groups of ARRAY_ROWS):
        For each IC tile (input channels in groups of ARRAY_COLS):
          Compute partial matmul, accumulate across IC tiles
        Requantize accumulated results

    This ensures the golden model exercises the same accumulation
    pattern as the hardware.
    """
    M, K = activations.shape
    K2, N = weights.shape
    assert K == K2

    # Pad dimensions to tile multiples if needed
    K_padded = ((K + ARRAY_COLS - 1) // ARRAY_COLS) * ARRAY_COLS
    N_padded = ((N + ARRAY_ROWS - 1) // ARRAY_ROWS) * ARRAY_ROWS

    act_padded = np.zeros((M, K_padded), dtype=np.int8)
    act_padded[:, :K] = activations

    wgt_padded = np.zeros((K_padded, N_padded), dtype=np.int8)
    wgt_padded[:K, :N] = weights

    # Accumulation buffer
    acc = np.zeros((M, N_padded), dtype=np.int32)

    # Tile loop matching FSM
    n_ic_tiles = K_padded // ARRAY_COLS
    n_oc_tiles = N_padded // ARRAY_ROWS

    for oc_tile in range(n_oc_tiles):
        oc_start = oc_tile * ARRAY_ROWS
        oc_end = oc_start + ARRAY_ROWS

        for ic_tile in range(n_ic_tiles):
            ic_start = ic_tile * ARRAY_COLS
            ic_end = ic_start + ARRAY_COLS

            # Tile matmul
            act_tile = act_padded[:, ic_start:ic_end]
            wgt_tile = wgt_padded[ic_start:ic_end, oc_start:oc_end]

            tile_result = matmul_int8(act_tile, wgt_tile)
            acc[:, oc_start:oc_end] += tile_result.astype(np.int64).astype(np.int32)

    # Requantize
    output = np.zeros((M, N_padded), dtype=np.int8)
    for m in range(M):
        for n in range(N_padded):
            output[m, n] = requantize(int(acc[m, n]), relu_en, shift, zero_point)

    # Trim to original dimensions
    return output[:, :N]


# ---------------------------------------------------------------------------
# File I/O — generate test vector files readable by SystemVerilog $fscanf
# ---------------------------------------------------------------------------
def write_matrix_file(filepath: str, matrix: np.ndarray):
    """Write expected output in hardware output order.
    
    Hardware processes one OC tile at a time, outputting all spatial positions
    for that tile before moving to the next OC tile. Each beat has ARRAY_COLS values.
    
    Output order: for each OC tile → for each spatial → 8 column values
    For single-tile (OC<=8), this is identical to row-major.
    """
    M, N = matrix.shape
    n_oc_tiles = (N + ARRAY_COLS - 1) // ARRAY_COLS
    with open(filepath, 'w') as f:
        for oc_tile in range(n_oc_tiles):
            oc_start = oc_tile * ARRAY_COLS
            for m in range(M):
                for col in range(ARRAY_COLS):
                    oc = oc_start + col
                    val = int(matrix[m, oc]) if oc < N else 0
                    f.write(f"{val}\n")


def write_config_file(filepath: str, in_channels: int, out_channels: int,
                      spatial_size: int, shift: int, zero_point: int,
                      relu_en: bool):
    """Write layer configuration file."""
    with open(filepath, 'w') as f:
        f.write(f"{in_channels}\n")
        f.write(f"{out_channels}\n")
        f.write(f"{spatial_size}\n")
        f.write(f"{shift}\n")
        f.write(f"{zero_point}\n")
        f.write(f"{1 if relu_en else 0}\n")


def write_packed_weights_file(filepath: str, weights: np.ndarray):
    """
    Write weights packed as 64-bit words (8 × INT8) for SRAM loading.
    Weights are organized as: for each OC tile, for each IC tile,
    ARRAY_ROWS rows of ARRAY_COLS weights packed into 64-bit words.
    """
    K, N = weights.shape
    K_padded = ((K + ARRAY_COLS - 1) // ARRAY_COLS) * ARRAY_COLS
    N_padded = ((N + ARRAY_ROWS - 1) // ARRAY_ROWS) * ARRAY_ROWS

    wgt_padded = np.zeros((K_padded, N_padded), dtype=np.int8)
    wgt_padded[:K, :N] = weights

    with open(filepath, 'w') as f:
        n_oc_tiles = N_padded // ARRAY_ROWS
        n_ic_tiles = K_padded // ARRAY_COLS

        for oc_tile in range(n_oc_tiles):
            for ic_tile in range(n_ic_tiles):
                for row in range(ARRAY_ROWS):
                    # Pack ARRAY_COLS weights into one 64-bit word
                    # SRAM word row = IC index, byte col = OC index
                    # So at SRAM read k, wgt_cols[c] = W[IC=k, OC=c]
                    packed = 0
                    for col in range(ARRAY_COLS):
                        ic = ic_tile * ARRAY_COLS + row
                        oc = oc_tile * ARRAY_ROWS + col
                        byte_val = int(wgt_padded[ic, oc]) & 0xFF
                        packed |= (byte_val << (col * 8))
                    f.write(f"{packed:016x}\n")


def write_packed_activations_file(filepath: str, activations: np.ndarray,
                                   out_channels: int = ARRAY_ROWS):
    """
    Write activations packed as 64-bit words (8 × INT8) for AXI streaming.
    
    Hardware consumption order: for each OC tile → for each spatial → for each IC tile.
    Same activations are replayed for each OC tile (different weights, same input).
    For single-tile (IC<=8, OC<=8), this produces one word per spatial position.
    """
    M, K = activations.shape
    K_padded = ((K + ARRAY_COLS - 1) // ARRAY_COLS) * ARRAY_COLS
    N_padded = ((out_channels + ARRAY_ROWS - 1) // ARRAY_ROWS) * ARRAY_ROWS

    act_padded = np.zeros((M, K_padded), dtype=np.int8)
    act_padded[:, :K] = activations

    n_ic_tiles = K_padded // ARRAY_COLS
    n_oc_tiles = N_padded // ARRAY_ROWS

    with open(filepath, 'w') as f:
        for oc_tile in range(n_oc_tiles):
            for m in range(M):
                for ic_tile in range(n_ic_tiles):
                    packed = 0
                    for col in range(ARRAY_COLS):
                        ic = ic_tile * ARRAY_COLS + col
                        byte_val = int(act_padded[m, ic]) & 0xFF
                        packed |= (byte_val << (col * 8))
                    f.write(f"{packed:016x}\n")


# ---------------------------------------------------------------------------
# Self-test (validates golden model correctness)
# ---------------------------------------------------------------------------
def self_test():
    """Run internal validation of the golden model."""
    print("Golden Model Self-Test")
    print("=" * 50)
    errors = 0

    # Test 1: saturate_int8
    assert saturate_int8(200) == 127, "saturate(200) failed"
    assert saturate_int8(-200) == -128, "saturate(-200) failed"
    assert saturate_int8(50) == 50, "saturate(50) failed"
    print("  [PASS] saturate_int8")

    # Test 2: relu
    assert relu(10) == 10, "relu(10) failed"
    assert relu(-5) == 0, "relu(-5) failed"
    assert relu(0) == 0, "relu(0) failed"
    print("  [PASS] relu")

    # Test 3: arithmetic_right_shift
    assert arithmetic_right_shift(256, 2) == 64, "asr(256,2) failed"
    assert arithmetic_right_shift(-256, 2) == -64, "asr(-256,2) failed"
    assert arithmetic_right_shift(7, 1) == 3, "asr(7,1) failed"
    print("  [PASS] arithmetic_right_shift")

    # Test 4: requantize
    r = requantize(1024, False, 4, 0)  # 1024 >> 4 = 64
    assert r == 64, f"requantize(1024,no_relu,4,0) = {r}, expected 64"
    r = requantize(-100, True, 0, 0)   # ReLU(-100) = 0
    assert r == 0, f"requantize(-100,relu,0,0) = {r}, expected 0"
    r = requantize(10000, False, 2, 0)  # 10000>>2=2500, saturate→127
    assert r == 127, f"requantize(10000,no_relu,2,0) = {r}, expected 127"
    print("  [PASS] requantize")

    # Test 5: 2×2 hand-computed matmul
    # [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    # [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    act = np.array([[1, 2], [3, 4]], dtype=np.int8)
    wgt = np.array([[5, 6], [7, 8]], dtype=np.int8)
    result = matmul_int8(act, wgt)
    expected = np.array([[19, 22], [43, 50]], dtype=np.int32)
    assert np.array_equal(result, expected), f"2x2 matmul failed: {result}"
    print("  [PASS] 2x2 matmul")

    # Test 6: Identity matrix (8×8)
    act = np.arange(1, 9, dtype=np.int8).reshape(1, 8)
    wgt = np.eye(8, dtype=np.int8)
    result = matmul_int8(act, wgt)
    expected = np.arange(1, 9, dtype=np.int32).reshape(1, 8)
    assert np.array_equal(result, expected), f"Identity failed: {result}"
    print("  [PASS] 8x8 identity matmul")

    # Test 7: Full layer with requantization
    act = np.full((1, 8), 10, dtype=np.int8)
    wgt = np.full((8, 8), 5, dtype=np.int8)
    # acc = 10*5*8 = 400 per output element
    # requantize(400, relu=False, shift=3, zp=0) = 400>>3 = 50
    output = inference_layer(act, wgt, False, 3, 0)
    assert np.all(output == 50), f"Layer output failed: {output}"
    print("  [PASS] Full layer inference")

    # Test 8: Tiled execution matches non-tiled
    np.random.seed(42)
    act = np.random.randint(-128, 128, (4, 16), dtype=np.int8)
    wgt = np.random.randint(-128, 128, (16, 16), dtype=np.int8)
    out_flat = inference_layer(act, wgt, True, 8, 0)
    out_tiled = inference_layer_tiled(act, wgt, True, 8, 0)
    assert np.array_equal(out_flat, out_tiled), \
        f"Tiled vs flat mismatch:\n{out_flat}\nvs\n{out_tiled}"
    print("  [PASS] Tiled execution matches flat")

    print("=" * 50)
    print(f"All self-tests PASSED ({8 - errors}/8)")
    return errors == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Accelerator Golden Model")
    parser.add_argument("--test", choices=["all", "self"],
                        help="Run self-tests")
    parser.add_argument("--generate", type=str,
                        help="Generate test vectors for named test")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for generated files")
    args = parser.parse_args()

    if args.test in ("all", "self"):
        success = self_test()
        sys.exit(0 if success else 1)

    if args.generate:
        print(f"Use test_generator.py to generate test: {args.generate}")
        sys.exit(0)

    # Default: run self-test
    self_test()
