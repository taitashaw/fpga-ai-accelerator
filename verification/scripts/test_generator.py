#!/usr/bin/env python3
"""
Test Vector Generator for AI Inference Accelerator
====================================================
Generates 6 directed test scenarios with stimulus files for RTL simulation.

Each test produces:
  - config.txt:       Layer configuration (IC, OC, spatial, shift, zp, relu)
  - weights.hex:      Packed 64-bit weight words (hex, for $readmemh)
  - activations.hex:  Packed 64-bit activation words (hex, for $readmemh)
  - expected.txt:     Expected INT8 outputs (decimal, one per line)
  - info.txt:         Human-readable test description

Author: John Bagshaw, Senior FPGA Design Engineer
Date:   February 2026
"""

import numpy as np
import os
import sys

# Add parent path for golden model import
sys.path.insert(0, os.path.dirname(__file__))
from golden_model import (
    ARRAY_ROWS, ARRAY_COLS, ACT_MIN, ACT_MAX, WGT_MIN, WGT_MAX,
    inference_layer_tiled, write_matrix_file, write_config_file,
    write_packed_weights_file, write_packed_activations_file
)


def generate_test(test_name: str, output_dir: str):
    """Generate test vectors for the named test."""
    os.makedirs(output_dir, exist_ok=True)

    generators = {
        "test_identity_8x8":    gen_identity_8x8,
        "test_random_small":    gen_random_small,
        "test_boundary_max":    gen_boundary_max,
        "test_boundary_min":    gen_boundary_min,
        "test_relu_activation": gen_relu_activation,
        "test_multi_tile":      gen_multi_tile,
        "test_deep_ic":         gen_deep_ic,
        "test_deep_oc":         gen_deep_oc,
        "test_large_tile":      gen_large_tile,
        "test_asymmetric":      gen_asymmetric,
    }

    if test_name == "all":
        for name, gen_fn in generators.items():
            test_dir = os.path.join(output_dir, name)
            print(f"Generating: {name}")
            gen_fn(test_dir)
        return True

    if test_name not in generators:
        print(f"Unknown test: {test_name}")
        print(f"Available: {list(generators.keys())}")
        return False

    generators[test_name](output_dir)
    return True


# ---------------------------------------------------------------------------
# Test 1: Identity Matrix (8×8)
# ---------------------------------------------------------------------------
def gen_identity_8x8(output_dir: str):
    """
    Pass-through test: activations × identity = activations.
    Verifies datapath integrity with no accumulation complexity.
    """
    os.makedirs(output_dir, exist_ok=True)

    act = np.arange(1, 9, dtype=np.int8).reshape(1, 8)  # [1,2,3,4,5,6,7,8]
    wgt = np.eye(8, dtype=np.int8)

    shift = 0
    zero_point = 0
    relu_en = False

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=8, out_channels=8, spatial_size=1,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: Identity 8x8\n")
        f.write("Activations: [1,2,3,4,5,6,7,8] (1×8)\n")
        f.write("Weights: 8×8 identity matrix\n")
        f.write(f"Expected output: {expected.flatten().tolist()}\n")
        f.write("Coverage: Datapath integrity, no accumulation\n")


# ---------------------------------------------------------------------------
# Test 2: Random Small (4×8 × 8×8)
# ---------------------------------------------------------------------------
def gen_random_small(output_dir: str):
    """
    Small random test with requantization including ReLU.
    4 spatial rows × 8 IC × 8 OC — exercises general MAC + quantization.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    act = np.random.randint(-50, 51, (4, 8), dtype=np.int8)
    wgt = np.random.randint(-50, 51, (8, 8), dtype=np.int8)

    shift = 5
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=8, out_channels=8, spatial_size=4,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: Random Small (4×8 × 8×8)\n")
        f.write(f"Activations shape: {act.shape}\n")
        f.write(f"Weights shape: {wgt.shape}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output shape: {expected.shape}\n")
        f.write(f"Expected:\n{expected}\n")
        f.write("Coverage: General MAC, ReLU clipping, requantization\n")


# ---------------------------------------------------------------------------
# Test 3: Boundary Max (all +127)
# ---------------------------------------------------------------------------
def gen_boundary_max(output_dir: str):
    """
    Maximum positive inputs: all activations and weights = +127.
    Accumulator = 127 × 127 × 8 = 129,032 per element.
    Exercises positive overflow boundary.
    """
    os.makedirs(output_dir, exist_ok=True)

    act = np.full((1, 8), 127, dtype=np.int8)
    wgt = np.full((8, 8), 127, dtype=np.int8)

    # 127*127*8 = 129,032. shift=10 → 129032>>10 = 126
    shift = 10
    zero_point = 0
    relu_en = False

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=8, out_channels=8, spatial_size=1,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: Boundary Max (all +127)\n")
        f.write(f"Accumulator per element: 127×127×8 = 129,032\n")
        f.write(f"Shift={shift} → 129032>>10 = 126\n")
        f.write(f"Expected output: {expected.flatten().tolist()}\n")
        f.write("Coverage: Positive overflow boundary, large accumulation\n")


# ---------------------------------------------------------------------------
# Test 4: Boundary Min (all -128)
# ---------------------------------------------------------------------------
def gen_boundary_min(output_dir: str):
    """
    Maximum negative inputs: activations = -128, weights = +127.
    Accumulator = -128 × 127 × 8 = -130,048 per element.
    Exercises negative overflow boundary.
    """
    os.makedirs(output_dir, exist_ok=True)

    act = np.full((1, 8), -128, dtype=np.int8)
    wgt = np.full((8, 8), 127, dtype=np.int8)

    # -128*127*8 = -130,048. shift=10 → -130048>>10 = -128 (exact!)
    shift = 10
    zero_point = 0
    relu_en = False

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=8, out_channels=8, spatial_size=1,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: Boundary Min (act=-128, wgt=+127)\n")
        f.write(f"Accumulator per element: -128×127×8 = -130,048\n")
        f.write(f"Shift={shift} → -130048>>10 = -128 (boundary!)\n")
        f.write(f"Expected output: {expected.flatten().tolist()}\n")
        f.write("Coverage: Negative overflow boundary, saturation edge\n")


# ---------------------------------------------------------------------------
# Test 5: ReLU Activation (mixed signs)
# ---------------------------------------------------------------------------
def gen_relu_activation(output_dir: str):
    """
    Mixed positive/negative activations with ReLU enabled.
    Verifies that negative accumulator values are clamped to zero
    before requantization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Alternating positive/negative activations
    act = np.array([[ 10, -20,  30, -40,  50, -60,  70, -80],
                    [-10,  20, -30,  40, -50,  60, -70,  80]], dtype=np.int8)
    # Weights that will produce both positive and negative accumulations
    wgt = np.array([[ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5],
                    [ 5,  5, -5, -5,  5,  5, -5, -5]], dtype=np.int8)

    shift = 4
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=8, out_channels=8, spatial_size=2,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: ReLU Activation (mixed signs)\n")
        f.write(f"Activations:\n{act}\n")
        f.write(f"Weights:\n{wgt}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output:\n{expected}\n")
        f.write("Coverage: ReLU clamping, negative→zero, mixed signs\n")


# ---------------------------------------------------------------------------
# Test 6: Multi-Tile (4×16 × 16×16 → requires 4 tiles on 8×8 array)
# ---------------------------------------------------------------------------
def gen_multi_tile(output_dir: str):
    """
    Multi-tile test: 16 IC × 16 OC on 8×8 array requires:
      - 2 IC tiles × 2 OC tiles = 4 tiles total
      - Cross-tile partial sum accumulation

    This is the critical test for verifying the FSM tiling logic.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(123)

    act = np.random.randint(-30, 31, (4, 16), dtype=np.int8)
    wgt = np.random.randint(-30, 31, (16, 16), dtype=np.int8)

    shift = 8
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=16, out_channels=16, spatial_size=4,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, wgt.shape[1])
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write("Test: Multi-Tile (4×16 × 16×16)\n")
        f.write(f"IC tiles: {16//ARRAY_COLS}, OC tiles: {16//ARRAY_ROWS}\n")
        f.write(f"Total tiles: {(16//ARRAY_COLS)*(16//ARRAY_ROWS)}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output:\n{expected}\n")
        f.write("Coverage: Tiling, IC accumulation, cross-tile psum\n")


# ---------------------------------------------------------------------------
# Test 7: Deep IC Tiling (IC=32, OC=8) — 4 IC tiles, 1 OC tile
# ---------------------------------------------------------------------------
def gen_deep_ic(output_dir: str):
    """
    Stress test for deep IC accumulation: 4 IC tiles summed before quantize.
    Verifies partial-sum preservation across many IC iterations.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(200)

    IC, OC, M = 32, 8, 4
    act = np.random.randint(-20, 21, (M, IC), dtype=np.int8)
    wgt = np.random.randint(-10, 11, (IC, OC), dtype=np.int8)

    shift = 10
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=IC, out_channels=OC, spatial_size=M,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, OC)
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    n_ic = IC // ARRAY_COLS
    n_oc = OC // ARRAY_ROWS
    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write(f"Test: Deep IC ({M}x{IC} x {IC}x{OC})\n")
        f.write(f"IC tiles: {n_ic}, OC tiles: {n_oc}, Total: {n_ic*n_oc}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output:\n{expected}\n")
        f.write("Coverage: Deep IC accumulation (4 tiles), partial-sum integrity\n")


# ---------------------------------------------------------------------------
# Test 8: Deep OC Tiling (IC=8, OC=32) — 1 IC tile, 4 OC tiles
# ---------------------------------------------------------------------------
def gen_deep_oc(output_dir: str):
    """
    Stress test for deep OC iteration: 4 OC tiles with activation replay.
    Verifies correct weight bank addressing across many OC tiles.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(300)

    IC, OC, M = 8, 32, 4
    act = np.random.randint(-25, 26, (M, IC), dtype=np.int8)
    wgt = np.random.randint(-15, 16, (IC, OC), dtype=np.int8)

    shift = 8
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=IC, out_channels=OC, spatial_size=M,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, OC)
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    n_ic = IC // ARRAY_COLS
    n_oc = OC // ARRAY_ROWS
    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write(f"Test: Deep OC ({M}x{IC} x {IC}x{OC})\n")
        f.write(f"IC tiles: {n_ic}, OC tiles: {n_oc}, Total: {n_ic*n_oc}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output:\n{expected}\n")
        f.write("Coverage: Deep OC iteration (4 tiles), activation replay, SRAM addressing\n")


# ---------------------------------------------------------------------------
# Test 9: Large Tile (IC=32, OC=32, Spatial=8) — 16 tiles, max spatial
# ---------------------------------------------------------------------------
def gen_large_tile(output_dir: str):
    """
    Full stress test: 4x4=16 weight tiles, 8 spatial positions.
    Exercises all tiling dimensions simultaneously at scale.
    Weight words: 128, activation words: 128, expected outputs: 256.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(400)

    IC, OC, M = 32, 32, 8
    act = np.random.randint(-15, 16, (M, IC), dtype=np.int8)
    wgt = np.random.randint(-8, 9, (IC, OC), dtype=np.int8)

    shift = 10
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=IC, out_channels=OC, spatial_size=M,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, OC)
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    n_ic = IC // ARRAY_COLS
    n_oc = OC // ARRAY_ROWS
    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write(f"Test: Large Tile ({M}x{IC} x {IC}x{OC})\n")
        f.write(f"IC tiles: {n_ic}, OC tiles: {n_oc}, Total: {n_ic*n_oc}\n")
        f.write(f"Spatial: {M}, Weight words: {n_ic*n_oc*8}, Act words: {n_oc*M*n_ic}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output shape: {expected.shape}\n")
        f.write(f"Expected:\n{expected}\n")
        f.write("Coverage: Full-scale tiling (16 tiles), max spatial, all loops\n")


# ---------------------------------------------------------------------------
# Test 10: Asymmetric Tiling (IC=24, OC=40, Spatial=6) — 3x5=15 tiles
# ---------------------------------------------------------------------------
def gen_asymmetric(output_dir: str):
    """
    Asymmetric tile counts: 3 IC tiles x 5 OC tiles = 15 total.
    Non-square tiling stresses loop boundary conditions and index arithmetic.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(500)

    IC, OC, M = 24, 40, 6
    act = np.random.randint(-20, 21, (M, IC), dtype=np.int8)
    wgt = np.random.randint(-10, 11, (IC, OC), dtype=np.int8)

    shift = 10
    zero_point = 0
    relu_en = True

    expected = inference_layer_tiled(act, wgt, relu_en, shift, zero_point)

    write_config_file(os.path.join(output_dir, "config.txt"),
                      in_channels=IC, out_channels=OC, spatial_size=M,
                      shift=shift, zero_point=zero_point, relu_en=relu_en)
    write_packed_weights_file(os.path.join(output_dir, "weights.hex"), wgt)
    write_packed_activations_file(os.path.join(output_dir, "activations.hex"), act, OC)
    write_matrix_file(os.path.join(output_dir, "expected.txt"), expected)

    n_ic = IC // ARRAY_COLS
    n_oc = OC // ARRAY_ROWS
    with open(os.path.join(output_dir, "info.txt"), 'w') as f:
        f.write(f"Test: Asymmetric ({M}x{IC} x {IC}x{OC})\n")
        f.write(f"IC tiles: {n_ic}, OC tiles: {n_oc}, Total: {n_ic*n_oc}\n")
        f.write(f"Spatial: {M}, Weight words: {n_ic*n_oc*8}, Act words: {n_oc*M*n_ic}\n")
        f.write(f"Shift={shift}, ZP={zero_point}, ReLU={relu_en}\n")
        f.write(f"Expected output shape: {expected.shape}\n")
        f.write(f"Expected:\n{expected}\n")
        f.write("Coverage: Asymmetric tiling (3x5=15 tiles), non-square loop bounds\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Vector Generator")
    parser.add_argument("--test", type=str, default="all",
                        help="Test name or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default="verification/tests",
                        help="Output directory")
    args = parser.parse_args()

    success = generate_test(args.test, args.output_dir)
    sys.exit(0 if success else 1)
