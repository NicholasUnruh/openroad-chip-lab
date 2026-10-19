# Scratchpad GEMM accelerator

`training_tile_v5` places a 4 KB SRAM scratchpad and an on-chip sequencer in front of the unchanged [V4 engine](training-v4.md). The host writes A and B once; the sequencer runs every 8×8×8 job of a GEMM of up to 32×32×32 and streams finished FP32 output tiles back.

```sh
make sim-training-v5
make sim-training-v5-netlist
make layout-training-v5
```

## Why a second memory level

The V4 lanes read 128 bits of B and 256 bits of C and write 256 bits of C every clock. SKY130 OpenRAM macros are cheap per byte but expensive per port bit: the C traffic alone needs twelve 44×64 macros, about 1.27 M µm², above all of V4's 716k µm². V4 therefore keeps its flop buffers as the register level. The scratchpad needs only 32 bits per clock, the rate of the engine's packed BF16 write port, so one `sky130_sram_1rw1r_128x256_8` (722,109 µm²) serves it.

## Scratchpad layout

Word address `sp_addr` selects 128-bit row `sp_addr[9:2]` and 32-bit lane `sp_addr[1:0]`. Bits 15:0 of a word hold an even BF16 element; bits 31:16 the following element. One row holds eight consecutive elements of a matrix row. A stored matrix with `C` column tiles places element (r, c) at word `4*base + 4*(r*C + c/8) + (c%8)/2`.

| Operand | Stored shape | Column tiles |
| --- | --- | --- |
| A | M×K, or K×M if `transpose_a` | K, or M |
| B | K×N, or N×K if `transpose_b` | N, or K |

Host writes are accepted every clock, including during a GEMM. The host must not write rows the active GEMM reads. A 32×32×32 GEMM fills all 256 rows; smaller ones leave space to preload the next operands.

## Command and results

| Signal | Contract |
| --- | --- |
| `gemm_start` | Accepted when `gemm_busy` is low. Latches `m_tiles`, `n_tiles`, `k_tiles` (tiles minus one), `a_base`, `b_base` (rows) and both transposes. Ignored while busy. |
| `result_valid` | An output tile is readable. `read_addr` selects element `8*row+col`; `read_data` is asynchronous FP32. Data is valid only in cycles where `result_valid` is high. |
| `result_tile` | `{tile_row, tile_col}` of the readable tile. Tiles arrive in row-major pairs. |
| `result_ack` | Releases the readable tile. Sampled at the rising edge while `result_valid` is high. |
| `gemm_done` | One-cycle pulse after the last tile is released. |
| `exception_flags` | OR of all job flags; cleared on accepted start. |

C starts at zero. Output tiles are computed in pairs over K, alternating engine buffers, so A/B loading overlaps computation. An unpaired final tile serializes loading and computation. Reset aborts the GEMM.

There is no DMA or bus adapter. See [results](training-v5-results.md) and the [fabrication gates](physical-flow.md).
