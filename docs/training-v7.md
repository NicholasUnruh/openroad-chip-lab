# MXFP8 block dot-product accelerator

`training_tile_v7` is the format the [numerics study](numerics-results.md) picked, built as a tile: V6's scratchpad, sequencer and C banks, with two OCP MXFP8 32-term block dot-product units (`rtl/mx_dot32_fp32.v`, E4M3 elements, one E8M0 scale per 32-element block, FP32 accumulation) instead of four BF16 dot8 units. It sustains 64 multiply-accumulates per clock against V6's 32, with the same 4 KB scratchpad and the same 128-bit read port.

```sh
make sim-training-v7            # 181 GEMMs, RTL
make sim-training-v7-netlist    # same suite on the Yosys netlist
tools/v7_mutations.sh           # seven planted sequencer bugs, each must fail the suite
make train-training-v7          # the study's MXFP8 training run on the RTL (about an hour)
```

## Why two units and 8×8×32 jobs

The read port, not the arithmetic, sets throughput. One scratchpad row is 128 bits: eight BF16 elements in V6, sixteen E4M3 elements here. MX blocks run along K, so a job reduces one whole 32-element block: an 8×8 output tile × K = 32 is 2,048 MACs and needs 16 rows of A, 16 rows of B and two scale rows, 34 reads. One block dot product per unit per clock covers the 64 outputs in 32 clocks with two units. Four units would finish in 16 clocks and then wait 18 for operands.

| | V6 | V7 |
| --- | ---: | ---: |
| Job | 8×8×8 BF16 | 8×8×32 MXFP8 |
| Scratchpad reads per job | 16 | 34 |
| Compute clocks per job | 16 | 32 |
| Units × MACs per unit | 4 × 8 | 2 × 32 |
| Peak / sustained MACs per clock | 32 / 32 | 64 / 60.2 |
| Operand registers (A+B, double-buffered) | 4,096 bits | 8,192 bits |

The two scale reads cost 6% of peak (2,048 MACs per 34 clocks). Holding scales in a side register file would win that back for about 1,024 more flops; the 6% loss was kept for the simpler contract.

## Host contract changes from V6

Interface signals, result draining, flags, aborts and C-bank behavior are V6's ([V6 interface](training-v6-interface.md)), except:

- **No transposes.** MX quantizes along the reduction dimension, so a transposed operand must be re-quantized along its new K anyway. The host quantizes each operand per GEMM and stores it K-contiguous: A as M×K, B as N×K (C = A·Bᵀ). This removes V6's transpose datapath.
- **K counts 32-element blocks.** `k_tiles` is the number of K blocks minus one (K up to 128). `m_tiles` and `n_tiles` are 8-row tiles minus one, as in V6.
- **Scales.** `a_scale_base` and `b_scale_base` point at scale rows. The scale row of (tile t, block kb) is `base + t·(k_tiles+1) + kb`; its bytes 0–7 are the E8M0 scales of the tile's eight rows. Bytes 8–15 are unused.
- **Element layout.** Element (r, k) of an operand is byte `16·base + r·32·(k_tiles+1) + k`, little-endian within each 32-bit word.

A GEMM uses 17·(m+n)·k scratchpad rows (m, n in tiles, k in blocks) and must fit in 256. The largest square GEMM is 32×32×32, as in V6.

## Arithmetic

Each unit returns `c + 2^(sa+sb−254) · Σ a[k]·b[k]` over 32 E4M3 pairs, correctly rounded once to FP32. Products and their sum are exact in fixed point, so the only rounding is the final one; there is no alignment-frame truncation as in V6's BF16 unit. An output chains its K blocks from +0, rounding once per block. Specials: an E8M0 scale of 0xFF or an E4M3 NaN (S.1111.111) gives the canonical NaN without the invalid flag; overflow, underflow and inexact flags are FP32's. The bit-exact reference is `tb/mx_model.h` (`mx_dot32`) chained by `reference()` in `tb/mx_tile.h`.

## Datapath and timing

The loader reads A scales, B scales, 16 B rows, then 16 A rows. Each clock of a job broadcasts one A row (32 bytes and its scale) to both units; unit u reads column 4u + c of B. Results write back seven clocks later (the unit's latency). Row r of A is complete two clocks after load index 19 + 2r and first needed 4r clocks into the job, so a job starts at load index 20. Loads run back to back, so jobs issue every 34 clocks. The mutation checks show the start threshold is exact: starting at index 19 corrupts results.

Results: [v7-results.md](v7-results.md).
