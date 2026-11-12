# Training tile V7 results

Design: [training-v7.md](training-v7.md). Every number below comes from a committed test or result file.

## Correctness

| Check | Result |
| --- | --- |
| RTL GEMM regression (`make sim-training-v7`) | 181 GEMMs, 1,710 jobs, every FP32 result bit-exact against `tb/mx_tile.h` |
| Yosys generic netlist (`make sim-training-v7-netlist`) | Same suite, same cycle counts, all pass |
| Shapes | All 88 shapes of 1–4 M tiles, 1–4 N tiles, 1–4 K blocks that fit the scratchpad, exact and scaled values, wraparound bases, stalled and delayed drains |
| Encodings | 64 GEMMs of raw E4M3 and E8M0 bytes, including NaN elements and NaN scales |
| Flags | Quiet NaN element and scale, overflow (flags 5), underflow (flags 3), cleared on start |
| Protocol | Busy-start rejection, 10 reset aborts at different clocks, scratchpad writes overlapped with a running GEMM |
| Mutations (`tools/v7_mutations.sh`) | 7 of 7 caught: early start, unit latency, scale block, short load, bank guard, no back-to-back loads, B-scale select |

The block dot-product unit has its own 2.1-million-vector test against `tb/mx_model.h` (`make sim-mx`).

## Throughput

Clock counts from RTL simulation with an ideal 32-bit host port; V6 numbers from [training-v6-results.md](training-v6-results.md).

| Benchmark | V6 (BF16) | V7 (MXFP8) |
| --- | ---: | ---: |
| 32³ GEMM including fill and drain, clocks | 2,181 | 1,616 |
| Eight pipelined GEMMs, clocks | 2,720 (16³) | 2,824 (16×16×32) |
| MACs per clock, pipelined | 12.0 | 23.2 |
| 16×8×128 GEMM (8 jobs), compute-bound, clocks | — | 366 |
| Peak / sustained MACs per clock | 32 / 32 | 64 / 60.2 |

The 32³ GEMM is host-bound in both tiles: V7 moves half the operand bytes (E4M3 plus 64 scale bytes against BF16), so its fill is 528 words against 1,024, but both drain the same 1,024 FP32 results. The compute-bound run is 366 = 8 × 34 + 94 clocks: one job per 34 clocks plus 94 clocks of start-up, unit latency and the last drain. The regression fails if any job loses a clock.

## Training on the tile

`make train-training-v7` runs the numerics study's training experiment ([numerics-results.md](numerics-results.md#training-emulation)) with every forward, dW and dX GEMM executed by the Verilated V7. The host quantizes each 32-element K block to E4M3 with an E8M0 scale exactly as the study does, then checks every tile result against the study's C++ MXFP8 datapath.

| Run | Seed 0 | Seed 1 | Seed 2 | Mean test loss | vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.81302 | 0.81226 | 0.81830 | 0.81452 | — |
| MXFP8, C++ study datapath | 0.81384 | 0.81414 | 0.82475 | 0.81758 | +0.38% |
| MXFP8, V7 RTL | 0.81384 | 0.81414 | 0.82475 | 0.81758 | +0.38% |

188,448 tile GEMMs (306,971,137 simulated clocks), all bit-exact against the study datapath, so the RTL's loss curve is the study's to the last bit and the tile passes the 1% gate. Results: `results/v7/train_*.txt`; figure: `tools/journal/v7_training_plot.py`.
