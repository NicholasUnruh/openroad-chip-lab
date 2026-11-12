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
| MXFP8, C++ study datapath | 0.81384 | 0.81414 | 0.82475 | 0.81758 | +0.37% |
| MXFP8, V7 RTL | 0.81384 | 0.81414 | 0.82475 | 0.81758 | +0.37% |

188,448 tile GEMMs (306,971,137 simulated clocks), all bit-exact against the study datapath, so the RTL's loss curve is the study's to the last bit and the tile passes the 1% gate. This is the same experiment, seeds and MXFP8 row as the numerics study; the percentage is computed from the unrounded means. Results: `results/v7/train_*.txt`; figure: `tools/journal/v7_training_plot.py`.

## SKY130HD layout

V7 uses V6's flow, 25 ns clock, SRAM macro placement and power grid. Only the die grows, from 1,700 to 1,800 µm, so the larger datapath lands near V6's utilization. Metrics: `results/sky130/training_tile_v{6,7}_layout.json` (`bin/layout training_tile_v7`).

| | V6 (BF16) | V7 (MXFP8) |
| --- | ---: | ---: |
| Standard-cell area, µm² | 766,541 | 974,138 |
| SRAM macro, µm² | 722,109 | 722,109 |
| Utilization (macro included) | 52.8% | 53.6% |
| Standard cells | 92,142 | 112,574 |
| Setup / hold slack, ns | +5.06 / +0.17 | +5.03 / +0.26 |
| fmax, MHz | 50.2 | 50.1 |
| Router DRC, electrical violations | 0, 0 | 0, 0 |
| Peak MACs/clock/mm², cells + macro | 21.5 | 37.7 (1.75×) |
| Peak MACs/clock/mm², cells only | 41.7 | 65.7 (1.57×) |
| 32³ GEMM MACs/clock/mm², cells + macro | 10.1 | 12.0 (1.19×) |

Both tiles close timing with the same 5 ns margin, and both critical paths are one pipeline stage inside a dot unit (V6 `p_s1` → `p_s2`, V7 `c_s2` → `c_s3`; `timing.rpt` from the flow). V7 doubles the peak MACs per clock for 27% more standard-cell area. Most of that growth is registers, not arithmetic: flops grow from 12,148 to 14,864, and their area grows by 98k µm². V7's two MXFP8 units (273k µm² before placement) are smaller than V6's four BF16 dot units (284k µm²). The tile logic around them grows by 153k µm², because V7 holds 4,352 more operand-register bits ([area by module](physical-flow.md)).

## SKY130HD power

Each routed netlist runs its own regression benchmark in Verilator with zero-delay cell models generated from the Liberty functions (`tools/liberty_cells.py`). The benchmark checks every result bit-exactly and dumps one benchmark window as a VCD at the 40 MHz target. OpenSTA reads the VCD and the extracted parasitics and reports power. All cell pins are annotated from the VCD except antenna diodes, which exist only in the routed database and switch nothing. `tools/tile_power.sh` fails the run if more than 1% of the remaining pins fall back to default activity. Energy per MAC is total energy over the whole benchmark, including host traffic and idle clocks. Results: `results/sky130/tile-power/`.

| Tile, benchmark | Clocks | MACs | Power, mW | SRAM macro, mW | Energy/MAC, pJ | Without macro, pJ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4, 32³ streamed | 4,480 | 32,768 | 54.6 | — | 187 | 187 |
| V6, 32³ | 2,181 | 32,768 | 64.9 | 14.4 | 108 | 84 |
| V7, 32³ | 1,616 | 32,768 | 75.5 | 14.4 | 93 | 75 |
| V6, eight pipelined 16³ | 2,720 | 32,768 | 63.6 | 14.4 | 132 | 102 |
| V7, eight pipelined 16×16×32 | 2,824 | 65,536 | 76.9 | 14.4 | 83 | 67 |

| Power group, mW | V6 pipelined | V7 pipelined |
| --- | ---: | ---: |
| Flops | 22.3 | 27.7 |
| Clock tree | 21.7 | 26.0 |
| Combinational logic | 5.3 | 8.8 |
| SRAM macro | 14.4 | 14.4 |

**The tile does not deliver the unit-level energy gain.** In isolation, an MXFP8 unit uses 2.1× less energy per MAC than a BF16 dot8 unit in SKY130HD (next section). At the tile level, V7 saves 1.16× per MAC on the 32³ GEMM and 1.59× on the pipelined benchmark. The breakdown shows why. Flops and the clock tree take 70% of each tile's power, and they toggle every clock whether the units are busy or not. All combinational logic, arithmetic included, is 8–12%. Tile power is therefore nearly constant per clock (64–77 mW), and energy per MAC follows MACs per clock more than the format. V7 gets its gain mostly by doing twice the MACs per clock, not from cheaper arithmetic. Clock gating the idle operand registers would be the next lever, and these tiles have none.

The SRAM macro's 14.4 mW is the same in both tiles and on both benchmarks. Its Liberty model charges a flat internal energy on every clock edge, enabled or not, so treat the macro column as an upper bound. The columns without the macro carry the comparison.

## ASAP7 prediction vs SKY130HD measurement

The numerics study ranked formats on routed ASAP7 units ([format cost](journal/2026-11-12-format-silicon-cost.md)). The same three harnesses were routed in SKY130HD at a 10 ns target, and their power was measured with the same method (`make ppa-sky130`; `python3 tools/numerics/report.py`). Gains are relative to the BF16 dot8 unit in the same PDK.

| Unit | SKY130 area/MAC, µm² | SKY130 energy/MAC, pJ | Area/MAC gain, ASAP7 → SKY130 | Energy/MAC gain, ASAP7 → SKY130 |
| --- | ---: | ---: | ---: | ---: |
| BF16 dot8 | 12,261 | 11.70 | 1 → 1 | 1 → 1 |
| MXFP8 dot32 | 5,939 | 5.53 | 1.94× → 2.06× | 2.33× → 2.12× |
| MXFP6 E2M3 dot32 | 3,269 | 3.50 | 3.58× → 3.75× | 3.40× → 3.34× |

| Tile claim | ASAP7 unit prediction | SKY130 tile measurement |
| --- | ---: | ---: |
| MXFP8 vs BF16, throughput per area | 1.90× | 1.75× peak (cells + macro), 1.57× (cells only) |
| MXFP8 vs BF16, energy per MAC | 2.33× | 1.16× (32³ GEMM), 1.59× (pipelined) |

**The ranking holds and the unit ratios transfer.** Every unit ratio agrees across PDKs within 11%, so the ASAP7 study's ordering of BF16, MXFP8 and MXFP6 E2M3 survives a real PDK. All three SKY130 units close at about 100 MHz, because the 10 ns target, not the logic, sets fmax. Their throughput ratios therefore reduce to the area ratios.

**The tile-level gain is smaller than the unit predicted.** Area per MAC is the closer of the two, because the arithmetic is a large share of the standard cells. Energy is far off, for the clock-and-flop reason above. A format study that scores bare units overstates the tile-level energy gain by about 1.5–2×. The ranking is still right; the size of the win is not.
