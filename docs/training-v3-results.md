# BF16 accelerator results

Verified 2026-09-12 against implementation `ba50e4c`. V3 uses native BF16 operands/multiplication and FP32 accumulation. RTL and generic synthesized-netlist simulations each pass 4,868 functional/training jobs plus 256 timed tiles.

## Functional evidence

| Check | Result |
| --- | --- |
| Structural width gate | Four unsigned 8×8→16 multipliers; 16-bit A/B, 32-bit C; 256-byte matrix storage |
| Numerical reference | Bit comparison with host FP32 `std::fma` on expanded BF16 operands |
| Encoding sweep | All 65,536 BF16 A encodings across 4,096 tiles; bounded B/C combinations |
| Boundaries | NaNs, infinities, signed zeros, subnormals, rounding and directed exception flags |
| Protocol | Both buffers, concurrent reads/writes, ownership, alignment, start/reset priority, all 19 abort phases |
| Training | BF16 forward/dW/dX/SGD, FP32 master weights; 48 updates reduce loss 3.85547→0.000000125672 |
| Larger GEMM | 7×13 by 13×9; 24 hardware jobs, retained C and padded edge tiles |
| Generic synthesis | 24,965 cells; zero structural problems |
| Regressions | V2 RTL/netlist and scalar RTL: 405 jobs plus 256 benchmark tiles each; MAC8 simulation/synthesis pass |

The encoding sweep is not exhaustive operand-pair or IEEE conformance verification. Training demonstrates one dense-layer workload; host software computes loss, dY, quantization and scheduling.

## Transfer benchmark

Every row counts 64 tiles and 8,192 FLOPs, including required initial loading, starts, computation and final result reads. Ports permit one 32-bit read and one 32-bit write per clock. External bus and host execution latency are excluded. V2 results are from the freshly rerun [FP32 baseline](training-v2-results.md).

| Mode | Total clocks | Total bytes | Projected MFLOPS at 40 MHz |
| --- | ---: | ---: | ---: |
| V2 FP32 stream A/B | 4,352 | 12,288 | 75.29 |
| V3 BF16 serial A/B | 3,328 | 8,192 | 98.46 |
| V3 BF16 overlap A/B | 1,312 | 8,192 | 249.76 |
| V2 FP32 stream A/B/C | 5,376 | 16,384 | 60.95 |
| V3 BF16 serial A/B/C | 4,352 | 12,288 | 75.29 |
| V3 BF16 overlap A/B/C | 2,147 | 12,288 | 152.62 |

Overlap alone improves BF16 A/B throughput 2.54×; BF16 plus buffering improves it 3.32× versus V2 at equal clocks. Steady launch intervals are 20 clocks for A/B and 33 for A/B/C; startup/drain remain included above. Results compare different operand precision.

## Physical implementation

SKY130HD, typical 25°C/1.8 V, 40 MHz constraint: setup +10.0163 ns, hold +0.405551 ns; estimated maximum clock 66.7391 MHz. Standard-cell area is 251,834 µm², 13.39% below recorded V2; core area is 658,839 µm². Final routing, antenna, timing and electrical violation counts are zero.

Physical-flow commit `0d046b5` applies 10% repair margins. The run uses `ORFS_TOOL_DIR=/tmp/bazel-orfs-inspect`. KLayout reads one `training_tile_v3` top, 815.31 µm square. Evidence: `build/layout-training-v3/`, `build/training_v3_gds_inspection.log`.

## Reproduction and evidence

Use the [V3 commands](training-v3.md). Local evidence: `build/training_v3_{sim,netlist_sim,structure_check,layout_verified}.log`, `build/training-v3_synth.log` and `build/training_v3_comparison.json`. Regression logs: `build/training_{v2_regression,v2_netlist_regression,scalar_regression}.log`.

Power and energy remain unmeasured; [fabrication gates](physical-flow.md) remain separate.
