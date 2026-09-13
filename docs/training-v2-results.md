# Four-lane training results

Verified 2026-09-12 against implementation commit `6b5e8dc`. Scalar and four-lane RTL and synthesized netlists each pass 405 numerical/protocol/training jobs plus 256 timed benchmark tiles. Dense-layer training takes 32 SGD updates from loss 3.85547 to 0.0000828033; both gradients pass finite differences within 0.003.

## Transfer benchmark

Each mode runs 64 checked tiles, counting 128 FLOPs per tile (two per FMA). Cycles include one start clock, computation/drain and one clock per 32-bit read/write. Preloading, host software and real external-interface latency are excluded. MFLOPS project these measured cycle counts at the respective constrained clocks; they are not silicon measurements.

| Mode | Bytes/tile | Scalar clocks/tile | V2 clocks/tile | Scalar MFLOPS, 10 MHz | V2 MFLOPS, 40 MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inputs resident, read C | 64 | 81 | 36 | 15.80 | 142.22 |
| Stream A/B, read C | 192 | 113 | 68 | 11.33 | 75.29 |
| Reuse B; stream A, read C | 128 | 97 | 52 | 13.20 | 98.46 |
| Stream A/B/C, accumulate, read C | 256 | 129 | 84 | 9.92 | 60.95 |

The first three modes overwrite C. Reusing B saves one third of traffic versus streaming A/B for the same operation. Streamed V2 arithmetic utilization is 23.53%; weight reuse raises it to 30.77%. The 192-byte register storage and idle-only writes limit utilization; SRAM and overlapping transfers remain future work.

At equal clocks, V2 is 3.37× faster in compute cycles and 1.66× faster with A/B transfers and C reads included. The different target clocks produce a 6.65× streamed projection.

## Routed physical comparison

SKY130HD, typical 25°C/1.8 V; values from `build/layout-training{,-v2}/metrics.json`:

| Metric | Scalar | V2 |
| --- | ---: | ---: |
| Constrained clock | 10 MHz | 40 MHz |
| Estimated maximum clock | 27.20 MHz | 66.87 MHz |
| Setup / hold slack | +63.23 / +0.467 ns | +10.05 / +0.408 ns |
| Standard-cell area | 125,031 µm² | 290,778 µm² |
| Core area | 325,045 µm² | 765,115 µm² |

At estimated maxima, streamed projections are 30.81 versus 125.88 MFLOPS: **4.09× throughput for 2.33× cell area**. These frequencies are timing estimates, not retargeted builds. Final routing, antenna, setup/hold and electrical violation counts are zero. KLayout independently reads one top cell, `training_tile_v2`, with an 878.88 µm square bounding box; evidence: `build/training_v2_gds_inspection.log` and `build/layout-training-v2/{timing.rpt,router_drc.rpt}`.

## Evidence and reproduction

Run the [V2 commands](training-v2.md) and corresponding scalar targets without `-v2`. Simulation evidence: `build/training_v2_sim.log`, `build/training_v2_netlist_sim.log`, `build/training_baseline_compare.log`, `build/training_baseline_netlist_compare.log`. Exact cycle/traffic comparison: `build/training_v2_comparison.json`. Generic V2 synthesis: 36,884 cells and zero structural problems in `build/training-v2_synth.log`.

The physical run reuses `ORFS_TOOL_DIR=/tmp/bazel-orfs-inspect`, pinned to bazel-orfs `a25cf9abcdc753b25e5e2d28487844556d8e6db2`, matching the default checkout. Its log is `build/training_v2_layout.log`.

Power/energy remain unqualified. Numerical regression is bounded evidence; [fabrication gates](physical-flow.md) remain separate.
