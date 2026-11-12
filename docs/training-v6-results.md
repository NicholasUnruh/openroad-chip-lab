# Fused dot-product results

Re-routed 2026-11-12 on current RTL, including the two dot-product pipeline stages added for the ASAP7 study: V6 routes in SKY130HD at 40 MHz with four times V4's multiply-accumulates in 7.0% more standard-cell area. It is not antenna-clean.

## Physical results

| Metric | V4 | V6 |
| --- | ---: | ---: |
| BF16 MACs per clock | 8 | 32 |
| Standard-cell area, µm² | 716,253 | 766,541 |
| SRAM macro, µm² | — | 722,109 |
| Setup / hold slack at 25 ns | +7.49 / +0.33 ns | +5.06 / +0.17 ns |
| Reported fmax | 57.1 MHz | 50.2 MHz |
| Final antenna violations | 0 | 9 nets, 18 pins |

Slew, capacitance, fanout and router DRC checks pass, and detailed routing converges to zero DRC errors. 1,758 repair diodes (1,098 after detailed routing) leave nine antenna nets, an open defect. The 1,700 µm die holds 92,142 standard cells at 53% utilization.

The pipeline stages moved the critical path out of the dot-product unit: setup slack rose from +2.80 to +5.06 ns (fmax 45.0 → 50.2 MHz) for 2.3% more standard-cell area (749,555 µm²) and 16 more clocks per GEMM. The 2026-10-19 route without them left 7 antenna nets after 807 post-route diodes.

## Throughput per area

At 40 MHz with ideal 32-bit host ports (cycle counts from RTL and generic-netlist simulation; the pipeline stages add 16 clocks to the earlier 2,165):

| Metric | V4 | V6 |
| --- | ---: | ---: |
| 32³ GEMM clocks | 4,480 | 2,181 |
| Eight pipelined 16³ GEMMs, clocks | — | 2,720 |
| FLOP/clock/mm², 32³, total area | 20.4 | 20.2 |
| FLOP/clock/mm², 32³, standard cells only | 20.4 | 39.2 |
| Peak FLOP/clock/mm², total area | 22.3 | 43.0 |

The datapath is much denser, but V6's SRAM macro, at about 176 µm² per byte, cancels the gain at the ideal port. The scratchpad pays for itself only when the host link is narrower than 32 bits each way. The 1,024-clock scratchpad fill bounds the 32³ result.

## Correctness evidence

| Check | Result |
| --- | --- |
| Dot-product unit vs specification | 4.2M vectors, seven distributions, bit-exact with flags |
| Specification vs exact oracle | Equal wherever nothing truncates; within 9·2^(T−32) otherwise |
| GEMM suite, RTL and generic netlist | 467 GEMMs: all shapes and transposes, flags, aborts, slow drains, training |
| Mutation checks | Unit 9/9; sequencer caught, plus a 32³ throughput bound |
| Power grid | `check_power_grid`: VDD and VSS fully connected after 1.6 µm strap fix |

Power is unmeasured: the default-activity estimate moved from 0.64 W to 0.07 W between the two routes, which says more about the estimate than the design. Local logs: `build/training_v6_{layout,netlist_sim}.log`; metrics in `build/layout-training-v6/`. [Fabrication gates](physical-flow.md) remain separate.
