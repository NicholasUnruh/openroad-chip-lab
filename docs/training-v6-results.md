# Fused dot-product results

Verified 2026-10-19 against `13fa6a8` RTL (unit later parameterized in `e70d113`, bit-identical for BF16): V6 routes in SKY130HD at 40 MHz with four times V4's multiply-accumulates in 4.6% more standard-cell area. It is not antenna-clean.

## Physical results

| Metric | V4 | V6 |
| --- | ---: | ---: |
| BF16 MACs per clock | 8 | 32 |
| Standard-cell area, µm² | 716,253 | 749,555 |
| SRAM macro, µm² | — | 722,109 |
| Setup / hold slack at 25 ns | +7.49 / +0.33 ns | +2.80 / +0.24 ns |
| Reported fmax | 57.1 MHz | 45.0 MHz |
| Final antenna violations | 0 | 7 nets, 10 pins |

Slew, capacitance, fanout and router DRC checks pass. Detailed routing converged to zero DRC errors after 13 iterations; 807 repair diodes leave seven antenna nets, an open defect. The 1,700 µm die holds 92,701 standard cells at 52% utilization.

## Throughput per area

At 40 MHz with ideal 32-bit host ports (cycle counts from RTL and generic-netlist simulation):

| Metric | V4 | V6 |
| --- | ---: | ---: |
| 32³ GEMM clocks | 4,480 | 2,165 |
| FLOP/clock/mm², 32³, total area | 20.4 | 20.6 |
| FLOP/clock/mm², 32³, standard cells only | 20.4 | 40.4 |
| Peak FLOP/clock/mm², total area | 22.3 | 43.5 |

The datapath is much denser, but V6's SRAM macro, at about 176 µm² per byte, cancels the gain at the ideal port. The scratchpad pays for itself only when the host link is narrower than 32 bits each way. The 1,024-clock scratchpad fill bounds the 32³ result.

## Correctness evidence

| Check | Result |
| --- | --- |
| Dot-product unit vs specification | 4.2M vectors, seven distributions, bit-exact with flags |
| Specification vs exact oracle | Equal wherever nothing truncates; within 9·2^(T−32) otherwise |
| GEMM suite, RTL and generic netlist | 467 GEMMs: all shapes and transposes, flags, aborts, slow drains, training |
| Mutation checks | Unit 9/9; sequencer caught, plus a 32³ throughput bound |
| Power grid | `check_power_grid`: VDD and VSS fully connected after 1.6 µm strap fix |

Power is unmeasured (the reported 0.64 W uses default activity). Local logs: `build/training_v6_{layout,netlist_sim}.log`; metrics in `build/layout-training-v6/`. [Fabrication gates](physical-flow.md) remain separate.
