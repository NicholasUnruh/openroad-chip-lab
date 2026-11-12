# Eight-lane BF16 results

Verified 2026-09-13: V4 delivers 2.29× V3's throughput on the same streamed 8×8 GEMM workload with one-third less host traffic. Both use BF16 operands/multiplication and FP32 accumulation. Physical checks pass. V3 is retired; its figures here are history.

## Equal-work comparison

Both RTL and generic netlists run 64 identical 8×8 matrix products: 65,536 FLOPs. V3 alternates partial output blocks across its two buffers, retaining C across K chunks. V4 computes each matrix in one job. Counts include initial loading, starts, computation and final draining through independent 32-bit read/write ports. Host execution and external bus latency are excluded.

| Metric | V3 | V4 |
| --- | ---: | ---: |
| Native tile / lanes | 4×4 / 4 | 8×8 / 8 |
| Matrix storage, bytes | 256 | 1,024 |
| Jobs | 512 | 64 |
| Total clocks | 10,272 | 4,480 |
| Host traffic, bytes | 49,152 | 32,768 |
| Projected MFLOPS at 40 MHz | 255.20 | 585.14 |
| Generic logic cells | 24,316 | 61,938 |
| Physical standard-cell area, µm² | 251,834 | 716,253 |

Banked B and simplified packed writes reduce generic cells 8.8% versus the first V4 implementation, preserving throughput. Supplying fresh FP32 C per job increases the native 64-tile overlapped benchmark to 8,387 clocks and 49,152 bytes (312.56 projected MFLOPS).

V4 meets the 25 ns constraint with +7.49368 ns setup and +0.33452 ns hold slack, with zero final routing, antenna and timing/electrical violations. KLayout validates the 1,358.125 µm square GDS. V3 physical area comes from its recorded earlier layout: V4 uses 2.84× that area. The 25% repair margins resolve an extracted-load overload found with 10% margins.

## Correctness evidence

RTL and generic-netlist simulations each pass 1,833 functional/training jobs plus 320 timed jobs (448 with the later gemm16/32 benchmarks). The suite covers:

| Check | Result |
| --- | --- |
| Structural gate | Eight unsigned 8×8→16 multipliers, BF16 A/B, FP32 C |
| Numerical reference | Bit comparison against FP32 `std::fma` with expanded BF16 inputs |
| Encoding sweep | All 65,536 BF16 A encodings in 1,024 tiles; bounded B/C combinations |
| Exceptions | Special values, subnormals, rounding, each lane and combined flags |
| Protocol | Both buffers, overlapping transfers, malformed writes, all 67 abort phases |
| Training | BF16 forward/dW/dX/SGD, FP32 master weights; loss 57.3555→0.000779807 after 48 updates |
| Padded GEMM | 7×13 by 13×9, four hardware jobs with retained C |
| Regressions before the V2/V3 retirement (history) | V3 RTL/netlist: 4,873 jobs plus 768 timed jobs each; V2 RTL/netlist and scalar RTL: 405 jobs each; MAC8 simulation/synthesis |

The encoding sweep is not exhaustive arithmetic conformance. Training covers one dense-layer workload.

## Evidence and reproduction

Use the [V4 commands](training-v4.md). Local logs: `build/training_v4_{sim,netlist_sim,layout,gds_inspection}.log`, `build/training-v4_{structure,synth}.log`; historical regression logs: `build/training_v4_{v3,v3_netlist,v2,v2_netlist,scalar}_regression.log`. Metrics and source hashes: `build/training_v4_comparison.json`. The shared benchmark prints `mode=gemm8_stream`.

Power and energy are unmeasured. [Fabrication gates](physical-flow.md) remain separate.
