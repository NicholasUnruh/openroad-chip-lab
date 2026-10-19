# Numerics PPA study at 7 nm

**Question:** which multiply-accumulate datapath gives the most training-usable throughput per mm² and per joule at an advanced node?

At frontier scale, MAC arithmetic, not lane count, sets compute density and energy. V6 showed a fused BF16 dot product costs 5.8× less SKY130 area per MAC than chained FMAs. This study repeats that comparison at ASAP7 and extends it to narrower formats.

## Design points

| Point | Inputs | Reduction | Accumulator |
| --- | --- | --- | --- |
| `fma_bf16` | BF16 | One FMA per clock (V3–V5 lane) | FP32 |
| `dot8_bf16` | BF16 | 8-term fused dot product (V6) | FP32 |
| `dot8_e4m3`, `dot8_e5m2` | OCP FP8 | 8-term fused dot product | FP32 |
| `dot32_mxfp8` | OCP MXFP8 (E4M3, shared E8M0 scale per 32) | 32-term block dot product | FP32 |
| `dot32_mxfp4` | OCP MXFP4 (E2M1, shared E8M0 scale per 32) | 32-term block dot product | FP32 |
| `dot8_int8` | INT8 | 8-term integer dot product | INT32 (inference reference) |

Frame-width variants of `dot8_bf16` (24–40 bits) measure how much accumulator precision costs.

## Method

1. **RTL** for each point, with a bit-exact C++ specification model and the existing verification harness: a randomized multi-distribution sweep, an exact-sum oracle, and mutation checks.
2. **PPA at ASAP7** through ORFS to route. Each unit sits behind input and output registers in a fixed harness. Report area per MAC, fmax from routed slack, and energy per MAC from OpenSTA `report_power` with switching activity from simulated realistic vectors.
3. **Accuracy.** Compute the error of GEMMs against an FP64 reference across data distributions; then train a fixed small model end to end with each datapath emulated bit-exactly, comparing loss curves with FP32.
4. **Ranking.** Compute effective throughput per mm² and per joule, weighted by whether the format trains to within a set loss tolerance of FP32.

## Endpoint

One results table per metric across all design points, plus the accuracy result, in `docs/numerics-results.md`. ASAP7 is predictive. Compare design points relative to each other; never quote these as foundry figures.

## Milestones

| Milestone | Scope |
| --- | --- |
| M1 | ASAP7 flow; `fma_bf16` vs `dot8_bf16` area, fmax and energy |
| M2 | FP8 and INT8 dot products |
| M3 | MXFP8 and MXFP4 block-scaled units |
| M4 | GEMM error and training-accuracy emulation; final ranking |
