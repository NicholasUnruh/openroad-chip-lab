# Training prototype results

Verified 2026-09-12 against RTL commit `c4c35f6`: the FP32 training tile passes numerical and synthesized-netlist tests and completes SKY130HD routing. It remains a core without pads or independent fabrication signoff.

| Check | Result | Local evidence |
| --- | --- | --- |
| RTL simulation | 402 jobs pass | `build/training_sim.log` |
| Generic netlist simulation | Same 402 jobs pass | `build/training_netlist_sim.log` |
| Dense-layer training | 32 SGD updates; loss 3.85547 → 0.0000828033 | Both simulation logs |
| Gradients | dW and dX match finite differences within 0.003 | Both simulation logs |
| Generic synthesis | 11,535 cells; zero structural problems | `build/training_synth.log` |
| Physical target | 100 ns / 10 MHz; typical 25°C, 1.8 V | `constraints/training_tile.sdc`, physical log |
| Final setup / hold slack | +63.2336 ns / +0.467235 ns | `build/layout-training/metrics.json` |
| Setup, hold, slew, capacitance, fanout violations | Zero | `build/layout-training/timing.rpt` |
| Router DRC / antenna violations | Zero / zero | `build/layout-training/router_drc.rpt`, physical log |
| Standard-cell area | 125,031 µm² | `build/layout-training/metrics.json` |
| GDS bounding box | 574.605 × 574.605 µm | `build/training_gds_inspection.log` |

The timing report estimates 27.20 MHz maximum under this model; only the 10 MHz constraint was targeted. One FMA executes per compute clock, giving 64 compute clocks per tile. Transfer and host work are additional. No workload-based power or energy benchmark was performed.

## Reproduce

```sh
make sim-training
make sim-training-netlist
make layout-training
```

The recorded physical run reused `ORFS_TOOL_DIR=/tmp/bazel-orfs-inspect`, whose checkout matches the default pinned bazel-orfs commit `a25cf9abcdc753b25e5e2d28487844556d8e6db2`. The default persistent checkout requires its own first build; the temporary cache is optional.

Physical log: `build/training_layout_cached.log`. KLayout 0.30.12 independently read the resulting GDS and found one top cell, `training_tile`. The generated GDS SHA256 is `72eead92e3c402eaaca5f59fdac4dce8abfe1e088e330f0d2b49d022b4484016`.

The original MAC8 simulation, synthesis and cached layout also pass. Imported HardFloat source bytes match the upstream archive; [checksums and provenance](../third_party/hardfloat/README.md) are retained. Floating-point tests are bounded regression evidence, not exhaustive IEEE conformance or formal equivalence.
