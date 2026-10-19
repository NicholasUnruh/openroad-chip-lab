# OpenROAD Chip Lab

Build digital ASICs from Verilog through simulation, synthesis, and SKY130HD layout. The default accelerator is an eight-lane 8×8 BF16 matrix tile with FP32 accumulation, banked operands and two buffers for overlapping transfers and computation.

```sh
make sim-training
make sim-training-v4-netlist
make layout-training-v4
```

The introductory 8-bit multiply-accumulate core remains available through `make sim`, `make synth`, and `make layout`.

| Guide | Purpose |
| --- | --- |
| [Setup](docs/setup.md) | Tools and first build |
| [Fused dot-product V6](docs/training-v6.md) | 32 MACs/clock, SRAM scratchpad; [results](docs/training-v6-results.md) |
| [Numerics study](docs/numerics-study.md) | ASAP7 BF16/FP8/MX/INT8 datapath comparison |
| [BF16 V4](docs/training-v4.md) | Default tile; [results](docs/training-v4-results.md), [hands-on lab](docs/v4-hands-on.md) |
| [V3](docs/training-v3.md), [V2](docs/training-v2.md) baselines | Earlier four-lane BF16 and FP32 tiles |
| [Scalar FP32 baseline](docs/training-tile.md) | Original training workload |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |
| [Storage recovery](docs/storage-recovery.md) | Offloaded compiled caches |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
