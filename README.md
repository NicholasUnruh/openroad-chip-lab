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
| [Fused dot-product V6](docs/training-v6.md) | 32 MACs/clock, SRAM scratchpad; [interface](docs/training-v6-interface.md), [results](docs/training-v6-results.md) |
| [Numerics study](docs/numerics-study.md) | ASAP7 BF16/FP8/MX/INT8 datapath comparison; [results](docs/numerics-results.md) |
| [BF16 V4](docs/training-v4.md) | Default tile; [interface](docs/training-v4-interface.md), [results](docs/training-v4-results.md), [hands-on lab](docs/v4-hands-on.md) |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |
| [Storage recovery](docs/storage-recovery.md) | Offloaded compiled caches |

Retired tiles (scalar FP32, V2, V3, V5) live at tag `archive/pre-cleanup-2026-10-30`, with copies in `~/projects/ARCHIVE/openroad-chip-lab-old-tiles-2026-10-30/`.

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
