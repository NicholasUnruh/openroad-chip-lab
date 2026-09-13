# OpenROAD Chip Lab

Build digital ASICs from Verilog through simulation, synthesis, and SKY130HD layout. The training prototype is a 4×4 FP32 matrix tile supporting forward passes, gradients, and SGD updates.

```sh
make sim-training
make sim-training-netlist
make layout-training
```

The introductory 8-bit multiply-accumulate core remains available through `make sim`, `make synth`, and `make layout`.

| Guide | Purpose |
| --- | --- |
| [Setup](docs/setup.md) | Tools and first build |
| [FP32 training tile](docs/training-tile.md) | Interface, training workload, verification and limits |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
