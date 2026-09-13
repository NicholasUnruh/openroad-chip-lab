# OpenROAD Chip Lab

Build digital ASICs from Verilog through simulation, synthesis, and SKY130HD layout. The latest training prototype is a four-lane, pipelined 4×4 FP32 matrix tile supporting forward passes, gradients, and SGD updates.

```sh
make sim-training-v2
make sim-training-v2-netlist
make layout-training-v2
```

The introductory 8-bit multiply-accumulate core remains available through `make sim`, `make synth`, and `make layout`.

| Guide | Purpose |
| --- | --- |
| [Setup](docs/setup.md) | Tools and first build |
| [Four-lane training tile](docs/training-v2.md) | Architecture, commands and measured results |
| [Scalar FP32 baseline](docs/training-tile.md) | Shared interface and training workload |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
