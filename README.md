# OpenROAD Chip Lab

Learn digital ASIC design by taking one small block from Verilog through simulation, synthesis, timing constraints, and SKY130HD physical layout.

```sh
make sim
make synth
make layout
```

The first design is an 8-bit multiply-accumulate core.

| Guide | Purpose |
| --- | --- |
| [Setup](docs/setup.md) | Tools and first build |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
