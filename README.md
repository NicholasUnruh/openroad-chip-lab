# OpenROAD Chip Lab

Learn digital ASIC design by taking one small block from Verilog through simulation, synthesis, timing constraints, and SKY130HD physical layout.

```sh
make sim
make synth
make layout
```

The first design is an 8-bit multiply-accumulate core. See its [interface contract](docs/mac8.md), [setup](docs/setup.md), [learning path](docs/learning-path.md), and [physical flow](docs/physical-flow.md).

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
