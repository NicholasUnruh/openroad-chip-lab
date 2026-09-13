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
| [BF16 accelerator](docs/training-v4.md) | Native arithmetic, buffer protocol and commands |
| [V4 hands-on lab](docs/v4-hands-on.md) | Trace one job in RTL, waveform and layout |
| [BF16 results](docs/training-v4-results.md) | Verification, transfer benchmarks and physical results |
| [Compact BF16 baseline](docs/training-v3.md) | Previous four-lane architecture |
| [Four-lane FP32 baseline](docs/training-v2.md) | Previous architecture |
| [Scalar FP32 baseline](docs/training-tile.md) | Original training workload |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
