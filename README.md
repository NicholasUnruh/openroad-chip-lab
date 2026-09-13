# OpenROAD Chip Lab

Build digital ASICs from Verilog through simulation, synthesis, and SKY130HD layout. The default accelerator is a four-lane BF16 matrix tile with FP32 accumulation and two buffers for overlapping transfers and computation.

```sh
make sim-training
make sim-training-v3-netlist
make layout-training-v3
```

The introductory 8-bit multiply-accumulate core remains available through `make sim`, `make synth`, and `make layout`.

| Guide | Purpose |
| --- | --- |
| [Setup](docs/setup.md) | Tools and first build |
| [BF16 accelerator](docs/training-v3.md) | Native arithmetic, buffer protocol and commands |
| [BF16 results](docs/training-v3-results.md) | Verification, transfer benchmarks and physical results |
| [Four-lane FP32 baseline](docs/training-v2.md) | Previous architecture |
| [Scalar FP32 baseline](docs/training-tile.md) | Original training workload |
| [MAC8 interface](docs/mac8.md) | RTL behavior |
| [Learning path](docs/learning-path.md) | Fundamentals and exercises |
| [Physical flow](docs/physical-flow.md) | Layout, results, fabrication gates |

**Fabrication gate:** the layout is a core. A chip needs IO, power, foundry DRC/LVS and signoff before fabrication.
