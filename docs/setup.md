# Setup

## Tools

| Tool | Used by |
| --- | --- |
| Icarus Verilog | `make sim` (MAC8) |
| Verilator 5 | Tile, dot-product and walkthrough simulations |
| Yosys | `make synth`, structural checks, netlists for simulation and layout |
| Python 3, C++17 compiler | Structural checks, testbenches, numerics emulation, cycle model |
| [Bazelisk](https://github.com/bazelbuild/bazelisk/releases) | `make layout*` and `make ppa*` |
| [KLayout](https://www.klayout.de/build.html) | GDS output and inspection |

Check them before a build:

```sh
command -v iverilog verilator yosys python3 g++ bazelisk klayout
bazelisk version
klayout -v
```

## Physical flow

`make sim` and `make synth` need no OpenROAD build. `make layout` clones the pinned [bazel-orfs](https://github.com/The-OpenROAD-Project/bazel-orfs) checkout into `${XDG_CACHE_HOME:-~/.cache}/openroad-chip-lab/bazel-orfs` (override with `ORFS_TOOL_DIR`), then builds OpenROAD, Yosys and their dependencies from source on first use. Its GDS target needs the real KLayout binary on `PATH`.
