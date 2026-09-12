# Setup

On this Ubuntu workstation, Icarus Verilog and Yosys were already installed. Bazelisk 1.29.0 and KLayout 0.30.12 are available in `~/.local/bin` (verified 2026-09-12). The KLayout wrapper points to a user-local package under `~/.cache/openroad-chip-lab/`.

Check the tools before a build:

```sh
command -v iverilog yosys bazelisk klayout
bazelisk version
klayout -v
```

`make sim` and `make synth` need no OpenROAD build. `make layout` clones the pinned [bazel-orfs](https://github.com/The-OpenROAD-Project/bazel-orfs) checkout under `~/.cache/openroad-chip-lab/bazel-orfs`, then builds OpenROAD, Yosys and dependencies from source on first use. Its local GDS target requires the real KLayout binary.

For another machine, get Bazelisk from its [official releases](https://github.com/bazelbuild/bazelisk/releases) and KLayout from the [official downloads](https://www.klayout.de/build.html). Run the tool checks above before `make layout`.
