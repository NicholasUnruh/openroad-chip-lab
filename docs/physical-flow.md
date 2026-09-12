# Physical flow

The first target uses OpenROAD Flow Scripts through a pinned bazel-orfs checkout with the open SKY130HD platform. `make layout` builds `mac8_final` and converts it to GDS with real KLayout. The stages are synthesis, floorplan, placement, clock-tree synthesis, routing and final physical reports. See the [upstream flow tutorial](https://openroad-flow-scripts.readthedocs.io/en/latest/tutorials/FlowTutorial.html) for each stage.

Run from this repo:

```sh
make sim
make synth
make layout
klayout build/layout/mac8.gds
cd ~/.cache/openroad-chip-lab/bazel-orfs
bazelisk run //chip-lab:mac8_final open_final
```

`make layout` copies the GDS to `build/layout/mac8.gds`, the final timing report to `build/layout/timing.rpt`, and the router's DRC report to `build/layout/router_drc.rpt`. In the OpenROAD shell, use `report_checks -path_delay max` to examine a critical path. The tool checkout's `bazel-bin/chip-lab/` contains per-stage results and reports. Set `ORFS_TOOL_DIR` if the checkout lives elsewhere.

The first SKY130HD run on 2026-09-12 used a 20 ns clock at the platform's typical 25 C, 1.8 V corner. Its final report showed 13.67 ns worst setup slack and 0.50 ns hold slack; detailed routing ended at zero router DRC errors and zero antenna violations. KLayout opened the 752,896-byte GDS as a 101.53 x 101.53 um `mac8` top cell. These are flow results under this lab's constraints, not independent signoff.

`mac8_gds` explicitly selects a wrapper for the real KLayout executable; Bazel's strict action environment may omit `~/.local/bin` from `PATH`. The default bazel-orfs mock writes a dummy file, so keep this override in `tools/orfs/BUILD.bazel`.

The flow produces a core layout, not a complete chip. A fabrication package also needs a pad frame and ESD, power integration, foundry-specific DRC/LVS, timing and signal-integrity signoff, and acceptance by a fabrication service. The [open SKY130 PDK](https://skywater-pdk.readthedocs.io/en/main/) describes itself as an experimental preview suitable for test-chip work, not production signoff. Keep each gate distinct.

For an eventual shared-shuttle test chip, [Tiny Tapeout's local hardening guide](https://tinytapeout.com/guides/local-hardening/) is a possible route. Its template, pinout and acceptance checks are separate work; this lab core is not a submission package.
