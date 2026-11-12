#!/bin/sh
# Render OpenROAD placement-density, routing-congestion and clock-net views of a routed design and
# dump every endpoint's setup/hold slack, from the bazel-orfs results of `make layout-training-v6`.
# usage: tools/physical_views.sh [training_tile_v6]
# Writes build/renders/<design>/{density,congestion,clock}.png and endpoints_{setup,hold}.txt.
# Needs xvfb-run; the GUI build of OpenROAD (openroad-qt) is part of the bazel-orfs toolchain.
set -eu
design=${1:-training_tile_v6}
lab_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tool_dir=${ORFS_TOOL_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openroad-chip-lab/bazel-orfs}
build_dir=$lab_dir/build
# A worktree reads the main checkout's build/.
while [ ! -d "$build_dir/layout" ] && [ "$build_dir" != /build ]; do
    build_dir=$(dirname "$(dirname "$build_dir")")/build
done
results=$tool_dir/bazel-bin/chip-lab/results/sky130hd/$design/base
platforms=$tool_dir/bazel-bazel-orfs/external/+orfs_repositories+orfs/flow/platforms
openroad=$tool_dir/bazel-out/k8-opt-exec/bin/external/openroad+/openroad-qt
for file in "$results/6_final.odb" "$results/6_final.spef" "$openroad"; do
    [ -e "$file" ] || { printf 'missing %s\nhelp: run make layout-training-v6 first\n' "$file" >&2; exit 1; }
done
out=$build_dir/renders/$design
mkdir -p "$out"
ODB=$results/6_final.odb SDC=$results/6_final.sdc SPEF=$results/6_final.spef OUT=$out \
LIBS="$platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib $platforms/sky130ram/sky130_sram_1rw1r_128x256_8/sky130_sram_1rw1r_128x256_8_TT_1p8V_25C.lib" \
    timeout 1800 xvfb-run -a -s '-screen 0 1920x1080x24' "$openroad" -no_init -no_splash \
    -log "$out/openroad.log" -gui "$lab_dir/tools/physical_views.tcl" > /dev/null 2>&1
grep -q 'ERROR' "$out/openroad.log" && { grep 'ERROR' "$out/openroad.log" >&2; exit 1; }
ls "$out"
