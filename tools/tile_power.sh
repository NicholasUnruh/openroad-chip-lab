#!/bin/sh
# Routed SKY130HD power of a training tile on one of its regression benchmarks.
# Verilates the routed netlist (zero-delay Liberty cell models, the SRAM simulation model) under
# the tile's own GEMM test, which checks every result bit-exactly, dumps the named benchmark as a
# VCD at the 40 MHz target, then annotates the routed design with it and runs OpenSTA
# report_power with extracted parasitics. Writes build/tile-power/<design>-<bench>/power.txt and
# summary.txt (energy per MAC over the whole benchmark, host traffic included).
# usage: tools/tile_power.sh <training_tile_v4|training_tile_v6|training_tile_v7> <bench>
#   benches (the BENCH mode prefix): gemm32 (all), gemm16_pipelined (v6), gemm16x32_pipelined (v7)
set -eu
[ "$#" -eq 2 ] || { sed -n 8,9p "$0"; exit 2; }
design=$1 bench=$2
case "$design" in
    training_tile_v4) test=training_tile_test ;;
    training_tile_v6) test=scratchpad_tile_test ;;
    training_tile_v7) test=mx_tile_test ;;
    *) sed -n 8,9p "$0"; exit 2 ;;
esac
lab_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tool_dir=${ORFS_TOOL_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openroad-chip-lab/bazel-orfs}
results="$tool_dir/bazel-bin/chip-lab/results/sky130hd/$design/base"
platforms=$(ls -d "$HOME"/.cache/bazel/_bazel_*/*/external/+orfs_repositories+orfs/flow/platforms 2>/dev/null | head -1)
openroad=$(find -H "$tool_dir/bazel-out" -path '*openroad+/openroad' -type f | head -1)
[ -s "$results/6_final.odb" ] || { printf 'error: no routed %s\nhelp: run bin/layout %s\n' "$design" "$design"; exit 1; }
cell_lib=$platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
sram=$platforms/sky130ram/sky130_sram_1rw1r_128x256_8/sky130_sram_1rw1r_128x256_8
out_dir="$lab_dir/build/tile-power/$design-$bench"
mkdir -p "$out_dir"

python3 "$lab_dir/tools/liberty_cells.py" "$out_dir/cells.v" "$cell_lib"
macro=
[ "$design" = training_tile_v4 ] || macro="$lab_dir/tb/sky130_sram_1rw1r_128x256_8.v"
# shellcheck disable=SC2086
verilator --cc --exe --build -j 4 --trace --trace-depth 2 -Wno-fatal -Wno-lint -Wno-style \
    --top-module "$design" --Mdir "$out_dir/sim" "$results/6_final.v" "$out_dir/cells.v" $macro \
    "$lab_dir/tb/$test.cpp" -CFLAGS "-std=c++17 -O1 -fno-fast-math -ffp-contract=off -I$lab_dir/tb" \
    > "$out_dir/verilator.log" 2>&1
"$out_dir/sim/V$design" "+vcd=$out_dir/activity.vcd" "+vcd_bench=$bench" > "$out_dir/sim.log"
line=$(grep "^BENCH .*mode=${bench}[ _]" "$out_dir/sim.log" | head -1)
[ -n "$line" ] || { printf 'error: benchmark %s did not run\n' "$bench"; cat "$out_dir/sim.log"; exit 1; }
cycles=$(printf '%s\n' "$line" | sed 's/.* cycles=\([0-9]*\).*/\1/')
flops=$(printf '%s\n' "$line" | sed 's/.* flops=\([0-9]*\).*/\1/')

cat > "$out_dir/power.tcl" <<TCL
read_liberty $cell_lib
read_liberty ${sram}_TT_1p8V_25C.lib
read_db $results/6_final.odb
read_sdc $results/6_final.sdc
read_spef $results/6_final.spef
read_vcd -scope TOP/$design $out_dir/activity.vcd
report_activity_annotation
report_power
exit
TCL
"$openroad" -no_init -exit "$out_dir/power.tcl" > "$out_dir/power.txt" 2>&1
# report_power rows: group, internal, switching, leakage, total (W), percent. The run fails if
# more than 1% of pins fell back to default activity instead of the VCD.
awk -v cycles="$cycles" -v flops="$flops" '
    $1 == "vcd" { annotated = $2 }
    $1 == "unannotated" { unannotated = $2 }
    $1 == "Macro" { macro = $5 }
    $1 == "Total" { total = $5 }
    END {
        seconds = cycles * 25e-9; macs = flops / 2
        printf "cycles=%d macs=%d total_power_w=%s macro_power_w=%s\n", cycles, macs, total, macro
        printf "annotated_pins=%d unannotated_pins=%d\n", annotated, unannotated
        printf "energy_per_mac_pj=%.3f stdcell_energy_per_mac_pj=%.3f\n", \
            total * seconds / macs * 1e12, (total - macro) * seconds / macs * 1e12
        if (unannotated * 100 > annotated + unannotated) { print "error: VCD covers too few pins"; exit 1 }
    }' "$out_dir/power.txt" > "$out_dir/summary.txt" || { cat "$out_dir/summary.txt"; exit 1; }
cat "$out_dir/summary.txt"
