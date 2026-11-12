#!/bin/sh
# Routed ASAP7 power for a PPA harness from simulated switching activity.
# Simulates the routed netlist (zero-delay Liberty cell models) and the Yosys netlist on the same
# realistic stimulus from tools/ppa_activity.py, requires identical outputs after warm-up, then
# annotates the routed design with the VCD and runs OpenSTA report_power with extracted parasitics.
# Writes build/ppa/<harness>/power.txt.
# usage: tools/ppa_power.sh <harness>
set -eu
[ "$#" -eq 1 ] || { printf 'usage: tools/ppa_power.sh <harness>\n'; exit 2; }
design=$1
lab_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tool_dir=${ORFS_TOOL_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/openroad-chip-lab/bazel-orfs}
results="$tool_dir/bazel-bin/chip-lab/results/asap7/$design/base"
platform=$(ls -d "$HOME"/.cache/bazel/_bazel_*/*/external/+orfs_repositories+orfs/flow/platforms/asap7 2>/dev/null | head -1)
openroad=$(find -H "$tool_dir/bazel-out" -path '*openroad+/openroad' -type f | head -1)
[ -s "$results/6_final.odb" ] || { printf 'error: no routed %s\nhelp: run make ppa PPA_TOP=%s\n' "$design" "$design"; exit 1; }
out_dir="$lab_dir/build/ppa/$design"
mkdir -p "$out_dir/reference"
lib=$platform/lib/NLDM
libs="$lib/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz $lib/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz
      $lib/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz $lib/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz
      $lib/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib"
# shellcheck disable=SC2086
python3 "$lab_dir/tools/liberty_cells.py" "$out_dir/cells.v" $libs
python3 "$lab_dir/tools/ppa_activity.py" "$design" "$out_dir"
python3 "$lab_dir/tools/ppa_activity.py" "$design" "$out_dir/reference" > /dev/null
iverilog -g2012 -o "$out_dir/gate.vvp" "$out_dir/testbench.v" "$out_dir/cells.v" "$results/6_final.v"
iverilog -g2012 -o "$out_dir/reference.vvp" "$out_dir/reference/testbench.v" "$lab_dir/build/ppa/$design.v"
vvp -n "$out_dir/gate.vvp" > /dev/null
vvp -n "$out_dir/reference.vvp" > /dev/null
# Outputs logged from the warm-up clock on must match bit for bit.
if [ "$(tail -n +17 "$out_dir/outputs.txt")" != "$(tail -n +17 "$out_dir/reference/outputs.txt")" ]; then
    printf 'error: routed %s differs from its Yosys netlist on the power stimulus\n' "$design"
    exit 1
fi
{
    for f in $libs; do printf 'read_liberty %s\n' "$f"; done
    cat <<TCL
read_db $results/6_final.odb
read_sdc $results/6_final.sdc
read_spef $results/6_final.spef
read_vcd -scope ppa_tb/dut $out_dir/activity.vcd
report_activity_annotation
report_power
exit
TCL
} > "$out_dir/power.tcl"
"$openroad" -no_init -exit "$out_dir/power.tcl" > "$out_dir/power.txt" 2>&1
awk '/^Total/{print "total_power_w=" $5}' "$out_dir/power.txt"
