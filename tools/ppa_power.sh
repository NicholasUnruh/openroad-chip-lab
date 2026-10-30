#!/bin/sh
# Routed ASAP7 power for a PPA harness with input activity from tools/ppa_activity.py.
# Reads the final ODB and extracted SPEF from the bazel-orfs run; writes build/ppa/<harness>/power.txt.
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
mkdir -p "$out_dir"
python3 "$lab_dir/tools/ppa_activity.py" "$design" "$out_dir/activity.tcl"
lib=$platform/lib/NLDM
cat > "$out_dir/power.tcl" <<TCL
foreach f {AO_RVT_FF_nldm_211120.lib.gz INVBUF_RVT_FF_nldm_220122.lib.gz OA_RVT_FF_nldm_211120.lib.gz
           SIMPLE_RVT_FF_nldm_211120.lib.gz SEQ_RVT_FF_nldm_220123.lib} {
    read_liberty $lib/asap7sc7p5t_\$f
}
read_db $results/6_final.odb
read_sdc $results/6_final.sdc
read_spef $results/6_final.spef
source $out_dir/activity.tcl
report_power
exit
TCL
"$openroad" -no_init -exit "$out_dir/power.tcl" > "$out_dir/power.txt" 2>&1
awk '/^Total/{print "total_power_w=" $5}' "$out_dir/power.txt"
