#!/bin/sh
# Quick SKY130HD standard-cell area estimate (ABC mapping, no placement); the macro is excluded.
# usage: tools/cell_area.sh <top> <liberty> <verilog files...>
set -eu
top=$1 lib=$2
shift 2
yosys -q -p "read_verilog -lib tb/sky130_sram_1rw1r_128x256_8.v; read_verilog -Ithird_party/hardfloat -Ithird_party/hardfloat/RISCV $*; \
  synth -top $top -flatten; dfflibmap -liberty $lib; abc -liberty $lib; opt_clean; tee -o /dev/stdout stat -liberty $lib" \
  | awk -v top="$top" '/Chip area/{print top, $NF}'
