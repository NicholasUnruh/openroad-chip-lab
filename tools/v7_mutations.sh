#!/bin/sh
# Mutation checks for training_tile_v7: each line plants one sequencer bug and the GEMM regression
# (tb/mx_tile_test.cpp) must fail. Prints one verdict per mutation; exits 1 if any survives.
# usage: tools/v7_mutations.sh   (about 5 minutes)
set -u
lab=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work="$lab/build/v7-mutations"
survivors=0

mutate() {
    name=$1
    dir="$work/$name"
    rm -rf "$dir"
    mkdir -p "$dir"
    sed "$2" "$lab/rtl/training_tile_v7.v" > "$dir/training_tile_v7.v"
    if cmp -s "$dir/training_tile_v7.v" "$lab/rtl/training_tile_v7.v"; then
        printf '%s: mutation did not apply\n' "$name"
        survivors=$((survivors + 1))
        return
    fi
    verilator --cc --exe --build -j 4 --top-module training_tile_v7 --Mdir "$dir/obj" \
        "$dir/training_tile_v7.v" "$lab/rtl/mx_dot32_fp32.v" "$lab/rtl/fp32_round.v" \
        "$lab/tb/sky130_sram_1rw1r_128x256_8.v" "$lab/tb/mx_tile_test.cpp" \
        -CFLAGS "-std=c++17 -O2 -I$lab/tb" > "$dir/build.log" 2>&1 || {
        printf '%s: build failed\n' "$name"
        survivors=$((survivors + 1))
        return
    }
    failure=$(timeout 600 "$dir/obj/Vtraining_tile_v7" 2>&1 | grep -m1 '^FAIL')
    if [ -n "$failure" ]; then
        printf '%s: caught (%s)\n' "$name" "$failure"
    else
        printf '%s: NOT CAUGHT\n' "$name"
        survivors=$((survivors + 1))
    fi
}

mutate early_start "s/load_index >= 6'd20/load_index >= 6'd19/"
mutate unit_latency 's/localparam LAT = 7;/localparam LAT = 6;/'
mutate scale_block 's/+ {6.b0, lk};/;/'
mutate short_load "s/load_index == 6'd33) load_active/load_index == 6'd32) load_active/"
mutate bank_guard 's/ \&\& !(nxt_k0 \&\& pending\[nxt_cbank\]);/;/'
mutate no_back_to_back "s/(!load_active || load_index == 6'd33);/!load_active;/"
mutate b_scale_select 's/sb_reg\[64\*bi+8\*(4\*u+x) +: 8\]/sb_reg[64*bi+8*(4*u) +: 8]/'
[ "$survivors" -eq 0 ] || exit 1
