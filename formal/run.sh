#!/bin/sh
# Bounded proofs of every MX/NVFP4 block dot-product unit against formal/mx_dot32_formal.v:
# per element type, MODE 0 (products), MODE 2 (specials) and one MODE 1 (alignment) case per c
# exponent across the range where c and the block sum interact, with 26 binades of margin each side.
# Writes build/formal/summary.txt (one line per case) and exits 1 unless every case passes.
# usage: [STRIDE=n] [CASE_TIMEOUT=s] formal/run.sh [jobs] [elem...]   (needs SymbiYosys and Bitwuzla on PATH)
# STRIDE=n checks every n-th c exponent plus both ends of the range.
# CASE_TIMEOUT is the per-case limit in seconds (default 3600); a case that hits it reports TIMEOUT.
set -eu
command -v sby bitwuzla > /dev/null || { echo "error: sby and bitwuzla must be on PATH" >&2; exit 2; }
lab=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out="$lab/build/formal"

if [ "${1:-}" = --case ]; then
    elem=$2 mode=$3 cexp=$4
    name="e${elem}_m${mode}_c${cexp}"
    cat > "$out/$name.sby" <<EOF
[options]
mode bmc
depth 9

[engines]
smtbmc bitwuzla

[script]
read_verilog -formal mx_dot32_fp32.v fp32_round.v
read_verilog -formal -sv mx_dot32_formal.v
chparam -set ELEM $elem -set MODE $mode -set CEXP $cexp mx_dot32_formal
prep -top mx_dot32_formal

[files]
$lab/rtl/mx_dot32_fp32.v
$lab/rtl/fp32_round.v
$lab/formal/mx_dot32_formal.v
EOF
    start=$(date +%s)
    rm -rf "${out:?}/$name"  # a stale status file would read as this run's result
    timeout "${CASE_TIMEOUT:-3600}" sby -f -d "$out/$name" "$out/$name.sby" > /dev/null 2>&1 || true
    status=$(cut -d' ' -f1 "$out/$name/status" 2>/dev/null || echo TIMEOUT)
    printf '%s elem=%s mode=%s cexp=%s seconds=%s\n' "$status" "$elem" "$mode" "$cexp" \
        "$(($(date +%s) - start))"
    exit 0
fi

jobs=${1:-4}
[ "$#" -gt 0 ] && shift
elems=${*:-0 1 2 3 4 5}
mkdir -p "$out"
# c exponent offsets (from 2^0) where an aligned c meets the products of two elements at unit
# scale: lowest product LSB - 25 to highest two-term sum MSB + 25.
range() {
    case $1 in
        0) echo "-43 44" ;;  # E4M3: product LSBs -18..10, sum below 2^20
        1 | 5) echo "-27 32" ;;  # E2M1 (and NVFP4 at unit block scales): -2..2, below 2^7
        2) echo "-57 58" ;;  # E5M2: -32..26, below 2^33
        3) echo "-33 34" ;;  # E3M2: -8..2, below 2^9
        4) echo "-31 32" ;;  # E2M3: -6..-2, below 2^7
    esac
}
for elem in $elems; do
    echo "$elem 0 0"
    echo "$elem 2 0"
    set -- $(range "$elem")
    d=$1
    while [ "$d" -le "$2" ]; do
        echo "$elem 1 $((127 + d))"
        [ "$d" -lt "$2" ] && [ $((d + ${STRIDE:-1})) -gt "$2" ] && echo "$elem 1 $((127 + $2))"
        d=$((d + ${STRIDE:-1}))
    done
done | xargs -P "$jobs" -L 1 "$0" --case | tee "$out/summary.txt"
total=$(wc -l < "$out/summary.txt")
passed=$(grep -c '^PASS' "$out/summary.txt" || true)
printf 'formal: %s of %s cases pass\n' "$passed" "$total"
[ "$passed" -eq "$total" ]
