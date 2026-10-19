# Fused dot-product accelerator

`training_tile_v6` keeps the [V5 host interface](training-v5.md) and scratchpad but replaces the eight chained FMA lanes with four fused 8-term dot-product units: 32 BF16 multiply-accumulates per clock, four times V4 and V5.

```sh
make sim-dot8
make sim-training-v6
make sim-training-v6-netlist
make layout-training-v6
```

## Why 32 MACs per clock

An 8×8×8 job needs eight A rows and eight B rows; the scratchpad read port returns one 128-bit row per clock, so it supplies one job per 16 clocks: 512 MACs / 16 = 32. The 32-bit host ports give the same bound for a 32³ GEMM (8 KB in and 4 KB out against 65,536 FLOPs). More units would idle.

## Dot-product arithmetic

Each unit computes `c + Σ a[k]·b[k]` over k = 0–7, with BF16 `a`, `b` and FP32 `c`, and rounds once. Results differ from eight chained FMAs.

1. Products are exact 16-bit significands; subnormal inputs use exponent 1 without a hidden bit.
2. Finite nonzero terms (eight products and `c`) are left-justified in a 32-bit frame below the largest term's top bit `T`. Products occupy 16 bits, `c` 24 bits. Terms are shifted right by their distance from `T`; shifted-out bits are truncated toward zero.
3. The signed frame values add exactly.
4. The sum rounds once to FP32, nearest-even, from frame bits only; FP32 subnormals round at 2^-149. Overflow returns infinity.
5. A zero sum is +0, unless every term is zero and every term sign is negative (then −0).

Total truncation error is below 9·2^(T−32). A nonzero sum that rounds to zero keeps its sign.

| Flag | Condition |
| --- | --- |
| Invalid (16) | Signaling NaN input, ∞·0, or opposite infinities |
| Overflow (4) | Rounded exponent above FP32 range |
| Underflow (2) | Nonzero frame sum below 2^-126 before rounding, and inexact |
| Inexact (1) | Rounding discards bits or step 2 truncates nonzero bits |

NaNs return the canonical `0x7fc00000`. A valid infinity propagates exactly with no flags.

## Datapath

Each unit owns output columns `u` and `u+4`. Per clock, one A row is broadcast and each unit reads its B column and accumulator; results write back six clocks later. A job takes 16 clocks; the loader fills the other A/B buffer meanwhile. Two C banks let the host drain a finished tile while the next computes. Output tiles run row-major with K innermost; the first K tile starts from +0.

`tb/dot8_model.h` implements this specification and an exact-sum oracle; the unit is `rtl/fp_dot8_fp32.v` with `FORMAT=0`. See [results](training-v6-results.md). Wherever no term is truncated, specification results must equal the correctly rounded exact sum.
