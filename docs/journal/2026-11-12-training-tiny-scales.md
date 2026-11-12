# Tiny block scales stop MXFP8 training, and an exact fallback

## Question

The MXFP8 GPT-2 run stopped at step 126 with "block scales left the exact FP32 range". It stopped there again on every one of five restarts (at step 126 or 137). What triggers this, and how can the run continue without giving up bit-exactness?

> **Emulation scope.** Only the transformer-block Linear GEMMs (forward and both backward GEMMs) are emulated. Attention QKᵀ/AV, the embeddings and the tied LM head stay in BF16 (arXiv 2509.25149). No result here is full-network low precision.

## Concept

**Why the fast kernel needs a range guard.** The kernel adds each exact integer block sum S to the FP32 accumulator as S · f_a · f_b, where f = 2^(shared + grid) is the block's scale times the element grid step. These FP32 products are exact only while their smallest bit, f_a · f_b, is at least 2^−149, FP32's smallest subnormal. Below that, the product itself rounds, so the result is rounded twice instead of once.

**How small block scales get.** For E4M3, f = 2^(⌊log2 amax⌋ − 8 − 9) ≈ amax · 2^−17, so the guard fails once amax_a · amax_b < about 2^−115 ≈ 2 × 10⁻³⁵. Activations never get that small. Gradients can: a block of 32 gradient values that are all nearly zero has a correspondingly tiny scale. Which layer produced them was not isolated. The kernel checks the smallest factor of each operand before every GEMM, a conservative test.

**What the hardware does.** The chip lab's unit computes S · 2^(sa+sb−254) + c exactly and rounds once, even when the result is a subnormal FP32 number. An emulator that rounds twice would be wrong only in the last bit of values near 10⁻³⁸. Bit-exact is a yes-or-no property, though, so the earlier code stopped the run rather than continue inexactly.

**An exact fallback.** In float64 the same computation is exact at any scale. Grid integers below 2^18 multiply to below 2^36, and 32-term block sums stay below 2^41, so a float64 matrix product of grid integers is exact in any summation order. float64's exponent range covers 2^−272, the smallest possible E4M3 factor product. Adding with round-to-odd in float64 and then converting to FP32 gives one correct rounding, including into the subnormal range. This is the CPU reference GEMM already cross-checked against the C++ units; it now runs on the GPU too.

## What I tried

1. The CPU reference GEMM (`mx.gemm_reference`) now uses per-block float64 matrix products instead of int64 batched products, so it runs on the GPU. `kernel.gemm_packed` sends any MX GEMM that fails the guard there instead of stopping. The trainer logs the number of such GEMMs per step.
2. A new cross-check distribution, `tiny`, scales rows by 2^−100 … 2^−40. Over 20 MXFP8 test GEMMs (64 × 64 outputs each), 66,483 block pairs fall below 2^−149, 35% of outputs are subnormal and 32% are exactly zero.
3. MXFP8 training was resumed from the failed run's step-107 checkpoint up to step 145, past both failure points.

## Output

`docs/journal/data/training-xcheck-gpu-tiny.txt` (GPU kernel with fallback against the C++ references, all eight distributions including `tiny`):

```
mxfp8           589344 elements    53900 outputs checked
mxfp8_e5m2      688704 elements        0 outputs checked
mxfp6_e3m2      503584 elements    41545 outputs checked
mxfp6_e2m3      531520 elements    45956 outputs checked
mxfp4           553280 elements    50398 outputs checked
nvfp4           509536 elements    36488 outputs checked
PASS
```

MXFP8 probe, steps 108–145 (`docs/journal/data/training-mxfp8-fallback-probe.jsonl`): the fallback ran in 4 of 38 steps, with 1, 2, 2 and 1 GEMMs out of about 2,300 per step. Throughput was 7,440 tokens/s, against 7,518 tokens/s for the original run over the same steps, a cost of about 1%.

## Result

Block-factor products below 2^−149, from blocks with amax products below about 2 × 10⁻³⁵, appear in MXFP8 training of GPT-2 after about 130 steps. They push a few GEMMs per step outside the range where the FP32 kernel is exact. Those GEMMs now run in exact float64 on the GPU, which matches the C++ units bit for bit (zero differences, including subnormal outputs). Training continues at about 1% lower throughput, and the emulation stays bit-exact everywhere. NVFP4 cannot hit this case, because its E4M3 block scales are at least 2^−9.

A side observation: resuming from the same checkpoint does not reproduce the same steps bit for bit. The failure step varied between 126 and 137 across restarts, so some GPU kernel in the step is not deterministic (not isolated further). Runs are therefore reproducible statistically, not bit for bit across a resume.

## Next

MXFP8 continues from its step-107 checkpoint after the other formats, since the run supervisor skips a run after five failures until it restarts. MXFP4 is running and NVFP4 follows. Report each format's final loss against BF16's 3.7283 as runs finish.
