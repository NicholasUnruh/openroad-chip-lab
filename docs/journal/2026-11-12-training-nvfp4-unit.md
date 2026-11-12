# Matching the NVFP4 emulation to the hardware unit

## Question

Does the training emulator's NVFP4 arithmetic match the chip lab's NVFP4 dot-product unit, now that the unit's reference (`nvfp4_dot32`, `nvfp4_quantize`, `nvfp4_chain`) exists?

> **Emulation scope.** Only the transformer-block Linear GEMMs (forward and both backward GEMMs) are emulated. Attention QKᵀ/AV, the embeddings and the tied LM head stay in BF16 (arXiv 2509.25149). No result here is full-network low precision.

## Concept

**What the unit computes.** The NVFP4 unit takes 32 E2M1 elements per operand, as two 16-element blocks, each with its own E4M3 scale. It forms both block sums exactly, multiplies each by its pair of scales, adds both to the FP32 accumulator and rounds **once** per 32 elements. The per-tensor FP32 scales are applied afterwards on the host: one FP32 multiply of the GEMM output by s_dec,a · s_dec,b, where s_dec = 1 / s_enc and s_enc = 6 · 448 / amax_tensor.

**Where a rounding happens matters.** The earlier emulation used a provisional definition that rounded once per 16-element block, two roundings per 32 elements. Each rounding to nearest can move the result by up to half an FP32 unit in the last place (ulp), so the two definitions agree only when the extra rounding happens to be exact.

**Three exact terms, one rounding.** Each block term, block sum × scale_a × scale_b, is exact in FP32: the block sum is below 2^12 grid units and each E4M3 scale has a 4-bit significand, so the product fits in 20 bits. The accumulator update acc + t₀ + t₁ is the same three-term problem the MXFP8 path already solves with the Boldo–Melquiond algorithm (two error-free TwoSums, one addition rounded to odd, one final addition rounded to nearest). The kernel reuses it.

**Element rounding order.** The unit's quantizer computes element = E2M1(x · (1 / (scale · s_dec))) in FP32, three roundings in that order. The provisional one computed x · s_enc / scale. The two can differ by 1 ulp before the final rounding to E2M1, which flips an element only when x sits almost exactly on an E2M1 rounding boundary.

## What I tried

1. The C++ cross-check (`xcheck.cpp`) now builds NVFP4 operands with the study's `nvfp4_quantize` and requires the study's `nvfp4_chain` to equal a generic exact-sum oracle that rounds once per 32 elements. The MX formats likewise require `mx_chain` (E4M3, E3M2, E2M3, E2M1) to equal the oracle.
2. `mx.py` (the CPU reference) and `kernel.py` (the GPU kernel) were changed to the unit's three rules: one rounding per 32 elements, the element formula above, and the host-side decode in FP32.
3. `nvfp4_semantics.py` measures how far the old and new definitions are apart on Gaussian operands with rows scaled up to 11×. It runs 20 GEMMs of 64 × 64 outputs with K = 768.

## Output

`docs/journal/data/training-nvfp4-semantics.txt`:

```
elements (A operands) differing: 0 of 983040
GEMM outputs differing:          35408 of 81920 (43.2%)
largest relative difference:     2.38e-07
```

`docs/journal/data/training-xcheck-gpu.txt` (GPU kernel against the C++ references) and `docs/journal/data/training-xcheck-cpu.txt` (CPU reference against the C++ references): PASS for every format, zero bit differences.

## Result

The emulator now matches the chip lab's NVFP4 unit bit for bit: the CPU reference and the GPU kernel both agree with `nvfp4_chain` on every element and every GEMM output in the randomized cross-check. The provisional definition quantized every element identically on this data, but 43% of its GEMM outputs were off by 1 ulp (at most 2.4 × 10⁻⁷ relative), because it rounded twice per 32 elements. That difference is far below the quantization error of FP4 itself, so it could not have changed a training result measurably. Still, a bit-exact claim needs bit-exact arithmetic. No NVFP4 training had run on the old definition.

The NVFP4 numbers stay labelled provisional in one respect: stochastic rounding, which the NVFP4 recipe applies to gradients, is not yet part of the emulator.

## Next

Re-measure NVFP4 training throughput with the new kernel, which does one three-term rounding per 32 elements instead of two two-term roundings. Then continue the queued runs: BF16 and MXFP8 first.
