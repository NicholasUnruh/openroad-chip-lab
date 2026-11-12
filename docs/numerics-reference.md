# Bit-exact format references

The C++ references below are the ground truth for every microscaled format in this repo: the RTL (`rtl/mx_dot32_fp32.v`) must match them bit for bit (`make sim-mx`, 2.1 million vectors per element type), and so must any emulation that claims to model this hardware. The signatures are stable; additions get new names instead of changed meanings.

## Unit references (`tb/mx_model.h`)

| Function | Meaning |
| --- | --- |
| `enum class MxElement { e4m3 = 0, e2m1 = 1, e5m2 = 2, e3m2 = 3, e2m3 = 4 }` | OCP MX element types; the value is the RTL `ELEM` parameter (NVFP4 is `ELEM = 5`) |
| `MxFormat mx_format(MxElement)` | Field widths, largest finite value and its exponent `emax` |
| `dot8::Operand mx_decode(uint8_t code, MxElement)` | One element as sign, significand and LSB exponent, with specials |
| `Dot8Result mx_dot32(a[32], b[32], uint8_t scale_a, uint8_t scale_b, uint32_t c, MxElement)` | Correctly rounded `c + 2^(sa+sb-254) · Σ aᵢbᵢ`, one FP32 rounding (RNE) per call |
| `Dot8Result nvfp4_dot32(a[32], b[32], std::array<uint8_t,2> scale_a, scale_b, uint32_t c)` | Correctly rounded `c + Σ_blk sa_blk·sb_blk · Σ_{16} aᵢbᵢ`: E2M1 elements, one E4M3 scale per 16 elements, both 16-element blocks summed exactly before one rounding |

Specials: an E8M0 scale of 0xFF, an E4M3 NaN element (S.1111.111) or an E4M3 NVFP4 scale whose low seven bits are 0x7F gives the canonical NaN 0x7FC00000. E5M2 has IEEE infinities and NaNs; signaling NaNs, infinity times zero and opposite infinities raise invalid (flag 16). E2M1, E3M2 and E2M3 have no specials. NVFP4 scale sign bits are ignored. Flags: 1 inexact, 2 underflow, 4 overflow, 16 invalid.

## Quantizers and GEMM chains (`tools/numerics/datapaths.h`)

| Function | Meaning |
| --- | --- |
| `uint8_t to_small(double x, int exp_bits, int man_bits, double max_value, int sr_random = -1)` | Saturating encode to a small float. RNE by default; with `sr_random` in 0..65535, stochastic rounding (below) |
| `uint8_t mx_quantize(const double *block, MxElement, std::array<uint8_t,32> &codes, const uint16_t *sr_random = nullptr)` | OCP scale `2^(floor(log2 amax) − emax)`, clamped to E8M0; returns the scale code |
| `float mx_chain(const Vec &a, const Vec &b, MxElement)` | K-long dot product: `mx_dot32` per 32-element block, chained from +0 |
| `float mx_chain_sr(a, b, MxElement, const std::function<uint16_t()> &random_b)` | Same, with b's elements stochastically rounded, one draw per element in order |
| `float nvfp4_encode_scale(double tensor_amax)` | `s_enc = 6·448 / amax` in FP32 |
| `uint8_t nvfp4_quantize(const double *block, float s_enc, std::array<uint8_t,16> &codes, const uint16_t *sr_random = nullptr)` | 16 elements; returns the E4M3 block scale code |
| `float nvfp4_chain(a, b, double amax_a, double amax_b, const std::function<uint16_t()> *random_b = nullptr)` | `nvfp4_dot32` per 32 elements chained from +0, then `acc · ((1/s_enc_a)·(1/s_enc_b))` once, in FP32 |

**NVFP4** follows "Pretraining Large Language Models with NVFP4" (NVIDIA, arXiv:2509.25149, Section 2 and Appendix B), every step in FP32: `s_enc = 6·448/amax_tensor`, `s_dec = 1/s_enc`; block scale `e4m3(amax_block/6 · s_enc)` (RNE, saturating at 448); element `e2m1(x · (1/(fp32(scale) · s_dec)))`; a zero block scale encodes its elements as zero. The hardware unit takes 32 elements (two scale blocks) per call and rounds once per call, not once per 16-element block; the per-tensor decode is a host-side FP32 multiply on the GEMM output.

**Stochastic rounding.** For |x| clipped to the format's maximum, let `step` be the code spacing at |x| (the subnormal spacing below the smallest normal), `T = floor(|x|/step · 2^16)` and `R` the 16-bit random input; the magnitude is `floor((T + R) / 2^16) · step`, saturated at the maximum, with x's sign. The probability of rounding up equals the discarded fraction to 16 bits, so the result is unbiased up to 2⁻¹⁶ of a step; clipping at the maximum is not.
