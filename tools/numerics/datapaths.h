// Bit-exact dot-product datapaths of the numerics study, shared by gemm_error and train_mlp.
#pragma once
#include "../../tb/dot8_model.h"
#include "../../tb/mx_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

using Vec = std::vector<double>;

inline uint32_t bits(float x) {
    uint32_t r;
    std::memcpy(&r, &x, 4);
    return r;
}
inline float value(uint32_t r) {
    float x;
    std::memcpy(&x, &r, 4);
    return x;
}

inline uint16_t to_bf16(float x) {
    uint32_t raw = bits(x);
    return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}

// Code of x in a sign/exponent/mantissa format, saturating at max_value. Rounds to nearest even,
// or stochastically when sr_random (a uniform 16-bit integer R) is given: with step the format's
// spacing at |x|, the magnitude truncated to 16 bits below step rounds up when its discarded
// fraction plus R / 2^16 reaches one step, so it rounds up with probability = that fraction.
inline uint8_t to_small(double x, int exp_bits, int man_bits, double max_value,
                        int sr_random = -1) {
    unsigned sign = std::signbit(x) ? 1u : 0u;
    double m = std::min(std::fabs(x), max_value);
    int bias = (1 << (exp_bits - 1)) - 1;
    if (m == 0) return uint8_t(sign << (exp_bits + man_bits));
    int e = std::max(int(std::floor(std::log2(m))), 1 - bias);
    double step = std::ldexp(1.0, e - man_bits);
    double rounded = sr_random < 0
                         ? std::nearbyint(m / step)
                         : std::floor((std::floor(std::ldexp(m / step, 16)) + sr_random) / 65536.0);
    double q = std::min(rounded * step, max_value);
    if (q >= std::ldexp(1.0, e + 1)) ++e;
    unsigned field = q >= std::ldexp(1.0, 1 - bias) ? unsigned(e + bias) : 0u;
    unsigned frac =
        unsigned(std::nearbyint(q / std::ldexp(1.0, (field ? int(field) : 1) - bias - man_bits))) -
        (field ? 1u << man_bits : 0u);
    return uint8_t(sign << (exp_bits + man_bits) | field << man_bits | frac);
}

inline double amax(const Vec &x) {
    double m = 0;
    for (double v : x) m = std::max(m, std::fabs(v));
    return m > 0 ? m : 1;
}

inline float fp32_dot(const Vec &a, const Vec &b) {
    float acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc = std::fma(float(a[i]), float(b[i]), acc);
    return acc;
}

inline float bf16_fma(const Vec &a, const Vec &b) {
    float acc = 0;
    for (size_t i = 0; i < a.size(); ++i)
        acc = std::fma(value(uint32_t(to_bf16(float(a[i]))) << 16),
                       value(uint32_t(to_bf16(float(b[i]))) << 16), acc);
    return acc;
}

// Chained 8-term fused dot products with a `frame`-bit alignment window. FP8 operands are scaled
// so amax_a / amax_b (per vector or per tensor, supplied by the caller) map to the format maximum.
inline float dot8_chain(const Vec &a, const Vec &b, Dot8Format f, double amax_a = 0,
                        double amax_b = 0, int frame = 32) {
    bool fp8 = f.exponent != 8;
    double max_value = f.ieee ? (f.exponent == 5 ? 57344.0 : 0) : 448.0;
    double sa = fp8 ? max_value / (amax_a ? amax_a : amax(a)) : 1;
    double sb = fp8 ? max_value / (amax_b ? amax_b : amax(b)) : 1;
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 8) {
        std::array<uint16_t, 8> x, y;
        for (size_t k = 0; k < 8; ++k) {
            x[k] = fp8 ? to_small(a[i0 + k] * sa, f.exponent, f.mantissa, max_value)
                       : to_bf16(float(a[i0 + k]));
            y[k] = fp8 ? to_small(b[i0 + k] * sb, f.exponent, f.mantissa, max_value)
                       : to_bf16(float(b[i0 + k]));
        }
        acc = dot8_spec(x, y, acc, f, frame).bits;
    }
    return float(double(value(acc)) / (sa * sb));
}

// OCP MX: a 32-element block shares 2^(floor(log2 amax) - emax_elem); returns the E8M0 scale.
// sr_random, when given, holds one 16-bit random integer per element for stochastic rounding.
inline uint8_t mx_quantize(const double *block, MxElement e, std::array<uint8_t, 32> &codes,
                           const uint16_t *sr_random = nullptr) {
    MxFormat f = mx_format(e);
    double m = 0;
    for (size_t k = 0; k < 32; ++k) m = std::max(m, std::fabs(block[k]));
    int shared = m > 0 ? int(std::floor(std::log2(m))) - f.emax : -127;
    shared = std::clamp(shared, -127, 127);
    for (size_t k = 0; k < 32; ++k)
        codes[k] = to_small(std::ldexp(block[k], -shared), int(f.exponent), int(f.mantissa),
                            f.max_value, sr_random ? sr_random[k] : -1);
    return uint8_t(shared + 127);
}

inline float mx_chain(const Vec &a, const Vec &b, MxElement e) {
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 32) {
        std::array<uint8_t, 32> x, y;
        uint8_t sa = mx_quantize(&a[i0], e, x), sb = mx_quantize(&b[i0], e, y);
        acc = mx_dot32(x, y, sa, sb, acc, e).bits;
    }
    return value(acc);
}

// NVFP4 quantization ("Pretraining Large Language Models with NVFP4", arXiv:2509.25149,
// Appendix B), every step in FP32 arithmetic, round to nearest even:
//   s_enc = 6 * 448 / amax_tensor, s_dec = 1 / s_enc                         (B.1)
//   block scale = e4m3(amax_block / 6 * s_enc), saturating at 448            (B.2)
//   element = e2m1(x * (1 / (fp32(block scale) * s_dec)))                    (B.3)
// A zero block scale encodes its elements as zero. sr_random, when given, rounds the elements
// stochastically (to_small). Returns the E4M3 block scale code.
inline float nvfp4_encode_scale(double tensor_amax) {
    return 6.0f * 448.0f / float(tensor_amax > 0 ? tensor_amax : 1);
}

inline uint8_t nvfp4_quantize(const double *block, float s_enc, std::array<uint8_t, 16> &codes,
                              const uint16_t *sr_random = nullptr) {
    float block_amax = 0;
    for (size_t k = 0; k < 16; ++k) block_amax = std::max(block_amax, std::fabs(float(block[k])));
    uint8_t scale = to_small(double(block_amax / 6.0f * s_enc), 4, 3, 448.0);
    dot8::Operand s = dot8::decode_input(scale, dot8_e4m3);
    float scale_value = std::ldexp(float(s.sig), s.lsb), s_dec = 1.0f / s_enc;
    for (size_t k = 0; k < 16; ++k)
        codes[k] = scale_value == 0
                       ? 0
                       : to_small(double(float(block[k]) * (1.0f / (scale_value * s_dec))), 2, 1,
                                  6.0, sr_random ? sr_random[k] : -1);
    return scale;
}

// Chained NVFP4 unit calls over 32-element K chunks from +0, then the per-tensor decode scales
// applied once in FP32: result = acc * (s_dec_a * s_dec_b). random_b, when given, supplies 16-bit
// random integers to round b's elements stochastically (one per element, in order).
inline float nvfp4_chain(const Vec &a, const Vec &b, double amax_a, double amax_b,
                         const std::function<uint16_t()> *random_b = nullptr) {
    float enc_a = nvfp4_encode_scale(amax_a), enc_b = nvfp4_encode_scale(amax_b);
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 32) {
        std::array<uint8_t, 32> x, y;
        std::array<uint8_t, 2> sa, sb;
        for (unsigned blk = 0; blk < 2; ++blk) {
            std::array<uint8_t, 16> cx, cy;
            uint16_t r[16];
            if (random_b)
                for (uint16_t &v : r) v = (*random_b)();
            sa[blk] = nvfp4_quantize(&a[i0 + 16 * blk], enc_a, cx);
            sb[blk] = nvfp4_quantize(&b[i0 + 16 * blk], enc_b, cy, random_b ? r : nullptr);
            std::copy(cx.begin(), cx.end(), x.begin() + 16 * blk);
            std::copy(cy.begin(), cy.end(), y.begin() + 16 * blk);
        }
        acc = nvfp4_dot32(x, y, sa, sb, acc).bits;
    }
    return value(acc) * ((1.0f / enc_a) * (1.0f / enc_b));
}

// MX chain with b's elements rounded stochastically from random_b.
inline float mx_chain_sr(const Vec &a, const Vec &b, MxElement e,
                         const std::function<uint16_t()> &random_b) {
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 32) {
        std::array<uint8_t, 32> x, y;
        uint16_t r[32];
        for (uint16_t &v : r) v = random_b();
        uint8_t sa = mx_quantize(&a[i0], e, x), sb = mx_quantize(&b[i0], e, y, r);
        acc = mx_dot32(x, y, sa, sb, acc, e).bits;
    }
    return value(acc);
}

inline float int8_dot(const Vec &a, const Vec &b, double amax_a = 0, double amax_b = 0) {
    double sa = 127 / (amax_a ? amax_a : amax(a)), sb = 127 / (amax_b ? amax_b : amax(b));
    int64_t acc = 0;
    for (size_t i = 0; i < a.size(); ++i)
        acc += int64_t(std::nearbyint(a[i] * sa)) * int64_t(std::nearbyint(b[i] * sb));
    return float(double(acc) / (sa * sb));
}
