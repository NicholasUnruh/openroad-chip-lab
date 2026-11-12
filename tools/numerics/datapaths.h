// Bit-exact dot-product datapaths of the numerics study, shared by gemm_error and train_mlp.
#pragma once
#include "../../tb/dot8_model.h"
#include "../../tb/mx_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using Vec = std::vector<double>;

inline uint32_t bits(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
inline float value(uint32_t r) { float x; std::memcpy(&x, &r, 4); return x; }

inline uint16_t to_bf16(float x) {
    uint32_t raw = bits(x);
    return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}

// Nearest-even code of x in a sign/exponent/mantissa format, saturating at max_value.
inline uint8_t to_small(double x, int exp_bits, int man_bits, double max_value) {
    unsigned sign = std::signbit(x) ? 1u : 0u;
    double m = std::min(std::fabs(x), max_value);
    int bias = (1 << (exp_bits - 1)) - 1;
    if (m == 0) return uint8_t(sign << (exp_bits + man_bits));
    int e = std::max(int(std::floor(std::log2(m))), 1 - bias);
    double step = std::ldexp(1.0, e - man_bits);
    double q = std::min(std::nearbyint(m / step) * step, max_value);
    if (q >= std::ldexp(1.0, e + 1)) ++e;
    unsigned field = q >= std::ldexp(1.0, 1 - bias) ? unsigned(e + bias) : 0u;
    unsigned frac = unsigned(std::nearbyint(q / std::ldexp(1.0, (field ? int(field) : 1) - bias - man_bits))) -
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
        acc = std::fma(value(uint32_t(to_bf16(float(a[i]))) << 16), value(uint32_t(to_bf16(float(b[i]))) << 16), acc);
    return acc;
}

// Chained 8-term fused dot products with a `frame`-bit alignment window. FP8 operands are scaled
// so amax_a / amax_b (per vector or per tensor, supplied by the caller) map to the format maximum.
inline float dot8_chain(const Vec &a, const Vec &b, Dot8Format f, double amax_a = 0, double amax_b = 0,
                        int frame = 32) {
    bool fp8 = f.exponent != 8;
    double max_value = f.ieee ? (f.exponent == 5 ? 57344.0 : 0) : 448.0;
    double sa = fp8 ? max_value / (amax_a ? amax_a : amax(a)) : 1;
    double sb = fp8 ? max_value / (amax_b ? amax_b : amax(b)) : 1;
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 8) {
        std::array<uint16_t, 8> x, y;
        for (size_t k = 0; k < 8; ++k) {
            x[k] = fp8 ? to_small(a[i0 + k] * sa, f.exponent, f.mantissa, max_value) : to_bf16(float(a[i0 + k]));
            y[k] = fp8 ? to_small(b[i0 + k] * sb, f.exponent, f.mantissa, max_value) : to_bf16(float(b[i0 + k]));
        }
        acc = dot8_spec(x, y, acc, f, frame).bits;
    }
    return float(double(value(acc)) / (sa * sb));
}

// OCP MX: each 32-element block shares 2^(floor(log2 amax) - emax_elem).
inline float mx_chain(const Vec &a, const Vec &b, MxElement e) {
    int emax = e == MxElement::e4m3 ? 8 : 2;
    double max_value = e == MxElement::e4m3 ? 448.0 : 6.0;
    int exp_bits = e == MxElement::e4m3 ? 4 : 2, man_bits = e == MxElement::e4m3 ? 3 : 1;
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 32) {
        std::array<uint8_t, 32> x, y;
        uint8_t scale[2];
        const Vec *src[2] = {&a, &b};
        std::array<uint8_t, 32> *dst[2] = {&x, &y};
        for (int s = 0; s < 2; ++s) {
            double m = 0;
            for (size_t k = 0; k < 32; ++k) m = std::max(m, std::fabs((*src[s])[i0 + k]));
            int shared = m > 0 ? int(std::floor(std::log2(m))) - emax : -127;
            shared = std::clamp(shared, -127, 127);
            scale[s] = uint8_t(shared + 127);
            for (size_t k = 0; k < 32; ++k)
                (*dst[s])[k] = to_small(std::ldexp((*src[s])[i0 + k], -shared), exp_bits, man_bits, max_value);
        }
        acc = mx_dot32(x, y, scale[0], scale[1], acc, e).bits;
    }
    return value(acc);
}

inline float int8_dot(const Vec &a, const Vec &b, double amax_a = 0, double amax_b = 0) {
    double sa = 127 / (amax_a ? amax_a : amax(a)), sb = 127 / (amax_b ? amax_b : amax(b));
    int64_t acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc += int64_t(std::nearbyint(a[i] * sa)) * int64_t(std::nearbyint(b[i] * sb));
    return float(double(acc) / (sa * sb));
}

