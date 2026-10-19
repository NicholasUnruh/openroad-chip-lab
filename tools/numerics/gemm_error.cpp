// Dot-product error of each study datapath against FP64 on the original, unquantized data.
// Error is |y - y64| / sum|a_i b_i|, which stays meaningful under cancellation.
// usage: build/gemm_error [samples]
#include "../../tb/dot8_model.h"
#include "../../tb/mx_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Vec = std::vector<double>;

static uint32_t bits(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
static float value(uint32_t r) { float x; std::memcpy(&x, &r, 4); return x; }

static uint16_t to_bf16(float x) {
    uint32_t raw = bits(x);
    return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}

// Nearest-even code of x in a sign/exponent/mantissa format, saturating at max_value.
static uint8_t to_small(double x, int exp_bits, int man_bits, double max_value) {
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

static double amax(const Vec &x) {
    double m = 0;
    for (double v : x) m = std::max(m, std::fabs(v));
    return m > 0 ? m : 1;
}

static float fp32_dot(const Vec &a, const Vec &b) {
    float acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc = std::fma(float(a[i]), float(b[i]), acc);
    return acc;
}

static float bf16_fma(const Vec &a, const Vec &b) {
    float acc = 0;
    for (size_t i = 0; i < a.size(); ++i)
        acc = std::fma(value(uint32_t(to_bf16(float(a[i]))) << 16), value(uint32_t(to_bf16(float(b[i]))) << 16), acc);
    return acc;
}

// Chained 8-term fused dot products; FP8 operands use per-vector amax scaling to the format max.
static float dot8_chain(const Vec &a, const Vec &b, Dot8Format f) {
    bool fp8 = f.exponent != 8;
    double max_value = f.ieee ? (f.exponent == 5 ? 57344.0 : 0) : 448.0;
    double sa = fp8 ? max_value / amax(a) : 1, sb = fp8 ? max_value / amax(b) : 1;
    uint32_t acc = 0;
    for (size_t i0 = 0; i0 < a.size(); i0 += 8) {
        std::array<uint16_t, 8> x, y;
        for (size_t k = 0; k < 8; ++k) {
            x[k] = fp8 ? to_small(a[i0 + k] * sa, f.exponent, f.mantissa, max_value) : to_bf16(float(a[i0 + k]));
            y[k] = fp8 ? to_small(b[i0 + k] * sb, f.exponent, f.mantissa, max_value) : to_bf16(float(b[i0 + k]));
        }
        acc = dot8_spec(x, y, acc, f).bits;
    }
    return float(double(value(acc)) / (sa * sb));
}

// OCP MX: each 32-element block shares 2^(floor(log2 amax) - emax_elem).
static float mx_chain(const Vec &a, const Vec &b, MxElement e) {
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

static float int8_dot(const Vec &a, const Vec &b) {
    double sa = 127 / amax(a), sb = 127 / amax(b);
    int64_t acc = 0;
    for (size_t i = 0; i < a.size(); ++i) acc += int64_t(std::nearbyint(a[i] * sa)) * int64_t(std::nearbyint(b[i] * sb));
    return float(double(acc) / (sa * sb));
}

int main(int argc, char **argv) {
    unsigned samples = argc > 1 ? unsigned(std::stoul(argv[1])) : 2000;
    std::vector<std::pair<std::string, std::function<float(const Vec &, const Vec &)>>> paths = {
        {"fp32_fma", fp32_dot},
        {"bf16_fma", bf16_fma},
        {"bf16_dot8", [](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_bf16); }},
        {"e4m3_dot8", [](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_e4m3); }},
        {"e5m2_dot8", [](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_e5m2); }},
        {"mxfp8", [](const Vec &a, const Vec &b) { return mx_chain(a, b, MxElement::e4m3); }},
        {"mxfp4", [](const Vec &a, const Vec &b) { return mx_chain(a, b, MxElement::e2m1); }},
        {"int8", int8_dot},
    };
    std::cout << "distribution K datapath median_err p99_err\n";
    for (const std::string dist : {"gaussian", "laplace", "outliers"}) {
        for (unsigned K : {256u, 4096u}) {
            std::mt19937_64 rng(K * 7919 + dist.size());
            std::normal_distribution<double> normal(0, 1);
            std::exponential_distribution<double> expo(1);
            std::vector<std::vector<double>> errors(paths.size());
            for (unsigned s = 0; s < samples; ++s) {
                Vec a(K), b(K);
                for (unsigned i = 0; i < K; ++i) {
                    if (dist == "laplace") {
                        a[i] = (rng() & 1 ? 1 : -1) * expo(rng);
                        b[i] = (rng() & 1 ? 1 : -1) * expo(rng);
                    } else {
                        a[i] = normal(rng);
                        b[i] = normal(rng);
                        if (dist == "outliers" && rng() % 100 == 0) a[i] *= 100;
                    }
                }
                double exact = 0, magnitude = 0;
                for (unsigned i = 0; i < K; ++i) {
                    exact += a[i] * b[i];
                    magnitude += std::fabs(a[i] * b[i]);
                }
                for (size_t p = 0; p < paths.size(); ++p)
                    errors[p].push_back(std::fabs(double(paths[p].second(a, b)) - exact) / magnitude);
            }
            for (size_t p = 0; p < paths.size(); ++p) {
                auto &e = errors[p];
                std::sort(e.begin(), e.end());
                std::cout << dist << ' ' << K << ' ' << paths[p].first << ' ' << e[e.size() / 2] << ' '
                          << e[e.size() * 99 / 100] << '\n';
            }
        }
    }
    return 0;
}
