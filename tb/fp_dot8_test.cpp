#include "Vfp_dot8_fp32.h"
#include "dot8_model.h"
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Vector { std::array<uint16_t, 8> a, b; uint32_t c; };

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string describe(const Vector &v) {
    std::string out = "c=" + std::to_string(v.c);
    for (unsigned k = 0; k < 8; ++k) out += " a" + std::to_string(k) + "=" + std::to_string(v.a[k]) +
                                            " b" + std::to_string(k) + "=" + std::to_string(v.b[k]);
    return out;
}

#ifndef DOT8_FORMAT
#define DOT8_FORMAT 0
#endif
constexpr Dot8Format format = DOT8_FORMAT == 0 ? dot8_bf16 : DOT8_FORMAT == 1 ? dot8_e4m3 : dot8_e5m2;
constexpr unsigned man_bits = format.mantissa, width = 1 + format.exponent + format.mantissa;
constexpr unsigned bias = (1u << (format.exponent - 1)) - 1, exp_max = (1u << format.exponent) - 1;
constexpr const char *format_name = DOT8_FORMAT == 0 ? "bf16" : DOT8_FORMAT == 1 ? "e4m3" : "e5m2";

static long double as_value(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Truncated results must stay within the documented error of the exactly rounded sum.
static void check_truncation_bound(const Vector &v, const Dot8Result &spec) {
    Dot8Result exact = dot8_exact(v.a, v.b, v.c, format);
    long double s = as_value(spec.bits), e = as_value(exact.bits);
    if (!std::isfinite(s) || !std::isfinite(e)) {
        require(spec.bits == exact.bits || (spec.flags & 4) || (exact.flags & 4), "truncated special mismatch: " + describe(v));
        return;
    }
    dot8::Terms t = dot8::terms(v.a, v.b, v.c, format);
    int top = -100000;
    for (const auto &x : t.term)
        if (x.sig && x.lsb + int(x.width) > top) top = x.lsb + int(x.width);
    long double ulp = std::ldexp(1.0L, std::max(std::ilogb(std::fabs(e) > std::fabs(s) ? e : s) - 23, -149));
    if (e == 0 && s == 0) ulp = std::ldexp(1.0L, -149);
    long double bound = 9 * std::ldexp(1.0L, top - 32) + ulp;
    require(std::fabs(s - e) <= bound, "truncation error exceeds bound: " + describe(v));
}

static uint16_t code(std::mt19937 &rng, unsigned lo, unsigned hi) {
    return uint16_t((rng() & 1) << (width - 1) | (lo + rng() % (hi - lo + 1)) << man_bits |
                    (rng() & ((1u << man_bits) - 1)));
}

static uint32_t fp32(std::mt19937 &rng, unsigned lo, unsigned hi) {
    return (rng() & 1u) << 31 | (lo + rng() % (hi - lo + 1)) << 23 | (rng() & 0x7fffffu);
}

// Zeros, ones, extreme finite values and the format's NaN and infinity encodings.
static std::vector<uint16_t> specials() {
    uint16_t sign = uint16_t(1u << (width - 1)), one = uint16_t(bias << man_bits), top = uint16_t(exp_max << man_bits);
    uint16_t max_normal = format.ieee ? uint16_t(top - 1) : uint16_t(top | ((1u << man_bits) - 2));
    std::vector<uint16_t> out{0, sign, one, uint16_t(one | sign), 1, uint16_t(1 | sign),
                              uint16_t(1u << man_bits), max_normal, uint16_t(max_normal | sign),
                              uint16_t(top | ((1u << man_bits) - 1))};
    if (format.ieee) {
        out.insert(out.end(), {top, uint16_t(top | sign), uint16_t(top | 1u), uint16_t(top | (1u << (man_bits - 1)))});
    }
    return out;
}

// Distributions: raw encodings, clustered exponents, wide spread, low magnitudes, large
// accumulators, sparse specials, signed all-zero sums, and exact cancellation.
static Vector generate(std::mt19937 &rng, unsigned kind) {
    static const std::vector<uint16_t> special = specials();
    Vector v;
    for (unsigned k = 0; k < 8; ++k) {
        switch (kind) {
        case 0: v.a[k] = uint16_t(rng() & ((1u << width) - 1)); v.b[k] = uint16_t(rng() & ((1u << width) - 1)); break;
        case 1: v.a[k] = code(rng, bias - 3, bias + 3); v.b[k] = code(rng, bias - 3, bias + 3); break;
        case 2: v.a[k] = code(rng, 1, exp_max - 1); v.b[k] = code(rng, 1, exp_max - 1); break;
        case 3: v.a[k] = code(rng, 0, bias / 2); v.b[k] = code(rng, 0, bias); break;
        case 4: v.a[k] = code(rng, exp_max - 3, exp_max - 1); v.b[k] = code(rng, bias - 2, bias + 2); break;
        case 6: v.a[k] = rng() & 1 ? uint16_t(1u << (width - 1)) : 0; v.b[k] = code(rng, 1, exp_max - 1); break;
        default:
            v.a[k] = rng() % 3 ? code(rng, bias - 3, bias + 3) : special[rng() % special.size()];
            v.b[k] = rng() % 3 ? code(rng, bias - 3, bias + 3) : special[rng() % special.size()];
        }
    }
    switch (kind) {
    case 0: v.c = rng(); break;
    case 3: v.c = rng() % 2 ? fp32(rng, 0, 4) : (rng() & 0x807fffffu); break;
    case 4: v.c = fp32(rng, 240, 254); break;
    case 6: v.c = rng() & 1 ? 0x80000000u : 0u; break;
    default: v.c = rng() % 8 ? fp32(rng, 120, 140) : (rng() % 2 ? 0x80000000u : 0u);
    }
    if (kind == 1 && rng() % 4 == 0) {
        // c cancels the first product exactly, leaving a small residue from the others.
        Vector copy = v;
        for (unsigned k = 1; k < 8; ++k) copy.a[k] = 0;
        v.c = dot8_spec(copy.a, copy.b, 0, format).bits ^ 0x80000000u;
        if (rng() % 2) for (unsigned k = 1; k < 8; ++k) v.a[k] = 0;
    }
    return v;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Vfp_dot8_fp32 dut;
        std::mt19937 rng(20261019);
        constexpr unsigned latency = 5, per_kind = 600000, kinds = 7;
        std::deque<std::pair<Vector, Dot8Result>> in_flight;
        uint64_t checked = 0, oracle_checked = 0, truncated = 0;
        std::array<unsigned, 32> flag_counts{};
        for (unsigned n = 0; n < kinds * per_kind + latency; ++n) {
            if (n < kinds * per_kind) {
                Vector v = generate(rng, n % kinds);
                Dot8Result spec = dot8_spec(v.a, v.b, v.c, format);
                if (spec.truncated) {
                    ++truncated;
                    check_truncation_bound(v, spec);
                }
                else {
                    Dot8Result exact = dot8_exact(v.a, v.b, v.c, format);
                    require(exact.bits == spec.bits && exact.flags == spec.flags,
                            "specification differs from exact oracle: " + describe(v) +
                            " spec=" + std::to_string(spec.bits) + "/" + std::to_string(spec.flags) +
                            " exact=" + std::to_string(exact.bits) + "/" + std::to_string(exact.flags));
                    ++oracle_checked;
                }
#if DOT8_FORMAT == 0
                for (unsigned k = 0; k < 8; ++k) {
                    dut.a[k / 2] = (dut.a[k / 2] & (k % 2 ? 0x0000ffffu : 0xffff0000u)) | uint32_t(v.a[k]) << (16 * (k % 2));
                    dut.b[k / 2] = (dut.b[k / 2] & (k % 2 ? 0x0000ffffu : 0xffff0000u)) | uint32_t(v.b[k]) << (16 * (k % 2));
                }
#else
                dut.a = 0;
                dut.b = 0;
                for (unsigned k = 0; k < 8; ++k) {
                    dut.a |= uint64_t(v.a[k]) << (8 * k);
                    dut.b |= uint64_t(v.b[k]) << (8 * k);
                }
#endif
                dut.c = v.c;
                in_flight.emplace_back(v, spec);
            }
            dut.clk = 0;
            dut.eval();
            dut.clk = 1;
            dut.eval();
            if (n + 1 >= latency && !in_flight.empty()) {
                auto [v, spec] = in_flight.front();
                in_flight.pop_front();
                require(dut.result == spec.bits && dut.exception_flags == spec.flags,
                        "RTL differs from specification: " + describe(v) + " got=" + std::to_string(dut.result) +
                        "/" + std::to_string(dut.exception_flags) + " expected=" + std::to_string(spec.bits) +
                        "/" + std::to_string(spec.flags));
                ++flag_counts[spec.flags];
                ++checked;
            }
        }
        require(in_flight.empty(), "pipeline drained");
        dut.final();
        std::cout << "PASS: fp_dot8_fp32 format=" << format_name << " vectors=" << checked << " oracle_exact=" << oracle_checked
                  << " truncated=" << truncated << " flags[nx,uf+nx,of+nx,nv]=" << flag_counts[1] << ","
                  << flag_counts[3] << "," << flag_counts[5] << "," << flag_counts[16] << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
