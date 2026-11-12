#include "Vmx_dot32_fp32.h"
#include "mx_model.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#ifndef MX_ELEM
#define MX_ELEM 0
#endif
// MX_ELEM 5 is NVFP4: E2M1 elements with two E4M3 block scales per operand.
constexpr bool nvfp4 = MX_ELEM == 5;
constexpr MxElement element = nvfp4 ? MxElement::e2m1 : MxElement(MX_ELEM);
const MxFormat format = mx_format(element);
const unsigned width = 1 + format.exponent + format.mantissa;
constexpr const char *element_names[] = {"mxfp8_e4m3", "mxfp4_e2m1", "mxfp8_e5m2",
                                         "mxfp6_e3m2", "mxfp6_e2m3", "nvfp4_e2m1"};
constexpr const char *element_name = element_names[MX_ELEM];

struct Vector {
    std::array<uint8_t, 32> a, b;
    uint16_t sa, sb;  // one E8M0 scale, or two E4M3 block scales for NVFP4
    uint32_t c;
};

static Dot8Result reference(const Vector &v) {
    if (nvfp4)
        return nvfp4_dot32(v.a, v.b, {uint8_t(v.sa), uint8_t(v.sa >> 8)},
                           {uint8_t(v.sb), uint8_t(v.sb >> 8)}, v.c);
    return mx_dot32(v.a, v.b, uint8_t(v.sa), uint8_t(v.sb), v.c, element);
}

// One operand's scales for a distribution. MX: an E8M0 exponent; NVFP4: two E4M3 block scales,
// with codes chosen to cover the same cases (raw bytes, realistic, subnormal or near-max, unit,
// NaN).
static uint16_t scale(std::mt19937 &rng, unsigned kind) {
    auto one = [&]() -> unsigned {
        switch (kind) {
            case 0:
                return rng() & 0xff;
            case 1:
            case 4:
                return nvfp4 ? 0x20 + rng() % 0x5f
                             : (kind == 1 ? 117 + rng() % 21 : 120 + rng() % 15);
            case 2:
                return nvfp4 ? (rng() % 2 ? rng() % 8 : 0x78 + rng() % 7)
                             : (rng() % 2 ? rng() % 8 : 247 + rng() % 8);
            case 3:
            case 5:
                return nvfp4 ? 0x38 : 127;
            default:
                return rng() % 16 ? (nvfp4 ? 0x38 : 127) : (nvfp4 ? 0x7f | (rng() & 0x80) : 255);
        }
    };
    unsigned low = one();
    return uint16_t(nvfp4 ? low | one() << 8 : low);
}

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string describe(const Vector &v) {
    std::string out =
        "sa=" + std::to_string(v.sa) + " sb=" + std::to_string(v.sb) + " c=" + std::to_string(v.c);
    for (unsigned k = 0; k < 32; ++k)
        out += " " + std::to_string(v.a[k]) + "*" + std::to_string(v.b[k]);
    return out;
}

// A finite element: E4M3 avoids its NaN code S.1111.111, E5M2 its all-ones exponent.
static uint8_t finite(std::mt19937 &rng) {
    uint8_t x = uint8_t(rng() & ((1u << width) - 1));
    if (element == MxElement::e4m3 && (x & 0x7f) == 0x7f) return uint8_t(x - 1);
    if (element == MxElement::e5m2 && (x & 0x7c) == 0x7c) return uint8_t(x & ~0x04u);
    return x;
}

static uint32_t fp32(std::mt19937 &rng, unsigned lo, unsigned hi) {
    return (rng() & 1u) << 31 | (lo + rng() % (hi - lo + 1)) << 23 | (rng() & 0x7fffffu);
}

// Distributions: raw codes, realistic scales, extreme scales, c-dominant sums, cancellation,
// signed zeros, and NaN scales/elements with special accumulators.
static Vector generate(std::mt19937 &rng, unsigned kind) {
    Vector v;
    for (unsigned k = 0; k < 32; ++k) {
        v.a[k] = kind == 0 ? uint8_t(rng() & ((1u << width) - 1)) : finite(rng);
        v.b[k] = kind == 0 ? uint8_t(rng() & ((1u << width) - 1)) : finite(rng);
        if (kind == 5) v.a[k] &= uint8_t(1u << (width - 1));  // signed zeros
    }
    v.sa = scale(rng, kind);
    v.sb = scale(rng, kind);
    switch (kind) {
        case 0:
            v.c = rng();
            break;
        case 1:
            v.c = fp32(rng, 110, 145);
            break;
        case 2:
            v.c = rng() % 2 ? fp32(rng, 0, 8)
                            : (rng() % 2 ? fp32(rng, 245, 254) : rng() & 0x807fffffu);
            break;
        case 3:
            v.c = fp32(rng, 140, 200);
            break;
        case 4: {
            v.c = 0;
            Dot8Result block = reference(v);
            v.c = (block.bits ^ 0x80000000u) + (rng() % 3) - 1;  // leaves a small residue
            if ((v.c & 0x7f800000u) == 0x7f800000u) v.c = 0;
            break;
        }
        case 5:
            v.c = rng() % 2 ? 0x80000000u : 0u;
            break;
        default: {
            static const uint32_t specials[] = {0x7fc00000u, 0x7f800001u, 0x7f800000u, 0xff800000u,
                                                0u,          0x80000000u, 0x00000001u};
            if (element == MxElement::e4m3 && rng() % 4 == 0)
                v.a[rng() % 32] = uint8_t(0x7f | (rng() & 0x80));
            // E5M2: infinities, quiet and signaling NaNs, and zeros to multiply them with.
            if (element == MxElement::e5m2)
                for (unsigned n = rng() % 3; n > 0; --n) {
                    static const uint8_t codes[] = {0x7c, 0xfc, 0x7e, 0x7d, 0x00, 0x80};
                    (rng() & 1 ? v.a : v.b)[rng() % 32] = codes[rng() % 6];
                }
            v.c = rng() % 2 ? specials[rng() % 7] : fp32(rng, 120, 135);
        }
    }
    return v;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Vmx_dot32_fp32 dut;
        std::mt19937 rng(20261019);
        constexpr unsigned latency = 7, per_kind = 300000, kinds = 7;
        std::deque<std::pair<Vector, Dot8Result>> in_flight;
        std::array<unsigned, 32> flag_counts{};
        unsigned checked = 0;
        for (unsigned n = 0; n < kinds * per_kind + latency; ++n) {
            if (n < kinds * per_kind) {
                Vector v = generate(rng, n % kinds);
                Dot8Result exact = reference(v);
                for (unsigned w = 0; w < width; ++w) dut.a[w] = dut.b[w] = 0;
                for (unsigned bit = 0; bit < 32 * width; ++bit) {
                    dut.a[bit / 32] |= uint32_t(v.a[bit / width] >> (bit % width) & 1)
                                       << (bit % 32);
                    dut.b[bit / 32] |= uint32_t(v.b[bit / width] >> (bit % width) & 1)
                                       << (bit % 32);
                }
                dut.scale_a = v.sa;
                dut.scale_b = v.sb;
                dut.c = v.c;
                in_flight.emplace_back(v, exact);
            }
            dut.clk = 0;
            dut.eval();
            dut.clk = 1;
            dut.eval();
            if (n + 1 >= latency && !in_flight.empty()) {
                auto [v, exact] = in_flight.front();
                in_flight.pop_front();
                require(dut.result == exact.bits && dut.exception_flags == exact.flags,
                        "RTL differs from exact rounding: " + describe(v) + " got=" +
                            std::to_string(dut.result) + "/" + std::to_string(dut.exception_flags) +
                            " expected=" + std::to_string(exact.bits) + "/" +
                            std::to_string(exact.flags));
                ++flag_counts[exact.flags];
                ++checked;
            }
        }
        dut.final();
        std::cout << "PASS: mx_dot32_fp32 element=" << element_name << " vectors=" << checked
                  << " (all correctly rounded) flags[nx,uf+nx,of+nx,nv]=" << flag_counts[1] << ","
                  << flag_counts[3] << "," << flag_counts[5] << "," << flag_counts[16] << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
