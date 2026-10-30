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
constexpr MxElement element = MX_ELEM == 0 ? MxElement::e4m3 : MxElement::e2m1;
constexpr unsigned width = MX_ELEM == 0 ? 8 : 4, exp_bits = MX_ELEM == 0 ? 4 : 2, man_bits = width - 1 - exp_bits;
constexpr const char *element_name = MX_ELEM == 0 ? "mxfp8_e4m3" : "mxfp4_e2m1";

struct Vector { std::array<uint8_t, 32> a, b; uint8_t sa, sb; uint32_t c; };

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string describe(const Vector &v) {
    std::string out = "sa=" + std::to_string(v.sa) + " sb=" + std::to_string(v.sb) + " c=" + std::to_string(v.c);
    for (unsigned k = 0; k < 32; ++k) out += " " + std::to_string(v.a[k]) + "*" + std::to_string(v.b[k]);
    return out;
}

// A finite element: E4M3 avoids its NaN code S.1111.111.
static uint8_t finite(std::mt19937 &rng) {
    uint8_t x = uint8_t(rng() & ((1u << width) - 1));
    return MX_ELEM == 0 && (x & 0x7f) == 0x7f ? uint8_t(x - 1) : x;
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
    switch (kind) {
    case 0: v.sa = uint8_t(rng()); v.sb = uint8_t(rng()); v.c = rng(); break;
    case 1: v.sa = uint8_t(117 + rng() % 21); v.sb = uint8_t(117 + rng() % 21); v.c = fp32(rng, 110, 145); break;
    case 2:
        v.sa = uint8_t(rng() % 2 ? rng() % 8 : 247 + rng() % 8);
        v.sb = uint8_t(rng() % 2 ? rng() % 8 : 247 + rng() % 8);
        v.c = rng() % 2 ? fp32(rng, 0, 8) : (rng() % 2 ? fp32(rng, 245, 254) : rng() & 0x807fffffu);
        break;
    case 3: v.sa = 127; v.sb = 127; v.c = fp32(rng, 140, 200); break;
    case 4: {
        v.sa = uint8_t(120 + rng() % 15);
        v.sb = uint8_t(120 + rng() % 15);
        v.c = 0;
        Dot8Result block = mx_dot32(v.a, v.b, v.sa, v.sb, 0, element);
        v.c = (block.bits ^ 0x80000000u) + (rng() % 3) - 1;  // leaves a small residue
        if ((v.c & 0x7f800000u) == 0x7f800000u) v.c = 0;
        break;
    }
    case 5: v.sa = 127; v.sb = 127; v.c = rng() % 2 ? 0x80000000u : 0u; break;
    default: {
        static const uint32_t specials[] = {0x7fc00000u, 0x7f800001u, 0x7f800000u, 0xff800000u, 0u, 0x80000000u, 0x00000001u};
        v.sa = uint8_t(rng() % 16 ? 127 : 255);
        v.sb = uint8_t(rng() % 16 ? 127 : 255);
        if (MX_ELEM == 0 && rng() % 4 == 0) v.a[rng() % 32] = uint8_t(0x7f | (rng() & 0x80));
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
                Dot8Result exact = mx_dot32(v.a, v.b, v.sa, v.sb, v.c, element);
                for (unsigned w = 0; w < 32 * width / 32; ++w) {
                    uint32_t wa = 0, wb = 0;
                    for (unsigned k = 0; k < 32 / width; ++k) {
                        wa |= uint32_t(v.a[w * (32 / width) + k]) << (width * k);
                        wb |= uint32_t(v.b[w * (32 / width) + k]) << (width * k);
                    }
                    dut.a[w] = wa;
                    dut.b[w] = wb;
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
                        "RTL differs from exact rounding: " + describe(v) + " got=" + std::to_string(dut.result) +
                        "/" + std::to_string(dut.exception_flags) + " expected=" + std::to_string(exact.bits) +
                        "/" + std::to_string(exact.flags));
                ++flag_counts[exact.flags];
                ++checked;
            }
        }
        dut.final();
        std::cout << "PASS: mx_dot32_fp32 element=" << element_name << " vectors=" << checked
                  << " (all correctly rounded) flags[nx,uf+nx,of+nx,nv]=" << flag_counts[1] << "," << flag_counts[3]
                  << "," << flag_counts[5] << "," << flag_counts[16] << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
