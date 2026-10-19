#include "Vbf16_dot8_fp32.h"
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

static long double as_value(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Truncated results must stay within the documented error of the exactly rounded sum.
static void check_truncation_bound(const Vector &v, const Dot8Result &spec) {
    Dot8Result exact = dot8_exact(v.a, v.b, v.c);
    long double s = as_value(spec.bits), e = as_value(exact.bits);
    if (!std::isfinite(s) || !std::isfinite(e)) {
        require(spec.bits == exact.bits || (spec.flags & 4) || (exact.flags & 4), "truncated special mismatch: " + describe(v));
        return;
    }
    dot8::Terms t = dot8::terms(v.a, v.b, v.c);
    int top = -100000;
    for (const auto &x : t.term)
        if (x.sig && x.lsb + int(x.width) > top) top = x.lsb + int(x.width);
    long double ulp = std::ldexp(1.0L, std::max(std::ilogb(std::fabs(e) > std::fabs(s) ? e : s) - 23, -149));
    if (e == 0 && s == 0) ulp = std::ldexp(1.0L, -149);
    long double bound = 9 * std::ldexp(1.0L, top - 32) + ulp;
    require(std::fabs(s - e) <= bound, "truncation error exceeds bound: " + describe(v));
}

static uint16_t bf16(std::mt19937 &rng, unsigned lo, unsigned hi) {
    return uint16_t((rng() & 1) << 15 | (lo + rng() % (hi - lo + 1)) << 7 | (rng() & 0x7f));
}

static uint32_t fp32(std::mt19937 &rng, unsigned lo, unsigned hi) {
    return (rng() & 1u) << 31 | (lo + rng() % (hi - lo + 1)) << 23 | (rng() & 0x7fffffu);
}

// Distributions: raw encodings, clustered exponents, wide spread, FP32 subnormal range,
// overflow range, exact cancellation, sparse zeros/specials and signed all-zero sums.
static Vector generate(std::mt19937 &rng, unsigned kind) {
    Vector v;
    for (unsigned k = 0; k < 8; ++k) {
        switch (kind) {
        case 0: v.a[k] = uint16_t(rng()); v.b[k] = uint16_t(rng()); break;
        case 1: v.a[k] = bf16(rng, 124, 130); v.b[k] = bf16(rng, 124, 130); break;
        case 2: v.a[k] = bf16(rng, 90, 170); v.b[k] = bf16(rng, 90, 170); break;
        case 3: v.a[k] = bf16(rng, 0, 70); v.b[k] = bf16(rng, 40, 75); break;
        case 4: v.a[k] = bf16(rng, 230, 254); v.b[k] = bf16(rng, 120, 140); break;
        case 6: v.a[k] = rng() & 1 ? 0x8000 : 0; v.b[k] = bf16(rng, 1, 254); break;
        default: {
            static const uint16_t specials[] = {0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x7f81, 0x0001, 0x8001,
                                                0x3f80, 0xbf80, 0x7f7f, 0x0080};
            v.a[k] = rng() % 3 ? bf16(rng, 120, 134) : specials[rng() % 12];
            v.b[k] = rng() % 3 ? bf16(rng, 120, 134) : specials[rng() % 12];
        }
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
        copy.c = 0;
        v.c = dot8_spec(copy.a, copy.b, 0).bits ^ 0x80000000u;
        if (rng() % 2) for (unsigned k = 1; k < 8; ++k) v.a[k] = 0;
    }
    return v;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Vbf16_dot8_fp32 dut;
        std::mt19937 rng(20261019);
        constexpr unsigned latency = 4, per_kind = 600000, kinds = 7;
        std::deque<std::pair<Vector, Dot8Result>> in_flight;
        uint64_t checked = 0, oracle_checked = 0, truncated = 0;
        std::array<unsigned, 32> flag_counts{};
        for (unsigned n = 0; n < kinds * per_kind + latency; ++n) {
            if (n < kinds * per_kind) {
                Vector v = generate(rng, n % kinds);
                Dot8Result spec = dot8_spec(v.a, v.b, v.c);
                if (spec.truncated) {
                    ++truncated;
                    check_truncation_bound(v, spec);
                }
                else {
                    Dot8Result exact = dot8_exact(v.a, v.b, v.c);
                    require(exact.bits == spec.bits && exact.flags == spec.flags,
                            "specification differs from exact oracle: " + describe(v) +
                            " spec=" + std::to_string(spec.bits) + "/" + std::to_string(spec.flags) +
                            " exact=" + std::to_string(exact.bits) + "/" + std::to_string(exact.flags));
                    ++oracle_checked;
                }
                for (unsigned k = 0; k < 8; ++k) {
                    dut.a[k / 2] = (dut.a[k / 2] & (k % 2 ? 0x0000ffffu : 0xffff0000u)) | uint32_t(v.a[k]) << (16 * (k % 2));
                    dut.b[k / 2] = (dut.b[k / 2] & (k % 2 ? 0x0000ffffu : 0xffff0000u)) | uint32_t(v.b[k]) << (16 * (k % 2));
                }
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
        std::cout << "PASS: bf16_dot8_fp32 vectors=" << checked << " oracle_exact=" << oracle_checked
                  << " truncated=" << truncated << " flags[nx,uf+nx,of+nx,nv]=" << flag_counts[1] << ","
                  << flag_counts[3] << "," << flag_counts[5] << "," << flag_counts[16] << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
