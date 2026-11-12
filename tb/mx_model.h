// Reference for the OCP MX block dot product (docs/numerics-study.md): correctly rounded
// c + 2^(sa + sb - 254) * sum of 32 element products, from the exact-sum oracle.
#pragma once
#include "dot8_model.h"
#include <array>
#include <cstdint>

// Element formats: OCP FP8 E4M3 (NaN at S.1111.111, no infinities) and FP4 E2M1 (no specials).
enum class MxElement { e4m3, e2m1 };

inline dot8::Operand mx_decode(uint8_t raw, MxElement e) {
    if (e == MxElement::e4m3) return dot8::decode_input(raw, dot8_e4m3);
    dot8::Operand x = dot8::decode(raw, 1, 2, 1);
    x.inf = x.nan = x.snan = false;
    return x;
}

inline Dot8Result mx_dot32(const std::array<uint8_t, 32> &a, const std::array<uint8_t, 32> &b,
                           uint8_t scale_a, uint8_t scale_b, uint32_t c, MxElement e) {
    using namespace dot8;
    std::array<Term, 33> term;
    bool nan = scale_a == 0xff || scale_b == 0xff, all_zero = true, all_negative = true;
    int scale = int(scale_a) + int(scale_b) - 254;
    for (unsigned k = 0; k < 32; ++k) {
        Operand x = mx_decode(a[k], e), y = mx_decode(b[k], e);
        nan |= x.nan || y.nan;
        bool sign = x.sign != y.sign;
        term[k] = {sign, x.sig * y.sig, x.lsb + y.lsb + scale, 0};
        all_zero &= term[k].sig == 0;
        all_negative &= sign;
    }
    Operand z = decode(c, 23, 8, 127);
    if (nan || z.nan) return {canonical_nan, z.snan ? 16u : 0u, false};
    if (z.inf) return {z.sign ? 0xff800000u : 0x7f800000u, 0, false};
    term[32] = {z.sign, z.sig, z.lsb, 0};
    return exact_round(term.data(), term.data() + term.size(), all_zero && z.sig == 0,
                       all_negative && z.sign);
}
